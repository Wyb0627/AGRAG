import json
import re
from dataclasses import dataclass
from typing import Dict, Any, List, TypedDict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from ..prompts import PromptTemplateManager
from ..utils.logging_utils import get_logger
from ..utils.llm_utils import fix_broken_generated_json, filter_invalid_triples
from ..utils.misc_utils import TripleRawOutput, NerRawOutput
from ..llm.openai_gpt import CacheOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer

logger = get_logger(__name__)


class ChunkInfo(TypedDict):
    num_tokens: int
    content: str
    chunk_order: List[Tuple]
    full_doc_ids: List[str]


@dataclass
class LLMInput:
    chunk_id: str
    input_message: List[Dict]


def _extract_ner_from_response(real_response):
    pattern = r'\{[^{}]*"named_entities"\s*:\s*\[[^\]]*\][^{}]*\}'
    match = re.search(pattern, real_response, re.DOTALL)
    return eval(match.group())["named_entities"]


class OpenIE:
    def __init__(self, llm_model: CacheOpenAI, hipporag_ref=None):
        # Init prompt template manager
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = llm_model
        self.hipporag_ref = hipporag_ref
        self.tfidf_vectorizer = None

    def fit_tfidf_vectorizer(self, all_passages: list):
        """
        在 index 阶段批量 fit 全局 TfidfVectorizer。
        """
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            stop_words='english',
            lowercase=True,
            analyzer='word'
        )
        self.tfidf_vectorizer.fit(all_passages)

    def ner(self, chunk_key: str, passage: str) -> NerRawOutput:
        # PREPROCESSING
        ner_input_message = self.prompt_template_manager.render(name='ner', passage=passage)
        raw_response = ""
        metadata = {}
        try:
            # LLM INFERENCE
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=ner_input_message,
            )
            metadata['cache_hit'] = cache_hit
            # 统计 token
            if self.hipporag_ref is not None and 'prompt_tokens' in metadata and 'completion_tokens' in metadata:
                self.hipporag_ref.llm_token_usage['openie_ner']['prompt_tokens'] += metadata['prompt_tokens']
                self.hipporag_ref.llm_token_usage['openie_ner']['completion_tokens'] += metadata['completion_tokens']
                self.hipporag_ref.llm_token_usage['openie_ner']['calls'] += 1
            if metadata['finish_reason'] == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
            extracted_entities = _extract_ner_from_response(real_response)
            unique_entities = list(dict.fromkeys(extracted_entities))

        except Exception as e:
            # For any other unexpected exceptions, log them and return with the error message
            logger.warning(e)
            metadata.update({'error': str(e)})
            return NerRawOutput(
                chunk_id=chunk_key,
                response=raw_response,  # Store the error message in metadata
                unique_entities=[],
                metadata=metadata  # Store the error message in metadata
            )

        return NerRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            unique_entities=unique_entities,
            metadata=metadata
        )

    def tfidf_ner(self, chunk_key: str, passage: str, top_k: int = 10, ngram_lengths: set = None,
                  vectorizer=None, tfidf_threshold: float = None) -> NerRawOutput:
        """
        支持最大3-gram、sklearn英文停用词、全局idf、n-gram选择、tfidf阈值。
        Args:
            chunk_key: str
            passage: str
            top_k: int, 返回top_k个高分词/短语（若未设置tfidf_threshold）
            ngram_lengths: set, 只保留指定长度的 n-gram（如{1,2,3}）
            vectorizer: sklearn TfidfVectorizer，默认用 self.tfidf_vectorizer
            tfidf_threshold: float, 只保留tfidf值大于该阈值的n-gram（优先于top_k）
        Returns:
            NerRawOutput
        """
        if vectorizer is None:
            vectorizer = self.tfidf_vectorizer
        if vectorizer is None:
            raise ValueError("TfidfVectorizer not fitted. Please call fit_tfidf_vectorizer first.")
        tfidf_vec = vectorizer.transform([passage])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_vec.toarray()[0]
        # ngram_lengths: 只保留指定长度的 n-gram
        if ngram_lengths:
            filtered = [(i, feature_names[i]) for i in range(len(feature_names)) if
                        len(feature_names[i].split()) in ngram_lengths]
        else:
            filtered = [(i, feature_names[i]) for i in range(len(feature_names))]
        filtered = [(i, ng) for i, ng in filtered if scores[i] > 0]
        # 支持 tfidf_threshold
        if tfidf_threshold is not None:
            filtered = [(i, ng) for i, ng in filtered if scores[i] > tfidf_threshold]
            filtered.sort(key=lambda x: scores[x[0]], reverse=True)
            top_entities = [ng for i, ng in filtered]
        else:
            filtered.sort(key=lambda x: scores[x[0]], reverse=True)
            top_entities = [ng for i, ng in filtered[:top_k]]
        unique_entities = list(dict.fromkeys(top_entities))
        metadata = {"method": "tfidf_ner", "ngram_lengths": ngram_lengths if ngram_lengths else 'all',
                    "tfidf_threshold": tfidf_threshold}
        return NerRawOutput(
            chunk_id=chunk_key,
            response=";".join(unique_entities),
            unique_entities=unique_entities,
            metadata=metadata
        )

    def triple_extraction(self, chunk_key: str, passage: str, named_entities: List[str],
                          corpus: List[str] = None) -> TripleRawOutput:
        def _extract_triples_from_response(real_response):
            pattern = r'\{[^{}]*"triples"\s*:\s*\[[^\]]*\][^{}]*\}'
            match = re.search(pattern, real_response, re.DOTALL)
            return eval(match.group())["triples"]

        # PREPROCESSING
        messages = self.prompt_template_manager.render(
            name='triple_extraction',
            passage=passage,
            named_entity_json=json.dumps({"named_entities": named_entities})
        )

        raw_response = ""
        metadata = {}
        try:
            # LLM INFERENCE
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=messages,
            )
            metadata['cache_hit'] = cache_hit
            # 统计 token
            if self.hipporag_ref is not None and 'prompt_tokens' in metadata and 'completion_tokens' in metadata:
                self.hipporag_ref.llm_token_usage['openie_triple']['prompt_tokens'] += metadata['prompt_tokens']
                self.hipporag_ref.llm_token_usage['openie_triple']['completion_tokens'] += metadata['completion_tokens']
                self.hipporag_ref.llm_token_usage['openie_triple']['calls'] += 1
            if metadata['finish_reason'] == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
            extracted_triples = _extract_triples_from_response(real_response)
            triplets = filter_invalid_triples(triples=extracted_triples)

        except Exception as e:
            logger.warning(f"Exception for chunk {chunk_key}: {e}")
            metadata.update({'error': str(e)})
            return TripleRawOutput(
                chunk_id=chunk_key,
                response=raw_response,
                metadata=metadata,
                triples=[]
            )

        # Success
        return TripleRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            metadata=metadata,
            triples=triplets
        )

    def openie(self, chunk_key: str, passage: str) -> Dict[str, Any]:
        ner_output = self.ner(chunk_key=chunk_key, passage=passage)
        triple_output = self.triple_extraction(chunk_key=chunk_key, passage=passage,
                                               named_entities=ner_output.unique_entities)
        return {"ner": ner_output, "triplets": triple_output}

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """
        Conduct batch OpenIE synchronously using multi-threading which includes NER and triple extraction.

        Args:
            chunks (Dict[str, ChunkInfo]): chunks to be incorporated into graph. Each key is a hashed chunk 
            and the corresponding value is the chunk info to insert.

        Returns:
            Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
                - A dict with keys as the chunk ids and values as the NER result instances.
                - A dict with keys as the chunk ids and values as the triple extraction result instances.
        """

        # Extract passages from the provided chunks
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}

        ner_results_list = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0

        with ThreadPoolExecutor() as executor:
            # Create NER futures for each chunk
            ner_futures = {
                executor.submit(self.ner, chunk_key, passage): chunk_key
                for chunk_key, passage in chunk_passages.items()
            }

            pbar = tqdm(as_completed(ner_futures), total=len(ner_futures), desc="NER")
            for future in pbar:
                result = future.result()
                ner_results_list.append(result)
                # Update metrics based on the metadata from the result
                metadata = result.metadata
                total_prompt_tokens += metadata.get('prompt_tokens', 0)
                total_completion_tokens += metadata.get('completion_tokens', 0)
                if metadata.get('cache_hit'):
                    num_cache_hit += 1

                pbar.set_postfix({
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'num_cache_hit': num_cache_hit
                })

        triple_results_list = []
        total_prompt_tokens, total_completion_tokens, num_cache_hit = 0, 0, 0
        with ThreadPoolExecutor() as executor:
            # Create triple extraction futures for each chunk
            re_futures = {
                executor.submit(self.triple_extraction, ner_result.chunk_id,
                                chunk_passages[ner_result.chunk_id],
                                ner_result.unique_entities): ner_result.chunk_id
                for ner_result in ner_results_list
            }
            # Collect triple extraction results with progress bar
            pbar = tqdm(as_completed(re_futures), total=len(re_futures), desc="Extracting triples")
            for future in pbar:
                result = future.result()
                triple_results_list.append(result)
                metadata = result.metadata
                total_prompt_tokens += metadata.get('prompt_tokens', 0)
                total_completion_tokens += metadata.get('completion_tokens', 0)
                if metadata.get('cache_hit'):
                    num_cache_hit += 1
                pbar.set_postfix({
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'num_cache_hit': num_cache_hit
                })

        ner_results_dict = {res.chunk_id: res for res in ner_results_list}
        triple_results_dict = {res.chunk_id: res for res in triple_results_list}
        logger.info(f"Total completion tokens: {total_completion_tokens}, "
                    f"Total prompt tokens: {total_prompt_tokens}, "
                    f"Number of cache hits: {num_cache_hit}")
        return ner_results_dict, triple_results_dict

    def batch_openie_tfidf(self, chunks: Dict[str, ChunkInfo],
                           top_k: int = 10,
                           ngram_lengths=None,
                           tfidf_threshold: float = 0.1) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """
        Conduct batch OpenIE synchronously using tfidf_ner for NER and LLM for triple extraction.
        Args:
            chunks (Dict[str, ChunkInfo]): chunks to be incorporated into graph. Each key is a hashed chunk 
            and the corresponding value is the chunk info to insert.
            top_k: int, tfidf_ner返回的实体数（若未设置tfidf_threshold）
            ngram_lengths: set, 只保留指定长度的 n-gram（如{1,2,3}）
            tfidf_threshold: float, 只保留tfidf值大于该阈值的n-gram
        Returns:
            Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
                - A dict with keys as the chunk ids and values as the NER result instances.
                - A dict with keys as the chunk ids and values as the triple extraction result instances.
        """
        if ngram_lengths is None:
            ngram_lengths = {1, 2, 3}
        # 1. Extract all passages for fit
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}
        all_passages = list(chunk_passages.values())
        self.fit_tfidf_vectorizer(all_passages)

        # 并发 NER + 进度条
        ner_results_list = []
        with ThreadPoolExecutor() as executor:
            ner_futures = {
                executor.submit(self.tfidf_ner, chunk_key, passage, top_k, ngram_lengths, None,
                                tfidf_threshold): chunk_key
                for chunk_key, passage in chunk_passages.items()
            }
            pbar = tqdm(as_completed(ner_futures), total=len(ner_futures), desc="TFIDF NER")
            for future in pbar:
                ner_results_list.append(future.result())
                pbar.set_postfix({'completed': len(ner_results_list)})

        # 并发 triple extraction + 进度条
        triple_results_list = []
        with ThreadPoolExecutor() as executor:
            triple_futures = {
                executor.submit(
                    self.triple_extraction,
                    ner_result.chunk_id,
                    chunk_passages[ner_result.chunk_id],
                    ner_result.unique_entities
                ): ner_result.chunk_id
                for ner_result in ner_results_list
            }
            pbar = tqdm(as_completed(triple_futures), total=len(triple_futures), desc="Extracting triples")
            for future in pbar:
                triple_results_list.append(future.result())
                pbar.set_postfix({'completed': len(triple_results_list)})

        ner_results = {r.chunk_id: r for r in ner_results_list}
        triple_results = {r.chunk_id: r for r in triple_results_list}
        return ner_results, triple_results
