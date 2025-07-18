# lightrag_example.py
import asyncio
import os
import logging
import nest_asyncio
import argparse
import json
from typing import Dict, List

import openai
import numpy as np
from openai import AsyncOpenAI
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.hf import hf_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import time
import tiktoken

# Apply nest_asyncio for Jupyter environments
nest_asyncio.apply()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"lightrag_processing.log")
    ]
)

logger = logging.getLogger(__name__)

# Suppress verbose logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)


def group_questions_by_source(question_list):
    grouped_questions = {}

    for question in question_list:
        source = question.get("source")

        if source not in grouped_questions:
            grouped_questions[source] = []

        grouped_questions[source].append(question)

    return grouped_questions


import threading

class TokenTracker:
    """Track token usage for LLM calls with thread safety and robust message handling"""

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self._lock = threading.Lock()  # Thread safety for concurrent access

        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logging.warning(f"Failed to initialize tiktoken encoding: {e}")
            self.encoding = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text with error handling"""
        if not text or not isinstance(text, str):
            return 0

        try:
            if self.encoding:
                return len(self.encoding.encode(text))
            else:
                # Fallback to word-based counting
                return len(text.split())
        except Exception as e:
            logging.warning(f"Error counting tokens: {e}")
            # Fallback to character-based counting (rough approximation)
            return len(text) // 4

    def validate_message(self, message: Dict) -> bool:
        """Validate message format"""
        if not isinstance(message, dict):
            return False

        # Check for required fields
        if "role" not in message:
            return False

        # Content can be None or empty string, but if present must be string
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            return False

        return True

    def add_request(self, messages: List[Dict], response_text: str):
        """Add token counts for a request with robust error handling"""
        try:
            # Validate and count input tokens
            input_tokens = 0
            valid_messages = 0

            if not isinstance(messages, list):
                logging.warning("Messages is not a list, skipping token counting")
                return

            for msg in messages:
                if self.validate_message(msg):
                    content = msg.get("content", "")
                    tokens = self.count_tokens(content)
                    input_tokens += tokens
                    valid_messages += 1
                    logging.debug(f"Message '{msg.get('role', 'unknown')}': {tokens} tokens")
                else:
                    logging.warning(f"Invalid message format: {msg}")

            # Count output tokens
            output_tokens = self.count_tokens(response_text)

            # Thread-safe update of totals
            with self._lock:
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                self.total_requests += 1

            logging.info(f"Tracked LightRAG request: {valid_messages} valid messages, "
                        f"{input_tokens} input tokens, {output_tokens} output tokens")

        except Exception as e:
            logging.error(f"Error in add_request: {e}")
            # Still increment request count to maintain consistency
            with self._lock:
                self.total_requests += 1

    def get_stats(self) -> Dict:
        """Get token statistics with thread safety"""
        with self._lock:
            return {
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens,
                "total_requests": self.total_requests,
                "avg_input_tokens_per_request": self.total_input_tokens / max(self.total_requests, 1),
                "avg_output_tokens_per_request": self.total_output_tokens / max(self.total_requests, 1)
            }

    def print_stats(self):
        """Print token statistics"""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("TOKEN USAGE STATISTICS")
        print("="*50)
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Total Input Tokens: {stats['total_input_tokens']:,}")
        print(f"Total Output Tokens: {stats['total_output_tokens']:,}")
        print(f"Total Tokens: {stats['total_tokens']:,}")
        print(f"Average Input Tokens/Request: {stats['avg_input_tokens_per_request']:.1f}")
        print(f"Average Output Tokens/Request: {stats['avg_output_tokens_per_request']:.1f}")
        print("="*50)

    def reset(self):
        """Reset all counters (useful for testing)"""
        with self._lock:
            self.total_input_tokens = 0
            self.total_output_tokens = 0
            self.total_requests = 0


# Global token tracker
token_tracker = TokenTracker()


SYSTEM_PROMPT = """
---Role---
You are a helpful assistant responding to user queries.

---Goal---
Generate direct and concise answers based strictly on the provided Knowledge Base.
Respond in plain text without explanations or formatting.
Maintain conversation continuity and use the same language as the query.
If the answer is unknown, respond with "I don't know". 
Respond with no more than 4096 tokens. 

---Conversation History---
{history}

---Knowledge Base---
{context_data}
"""


async def llm_model_func(
        prompt: str,
        system_prompt: str = None,
        history_messages: list = [],
        keyword_extraction: bool = False,
        **kwargs
) -> str:
    """LLM interface function using OpenAI-compatible API with token tracking"""
    # Get API configuration from kwargs
    model_name = kwargs.get("model_name", "Qwen/Qwen2.5-14B-Instruct")
    base_url = kwargs.get("base_url", "http://localhost:8001/v1")
    api_key = kwargs.get("api_key", "vllm")

    # Validate input parameters
    if not isinstance(prompt, str) or not prompt.strip():
        logging.error("Invalid prompt: prompt must be a non-empty string")
        return "Error: Invalid prompt"

    # 限制prompt token数不超过8192
    maximum_length = 8000
    try:
        tokenizer = kwargs.get("tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) > maximum_length:
            tokens = tokens[:maximum_length]
            prompt = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            logging.info(f"Truncated prompt to {len(tokens)} tokens")
    except Exception as e:
        logging.warning(f"Error in tokenizer truncation: {e}, using word-based fallback")
        # fallback: 用空格分词近似
        words = prompt.split()
        if len(words) > maximum_length:
            prompt = " ".join(words[:maximum_length])
            logging.info(f"Truncated prompt to {len(words)} words")

    # Prepare messages for token tracking
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    try:
        response_text = await openai_complete_if_cache(
            model_name,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=base_url,
            api_key=api_key,
            # **kwargs
        )

        # Track token usage
        token_tracker.add_request(messages, response_text)

        return response_text

    except openai.BadRequestError as e:
        try:
            response_text = await openai_complete_if_cache(
                model_name,
                prompt,
                system_prompt=system_prompt,
                history_messages=[],
                base_url=base_url,
                api_key=api_key,
                # **kwargs
            )
            token_tracker.add_request(messages, response_text)
            return response_text
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            # Track failed request with empty response
            token_tracker.add_request(messages, "")
            return 'Response Failed'

    except Exception as e:
        logging.error(f"Unexpected error in llm_model_func: {e}")
        # Track failed request with empty response
        token_tracker.add_request(messages, "")
        return f"Error: {str(e)}"


async def initialize_rag(
        base_dir: str,
        source: str,
        model_name: str,
        embed_model_name: str,
        llm_base_url: str,
        llm_api_key: str
) -> LightRAG:
    """Initialize LightRAG instance for a specific corpus"""
    working_dir = os.path.join(base_dir, source)

    # Create directory for this corpus
    os.makedirs(working_dir, exist_ok=True)

    # Initialize embedding function
    tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
    embed_model = AutoModel.from_pretrained(embed_model_name)
    embedding_func = EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=lambda texts: hf_embed(texts, tokenizer, embed_model),
    )

    # Create LLM configuration
    llm_kwargs = {
        "model_name": model_name,
        "base_url": llm_base_url,
        "api_key": llm_api_key
    }

    # Create RAG instance
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        llm_model_name=model_name,
        llm_model_max_async=4,
        llm_model_max_token_size=8192,
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        embedding_func=embedding_func,
        llm_model_kwargs=llm_kwargs
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


async def process_corpus(
        corpus_name: str,
        context: str,
        base_dir: str,
        model_name: str,
        embed_model_name: str,
        llm_base_url: str,
        llm_api_key: str,
        questions: List[dict],
        sample: int,
        retrieve_topk: int,
        rag_mode: str
):
    """Process a single corpus: index it and answer its questions"""
    logging.info(f"📚 Processing corpus: {corpus_name}")

    # Initialize RAG for this corpus
    rag = await initialize_rag(
        base_dir=base_dir,
        source=corpus_name,
        model_name=model_name,
        embed_model_name=embed_model_name,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key
    )

    # Index the corpus content
    rag.insert(context)
    logging.info(f"✅ Indexed corpus: {corpus_name} ({len(context.split())} words)")

    corpus_questions = questions.get(corpus_name, [])

    if not corpus_questions:
        logging.warning(f"No questions found for corpus: {corpus_name}")
        return

    # Sample questions if requested
    if sample and sample < len(corpus_questions):
        corpus_questions = corpus_questions[:sample]

    logging.info(f"🔍 Found {len(corpus_questions)} questions for {corpus_name}")

    # Prepare output path
    output_dir = f"./results/{base_dir}/{corpus_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"predictions_{corpus_name}.json")

    # Process questions
    results = []
    query_type = rag_mode

    for q in tqdm(corpus_questions, desc=f"Answering questions for {corpus_name}"):
        # Prepare query parameters
        query_param = QueryParam(
            mode=query_type,
            top_k=retrieve_topk,
            max_token_for_text_unit=4000,
            max_token_for_global_context=2000,
            max_token_for_local_context=2000
        )

        # Execute query
        try:
            response, context = rag.query(
                q["question"],
                param=query_param,
                system_prompt=SYSTEM_PROMPT
            )

            # Handle both async and sync responses
            if asyncio.iscoroutine(response):
                response = await response
        except:
            response = 'Fail to response'
            context = 'No context'
        predicted_answer = str(response)

        # Collect results
        results.append({
            "id": q["id"],
            "question": q["question"],
            "source": corpus_name,
            "context": context,
            "evidence": q["evidence"],
            "question_type": q["question_type"],
            "generated_answer": predicted_answer,
            "gold_answer": q.get("answer"),

        })

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"💾 Saved {len(results)} predictions to: {output_path}")


def main():
    start_time = time.time()
    # Define subset paths
    SUBSET_PATHS = {
        "medical": {
            "corpus": "../dataset/GraphRAG-Benchmark/Datasets/Corpus/medical.json",
            "questions": "../dataset/GraphRAG-Benchmark/Datasets/Questions/medical_questions.json"
        },
        "novel": {
            "corpus": "../dataset/GraphRAG-Benchmark/Datasets/Corpus/novel.json",
            "questions": "../dataset/GraphRAG-Benchmark/Datasets/Questions/novel_questions.json"
        }
    }

    parser = argparse.ArgumentParser(description="LightRAG: Process Corpora and Answer Questions")

    # Core arguments
    parser.add_argument("--subset", required=True, choices=["medical", "novel"],
                        help="Subset to process (medical or novel)")
    parser.add_argument("--base_dir", default="./lightrag_workspace", help="Base working directory")

    # Model configuration
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-14B-Instruct", help="LLM model identifier")
    parser.add_argument("--embed_model", default="BAAI/bge-large-en-v1.5", help="Embedding model name")
    parser.add_argument("--retrieve_topk", type=int, default=5, help="Number of top documents to retrieve")
    parser.add_argument("--sample", type=int, default=None, help="Number of questions to sample per corpus")
    parser.add_argument("--mode", type=str, default='hybrid', help="RAG mode")
    # API configuration
    parser.add_argument("--llm_base_url", default="http://localhost:8001/v1",
                        help="Base URL for LLM API")
    parser.add_argument("--llm_api_key", default="vllm",
                        help="API key for LLM service (can also use LLM_API_KEY environment variable)")

    args = parser.parse_args()
    args.base_dir = args.base_dir + '_' + args.mode + '_' + args.subset
    # Validate subset
    if args.subset not in SUBSET_PATHS:
        logging.error(f"Invalid subset: {args.subset}. Valid options: {list(SUBSET_PATHS.keys())}")
        return

    # Get file paths for this subset
    corpus_path = SUBSET_PATHS[args.subset]["corpus"]
    questions_path = SUBSET_PATHS[args.subset]["questions"]

    # Handle API key security
    api_key = args.llm_api_key or os.getenv("LLM_API_KEY", "")
    if not api_key:
        logging.warning("No API key provided! Requests may fail.")

    # Create workspace directory
    os.makedirs(args.base_dir, exist_ok=True)

    # Load corpus data
    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus_data = json.load(f)
        logging.info(f"Loaded corpus with {len(corpus_data)} documents from {corpus_path}")
    except Exception as e:
        logging.error(f"Failed to load corpus: {e}")
        return

    # Sample corpus data if requested
    if args.sample:
        corpus_data = corpus_data[:1]

    # Load question data
    try:
        with open(questions_path, "r", encoding="utf-8") as f:
            question_data = json.load(f)
            grouped_questions = group_questions_by_source(question_data)
        logging.info(f"Loaded questions with {len(question_data)} entries from {questions_path}")
    except Exception as e:
        logging.error(f"Failed to load questions: {e}")
        return

    # Process each corpus in the subset
    for item in corpus_data:
        corpus_name = item["corpus_name"]
        context = item["context"]
        asyncio.run(
            process_corpus(
                corpus_name=corpus_name,
                context=context,
                base_dir=args.base_dir,
                model_name=args.model_name,
                embed_model_name=args.embed_model,
                llm_base_url=args.llm_base_url,
                llm_api_key=api_key,
                questions=grouped_questions,
                sample=args.sample,
                retrieve_topk=args.retrieve_topk,
                rag_mode=args.mode
            )
        )

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n===== Total running time: {elapsed:.2f} seconds =====")

    # Print token usage statistics
    token_tracker.print_stats()


if __name__ == "__main__":
    main()
