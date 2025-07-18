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
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time
import tiktoken

# Apply nest_asyncio for Jupyter environments
nest_asyncio.apply()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"graphrag_processing.log")
    ]
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.WARNING)


class TokenTracker:
    """Track token usage for LLM calls"""

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))

    def add_request(self, messages: List[Dict], response_text: str):
        """Add token counts for a request"""
        # Count input tokens
        input_tokens = sum(self.count_tokens(msg.get("content", "")) for msg in messages)

        # Count output tokens
        output_tokens = self.count_tokens(response_text)

        # Update totals
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_requests += 1

    def get_stats(self) -> Dict:
        """Get token statistics"""
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
        print("\n" + "=" * 50)
        print("TOKEN USAGE STATISTICS")
        print("=" * 50)
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Total Input Tokens: {stats['total_input_tokens']:,}")
        print(f"Total Output Tokens: {stats['total_output_tokens']:,}")
        print(f"Total Tokens: {stats['total_tokens']:,}")
        print(f"Average Input Tokens/Request: {stats['avg_input_tokens_per_request']:.1f}")
        print(f"Average Output Tokens/Request: {stats['avg_output_tokens_per_request']:.1f}")
        print("=" * 50)


# Global token tracker
token_tracker = TokenTracker()


def group_questions_by_source(question_list):
    grouped_questions = {}

    for question in question_list:
        source = question.get("source")

        if source not in grouped_questions:
            grouped_questions[source] = []

        grouped_questions[source].append(question)

    return grouped_questions


def limit_messages_to_tokens(messages: List[Dict], max_tokens: int = 4096, model: str = "gpt-3.5-turbo") -> List[Dict]:
    """
    Limit messages to a maximum number of tokens by truncating from the beginning if necessary.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        max_tokens: Maximum number of tokens allowed (default: 4096)
        model: Model name for tokenizer (default: "gpt-3.5-turbo")

    Returns:
        List of messages truncated to fit within max_tokens
    """
    try:
        # Initialize tokenizer
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base encoding if model not found
        encoding = tiktoken.get_encoding("cl100k_base")

    # Calculate tokens for each message
    total_tokens = 0
    message_tokens = []

    for message in messages:
        # Count tokens in the message content
        content = message.get("content", "")
        tokens = len(encoding.encode(content))
        message_tokens.append(tokens)
        total_tokens += tokens

    # If total tokens exceed max_tokens, truncate from the beginning
    if total_tokens > max_tokens:
        logging.warning(f"Messages exceed {max_tokens} tokens ({total_tokens}), truncating...")

        # Always keep system messages at the beginning
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        non_system_messages = [msg for msg in messages if msg.get("role") != "system"]

        # Calculate tokens for system messages
        system_tokens = sum(len(encoding.encode(msg.get("content", ""))) for msg in system_messages)
        remaining_tokens = max_tokens - system_tokens

        if remaining_tokens <= 0:
            # If system messages alone exceed the limit, keep only the first system message
            logging.warning("System messages exceed token limit, keeping only the first one")
            return system_messages[:1]

        truncated_messages = system_messages.copy()

        # Process non-system messages from the end (most recent first)
        for msg in reversed(non_system_messages):
            content = msg.get("content", "")
            tokens = len(encoding.encode(content))

            if tokens <= remaining_tokens:
                truncated_messages.insert(len(system_messages), msg)
                remaining_tokens -= tokens
            else:
                # If a single message is too long, truncate its content
                if msg.get("role") == "user" and remaining_tokens > 10:  # Ensure minimum tokens for truncation
                    # Truncate content to fit remaining tokens
                    encoded_content = encoding.encode(content)
                    if len(encoded_content) > remaining_tokens:
                        truncated_encoded = encoded_content[-remaining_tokens:]
                        truncated_content = encoding.decode(truncated_encoded)

                        # Add ellipsis to indicate truncation
                        if truncated_content != content:
                            truncated_content = "..." + truncated_content

                        truncated_messages.insert(len(system_messages), {
                            "role": msg["role"],
                            "content": truncated_content
                        })
                break

        final_tokens = sum(len(encoding.encode(msg.get("content", ""))) for msg in truncated_messages)
        logging.info(f"Truncated messages from {total_tokens} to {final_tokens} tokens")
        return truncated_messages

    return messages


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

MODEL = 'Qwen/Qwen2.5-14B-Instruct'


async def llm_model_if_cache(
        prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key="vllm", base_url='http://localhost:8001/v1'
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt + '\nRespond with no more than 4096 tokens.'})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # Limit messages to 4096 tokens
    messages = limit_messages_to_tokens(messages, max_tokens=4096, model="gpt-3.5-turbo")

    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            # Track cached response tokens
            token_tracker.add_request(messages, if_cache_return["return"])
            logging.debug(
                f"Tracked cached request: {len(messages)} messages, response length: {len(if_cache_return['return'])}")
            return if_cache_return["return"]
    # -----------------------------------------------------

    try:
        response = await openai_async_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            **kwargs
        )

        response_text = response.choices[0].message.content

        # Track token usage
        token_tracker.add_request(messages, response_text)
        logging.debug(f"Tracked API request: {len(messages)} messages, response length: {len(response_text)}")

        # Cache the response if having-------------------
        if hashing_kv is not None:
            await hashing_kv.upsert(
                {args_hash: {"return": response_text, "model": MODEL}}
            )
        # -----------------------------------------------------
        return response_text

    except Exception as e:
        logging.error(f"Error in llm_model_if_cache: {e}")
        # Track failed request with empty response
        token_tracker.add_request(messages, "")
        return f"Error: {str(e)}"


EMBED_MODEL = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    # cache_folder=os.path.join(base_dir, source),
    device="cuda"
)


@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)


def initialize_rag(
        base_dir: str,
        source: str,
        model_name: str,
        embed_model_name: str,
        llm_base_url: str,
        llm_api_key: str,
        rag_mode: str = 'local',
) -> GraphRAG:
    """Initialize LightRAG instance for a specific corpus"""
    working_dir = os.path.join(base_dir, source)
    # Create directory for this corpus
    os.makedirs(working_dir, exist_ok=True)
    # Initialize embedding function
    # We're using Sentence Transformers to generate embeddings for the BGE model
    rag = GraphRAG(
        working_dir=working_dir,
        embedding_func=local_embedding,
        best_model_func=llm_model_if_cache,
        cheap_model_func=llm_model_if_cache,
        enable_naive_rag=True if rag_mode == 'naive' else False,
    )
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
        rag_mode: str,
        compress: bool
):
    """Process a single corpus: index it and answer its questions"""
    logging.info(f"📚 Processing corpus: {corpus_name}")

    # Initialize RAG for this corpus
    rag = initialize_rag(
        base_dir=base_dir,
        source=corpus_name,
        model_name=model_name,
        embed_model_name=embed_model_name,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        rag_mode=rag_mode
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
            # naive_max_token_for_text_unit=8100,
            local_max_token_for_text_unit=3000,
            local_max_token_for_local_context=3500,
            local_max_token_for_community_report=3200,
            global_max_token_for_community_report=7000,
            compress=compress
        )

        # Execute query
        # try:
        response, context = rag.query(
            q["question"],
            param=query_param,
            # system_prompt=SYSTEM_PROMPT
        )
        # Handle both async and sync responses
        if asyncio.iscoroutine(response):
            response = await response
        # except:
        #     response = 'Fail to response'
        #    context = 'No context'
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
    parser.add_argument("--base_dir", default="./graphrag_workspace", help="Base working directory")

    # Model configuration
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-14B-Instruct", help="LLM model identifier")
    parser.add_argument("--embed_model", default="BAAI/bge-large-en-v1.5", help="Embedding model name")
    parser.add_argument("--retrieve_topk", type=int, default=5, help="Number of top documents to retrieve")
    parser.add_argument("--sample", type=int, default=None, help="Number of questions to sample per corpus")
    parser.add_argument("--mode", type=str, default='local', help="RAG mode")
    parser.add_argument("--compress", action='store_true', help="LongLLMLingua")
    # API configuration
    parser.add_argument("--llm_base_url", default="http://localhost:8001/v1",
                        help="Base URL for LLM API")
    parser.add_argument("--llm_api_key", default="vllm",
                        help="API key for LLM service (can also use LLM_API_KEY environment variable)")

    args = parser.parse_args()
    args.base_dir = args.base_dir + '_' + args.mode + '_' + args.subset
    if args.compress:
        args.base_dir = args.base_dir + '_' + 'LongLLMLingua'
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
                rag_mode=args.mode,
                compress=args.compress
            )
        )

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n===== Total running time: {elapsed:.2f} seconds =====")

    # Print token usage statistics
    token_tracker.print_stats()


if __name__ == "__main__":
    main()
