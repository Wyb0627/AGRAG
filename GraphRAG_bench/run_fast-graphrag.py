import asyncio
import os
import signal
import atexit
import threading
import re

os.environ['CONCURRENT_TASK_LIMIT'] = '8'
import logging
import argparse
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
from fast_graphrag import GraphRAG, QueryParam
from fast_graphrag._llm import OpenAIEmbeddingService, BaseLLMService, OpenAILLMService
from fast_graphrag._types import BaseModelAlias
from fast_graphrag._utils import logger
from llm_huggingface import HuggingFaceEmbeddingService
from tqdm import tqdm
import time
import openai
from openai import AsyncOpenAI
from pydantic import BaseModel
import tiktoken

# Load environment variables
load_dotenv()

# Global variable to store embedding services for cleanup
# _embedding_services = []
# Configuration constants
DOMAIN = (
    "Analyze this story and identify the characters. Focus on how they interact with each other, the locations they explore, and their relationships. "
    "Respond with no more than 4096 tokens. ")
EXAMPLE_QUERIES = [
    "What is the significance of Christmas Eve in A Christmas Carol?",
    "How does the setting of Victorian London contribute to the story's themes?",
    "Describe the chain of events that leads to Scrooge's transformation.",
    "How does Dickens use the different spirits (Past, Present, and Future) to guide Scrooge?",
    "Why does Dickens choose to divide the story into \"staves\" rather than chapters?"
]
ENTITY_TYPES = ["Character", "Animal", "Place", "Object", "Activity", "Event"]


def group_questions_by_source(question_list: List[dict]) -> Dict[str, List[dict]]:
    """Group questions by their source"""
    grouped_questions = {}
    for question in question_list:
        source = question.get("source")
        if source not in grouped_questions:
            grouped_questions[source] = []
        grouped_questions[source].append(question)
    return grouped_questions


def clean_llm_response(response_text: str) -> str:
    """
    Clean and fix common LLM response format issues
    """
    if not response_text:
        return response_text
    
    # Remove markdown code blocks if present
    response_text = re.sub(r'```json\s*', '', response_text)
    response_text = re.sub(r'```\s*$', '', response_text)
    
    # Fix common JSON formatting issues
    # Replace single quotes with double quotes
    response_text = re.sub(r"'([^']*)'", r'"\1"', response_text)
    
    # Fix trailing commas
    response_text = re.sub(r',(\s*[}\]])', r'\1', response_text)
    
    # Fix missing quotes around property names
    response_text = re.sub(r'(\w+):', r'"\1":', response_text)
    
    return response_text.strip()


def validate_and_fix_json_response(response_text: str) -> str:
    """
    Validate JSON response and attempt to fix common issues
    """
    try:
        # First, try to parse as-is
        json.loads(response_text)
        return response_text
    except json.JSONDecodeError:
        # Try to clean and fix the response
        cleaned_response = clean_llm_response(response_text)
        try:
            json.loads(cleaned_response)
            return cleaned_response
        except json.JSONDecodeError:
            # If still invalid, return original and let the caller handle it
            return response_text


def process_corpus(
        corpus_name: str,
        context: str,
        base_dir: str,
        model_name: str,
        embed_model_path: str,
        llm_base_url: str,
        llm_api_key: str,
        questions: List[dict],
        sample: int
):
    """Process a single corpus: index it and answer its questions"""
    logging.info(f"📚 Processing corpus: {corpus_name}")

    # Prepare output directory
    output_dir = f"./results/{base_dir}/{corpus_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"predictions_{corpus_name}.json")

    # Initialize embedding model
    try:
        logging.info(f"✅ Loaded embedding model: {embed_model_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load embedding model: {e}")
        return

    # Initialize GraphRAG
    grag = GraphRAG(
        working_dir=os.path.join(base_dir, corpus_name),
        domain=DOMAIN,
        example_queries="\n".join(EXAMPLE_QUERIES),
        entity_types=ENTITY_TYPES,
        config=GraphRAG.Config(
            llm_service=TokenTrackingLLMService(
                model=model_name,
                base_url=llm_base_url,
                api_key=llm_api_key,
                token_tracker=token_tracker,  # Pass token tracker
            ),
            embedding_service=HuggingFaceEmbeddingService(
                model='BAAI/bge-large-en-v1.5'
            ),
        ),
    )

    # Register embedding service for global cleanup
    # global _embedding_services
    # if hasattr(grag, 'config') and hasattr(grag.config, 'embedding_service'):
    #     _embedding_services.append(grag.config.embedding_service)

    # Index the corpus content
    grag.insert(context)
    logging.info(f"✅ Indexed corpus: {corpus_name} ({len(context.split())} words)")

    # Get questions for this corpus
    corpus_questions = questions.get(corpus_name, [])
    if not corpus_questions:
        logging.warning(f"⚠️ No questions found for corpus: {corpus_name}")
        return

    # Sample questions if requested
    if sample and sample < len(corpus_questions):
        corpus_questions = corpus_questions[:sample]

    logging.info(f"🔍 Found {len(corpus_questions)} questions for {corpus_name}")

    # Process questions
    results = []
    for q in tqdm(corpus_questions, desc=f"Answering questions for {corpus_name}"):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Execute query
                response = grag.query(q["question"],
                                      params=QueryParam(
                                          entities_max_tokens=1500,
                                          relations_max_tokens=1000,
                                          chunks_max_tokens=4000,
                                      ))
                context_chunks = response.to_dict()['context']['chunks']
                contexts = [item[0]["content"] for item in context_chunks]
                predicted_answer = response.response

                # Collect results
                results.append({
                    "id": q["id"],
                    "question": q["question"],
                    "source": corpus_name,
                    "context": contexts,
                    "evidence": q.get("evidence", ""),
                    "question_type": q.get("question_type", ""),
                    "generated_answer": predicted_answer,
                    "gold_answer": q.get("answer", "")
                })
                break  # Success, exit retry loop
                
            except Exception as e:
                error_msg = str(e)
                logging.warning(f"⚠️ Attempt {attempt + 1}/{max_retries} failed for question {q.get('id')}: {error_msg}")
                
                # Check if it's a validation error that we can handle
                if "validation error" in error_msg.lower() or "pydantic" in error_msg.lower():
                    if attempt < max_retries - 1:
                        logging.info(f"🔄 Retrying due to validation error...")
                        continue
                    else:
                        logging.error(f"❌ All retries failed for question {q.get('id')}: {error_msg}")
                        results.append({
                            "id": q["id"],
                            "error": f"Validation error after {max_retries} attempts: {error_msg}"
                        })
                else:
                    # Non-validation error, don't retry
                    logging.error(f"❌ Error processing question {q.get('id')}: {error_msg}")
                    results.append({
                        "id": q["id"],
                        "error": error_msg
                    })
                    break

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"💾 Saved {len(results)} predictions to: {output_path}")


class TokenTracker:
    """Thread-safe token usage tracker for LLM calls"""

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self._lock = threading.Lock()  # Thread safety lock
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to load tiktoken encoding: {e}. Using fallback tokenization.")
            self.encoding = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)

        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Fallback to word-based counting
            return len(text.split())

    def add_request(self, messages: List[Dict], response_text: str):
        """Thread-safe method to add token counts for a request"""
        # Count input tokens
        input_tokens = sum(self.count_tokens(msg.get("content", "")) for msg in messages)

        # Count output tokens
        output_tokens = self.count_tokens(response_text)

        # Thread-safe update of totals
        with self._lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_requests += 1

    def get_stats(self) -> Dict:
        """Thread-safe method to get token statistics"""
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


class TokenTrackingLLMService(OpenAILLMService):
    """LLM service wrapper that tracks token usage"""

    def __init__(self, *args, **kwargs):
        # Extract token_tracker from kwargs if provided
        self.token_tracker = kwargs.pop('token_tracker', None)
        super().__init__(*args, **kwargs)
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer in TokenTrackingLLMService: {e}. Using fallback.")
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if not isinstance(text, str):
            text = str(text)
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback to word-based counting
            return len(text.split())

    async def complete(self, prompt: str, **kwargs) -> str:
        """Override complete method to track tokens"""
        # Ensure prompt is a string
        if not isinstance(prompt, str):
            prompt = str(prompt)

        # Prepare messages for tracking
        messages = [{"role": "user", "content": prompt}]

        # Call parent method
        response_text = await super().complete(prompt, **kwargs)

        # Track token usage
        if self.token_tracker:
            self.token_tracker.add_request(messages, response_text)
            logger.debug(f"Tracked complete request: {len(messages)} messages, response length: {len(response_text)}")

        return response_text

    async def send_message(
            self,
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, str]] | None = None,
            response_model: Any = None,
            **kwargs: Any,
    ) -> tuple[Any, list[dict[str, str]]]:
        """Override send_message to track tokens"""
        # Ensure all inputs are strings
        if not isinstance(prompt, str):
            prompt = str(prompt)
        
        if system_prompt is not None and not isinstance(system_prompt, str):
            system_prompt = str(system_prompt)
        
        # Prepare messages for tracking
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            # Ensure history messages have string content
            for msg in history_messages:
                if isinstance(msg, dict) and "content" in msg:
                    if not isinstance(msg["content"], str):
                        msg["content"] = str(msg["content"])
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Call parent method
            response_content, response_messages = await super().send_message(
                prompt, system_prompt, history_messages, response_model, **kwargs
            )
            
            # Track token usage
            if self.token_tracker:
                # Convert response_content to string for token counting
                response_text = str(response_content)
                self.token_tracker.add_request(messages, response_text)
                logger.debug(f"Tracked send_message: {len(messages)} messages, response length: {len(response_text)}")
            
            return response_content, response_messages
            
        except Exception as e:
            # If there's a validation error, try to fix the response format
            if "validation error" in str(e).lower() or "pydantic" in str(e).lower():
                logger.warning(f"Validation error detected, attempting to fix response format: {e}")
                try:
                    # Call parent method without response_model to get raw response
                    raw_response_content, raw_response_messages = await super().send_message(
                        prompt, system_prompt, history_messages, None, **kwargs
                    )
                    # Try to fix the response format
                    if isinstance(raw_response_content, str) and response_model is not None:
                        fixed_response = validate_and_fix_json_response(raw_response_content)
                        try:
                            parsed = response_model.parse_raw(fixed_response)
                        except Exception as parse_e:
                            logger.error(f"Failed to parse fixed response: {parse_e}")
                            raise
                        # Track token usage for the attempt
                        if self.token_tracker:
                            response_text = str(raw_response_content)
                            self.token_tracker.add_request(messages, response_text)
                            logger.debug(f"Tracked send_message (fixed): {len(messages)} messages, response length: {len(response_text)}")
                        return parsed, raw_response_messages
                except Exception as fix_error:
                    logger.error(f"Failed to fix response format: {fix_error}")
            # Re-raise the original error if we can't fix it
            raise


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

    parser = argparse.ArgumentParser(description="GraphRAG: Process Corpora and Answer Questions")

    # Core arguments
    parser.add_argument("--subset", required=True, choices=["medical", "novel"],
                        help="Subset to process (medical or novel)")
    parser.add_argument("--base_dir", default="./fast_graphrag_workspace",
                        help="Base working directory for GraphRAG")

    # Model configuration
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-14B-Instruct",
                        help="LLM model identifier")
    parser.add_argument("--embed_model_path", default="BAAI/bge-large-en-v1.5",
                        help="Path to embedding model directory")
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of questions to sample per corpus")

    # API configuration
    parser.add_argument("--llm_base_url", default="http://localhost:8001/v1",
                        help="Base URL for LLM API")
    parser.add_argument("--llm_api_key", default="vllm",
                        help="API key for LLM service (can also use LLM_API_KEY environment variable)")

    args = parser.parse_args()
    args.base_dir = args.base_dir + '_' + args.subset
    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"fastgraphrag_{args.subset}.log")
        ]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)

    logging.info(f"🚀 Starting GraphRAG processing for subset: {args.subset}")

    # Validate subset
    if args.subset not in SUBSET_PATHS:
        logging.error(f"❌ Invalid subset: {args.subset}. Valid options: {list(SUBSET_PATHS.keys())}")
        return

    # Get file paths for this subset
    corpus_path = SUBSET_PATHS[args.subset]["corpus"]
    questions_path = SUBSET_PATHS[args.subset]["questions"]

    # Handle API key security
    api_key = args.llm_api_key or os.getenv("LLM_API_KEY", "")
    if not api_key:
        logging.warning("⚠️ No API key provided! Requests may fail.")

    # Create workspace directory
    os.makedirs(args.base_dir, exist_ok=True)

    # Load corpus data
    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus_data = json.load(f)
        logging.info(f"📖 Loaded corpus with {len(corpus_data)} documents from {corpus_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load corpus: {e}")
        return

    # Sample corpus data if requested
    if args.sample:
        corpus_data = corpus_data[:1]

    # Load question data
    try:
        with open(questions_path, "r", encoding="utf-8") as f:
            question_data = json.load(f)
        grouped_questions = group_questions_by_source(question_data)
        logging.info(f"❓ Loaded questions with {len(question_data)} entries from {questions_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load questions: {e}")
        return

    # Process each corpus in the subset
    error_list = ['58553']
    for item in corpus_data:
        corpus_name = item["corpus_name"]
        context = item["context"]
        '''
        skip = False
        for error_corpus_name in error_list:
            if error_corpus_name not in corpus_name.lower():
                skip = True
        if skip:
            continue
        else:
            os.removedirs(os.path.join(args.base_dir, corpus_name))
        '''
        process_corpus(
            corpus_name=corpus_name,
            context=context,
            base_dir=args.base_dir,
            model_name=args.model_name,
            embed_model_path=args.embed_model_path,
            llm_base_url=args.llm_base_url,
            llm_api_key=api_key,
            questions=grouped_questions,
            sample=args.sample
        )
        token_tracker.print_stats()

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n===== Total running time: {elapsed:.2f} seconds =====")

    # Print final token usage statistics
    token_tracker.print_stats()


if __name__ == "__main__":
    main()
