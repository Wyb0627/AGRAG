import os
import argparse

#  hipporag2 openai==1.58.1
parser = argparse.ArgumentParser(description="HippoRAG: Process Corpora and Answer Questions")

# Core arguments
parser.add_argument("--subset", required=True, choices=["medical", "novel"],
                    help="Subset to process (medical or novel)")
# parser.add_argument("--base_dir", default="./hipporag2_workspace",
#                    help="Base working directory for HippoRAG")
parser.add_argument("--gpus", type=str, default='0')
# Model configuration
parser.add_argument("--model_name", default="Qwen/Qwen2.5-14B-Instruct",
                    help="LLM model identifier")
parser.add_argument("--embed_model_name", default="facebook/contriever", help="Path to embedding model directory")
# parser.add_argument("--embed_model_name", default="GritLM/GritLM-7B",help="Path to embedding model directory")

parser.add_argument("--sample", type=int, default=None,
                    help="Number of questions to sample per corpus")
parser.add_argument("--rag_mode", type=str, default='gorag')
parser.add_argument("--graph_summary", action='store_true')
parser.add_argument("--linking_top_k", type=int, default=5,
                    help="The top k fact selected for linking")
parser.add_argument("--include_passage_nodes", action='store_true')
# API configuration
parser.add_argument("--llm_base_url", default="http://localhost:8001/v1",
                    help="Base URL for LLM API")
parser.add_argument("--llm_api_key", default="",
                    help="API key for LLM service (can also use OPENAI_API_KEY environment variable)")

args = parser.parse_args()
args.base_dir = f'{args.subset}_{args.rag_mode}_workspace_LLMNER'
# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

import json
import logging
from typing import Dict, List
from dotenv import load_dotenv
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

# Import HippoRAG components after setting environment
from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.misc_utils import string_to_bool
from src.hipporag.utils.config_utils import BaseConfig
from time import time

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{args.rag_mode}_processing.log")
    ]
)

global_token_usage = {}
global_index_time = 0.0
global_qa_time = 0.0


def merge_token_usage(global_usage, summary_str):
    for line in summary_str.splitlines():
        if ':' in line and 'prompt=' in line:
            step, rest = line.split(':', 1)
            parts = [p.strip() for p in rest.split(',')]
            prompt = int(parts[0].split('=')[1].strip())
            completion = int(parts[1].split('=')[1].strip())
            calls = int(parts[2].split('=')[1].strip())
            if step not in global_usage:
                global_usage[step] = {'prompt': 0, 'completion': 0, 'calls': 0}
            global_usage[step]['prompt'] += prompt
            global_usage[step]['completion'] += completion
            global_usage[step]['calls'] += calls


def group_questions_by_source(question_list: List[dict]) -> Dict[str, List[dict]]:
    """Group questions by their source"""
    grouped_questions = {}
    for question in question_list:
        source = question.get("source")
        if source not in grouped_questions:
            grouped_questions[source] = []
        grouped_questions[source].append(question)
    return grouped_questions


def split_text(
        text: str,
        tokenizer: AutoTokenizer,
        chunk_token_size: int = 256,
        chunk_overlap_token_size: int = 32
) -> List[str]:
    """Split text into chunks based on token length with overlap"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    start = 0
    while start < len(tokens):
        end = min(start + chunk_token_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(
            chunk_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        chunks.append(chunk_text)
        if end == len(tokens):
            break
        start += chunk_token_size - chunk_overlap_token_size
    return chunks


def process_corpus(
        corpus_name: str,
        context: str,
        base_dir: str,
        model_name: str,
        embed_model_name: str,
        llm_base_url: str,
        llm_api_key: str,
        questions: List[dict],
        sample: int
):
    """Process a single corpus: index it and answer its questions"""
    logging.info(f"📚 Processing corpus: {corpus_name}")

    # Prepare output directory
    output_dir = f"./results/{base_dir}{'_passage_node' if args.include_passage_nodes else ''}/{corpus_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"predictions_{corpus_name}.json")
    # if not os.path.exists(base_dir):
    #     os.makedirs(base_dir)
    # Initialize tokenizer for text splitting
    try:
        tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
        logging.info(f"✅ Loaded tokenizer: {embed_model_name}")
    except Exception as e:
        logging.error(f"❌ Failed to load tokenizer: {e}")
        return

    # Split text into chunks
    chunks = split_text(context, tokenizer)
    logging.info(f"✂️ Split corpus into {len(chunks)} chunks")

    # Format chunks as documents
    docs = [f'{idx}:{chunk}' for idx, chunk in enumerate(chunks)]

    # Get questions for this corpus
    corpus_questions = questions.get(corpus_name, [])
    if not corpus_questions:
        logging.warning(f"⚠️ No questions found for corpus: {corpus_name}")
        return

    # Sample questions if requested
    if sample and sample < len(corpus_questions):
        corpus_questions = corpus_questions[:sample]

    logging.info(f"🔍 Found {len(corpus_questions)} questions for {corpus_name}")

    # Prepare queries and gold answers
    all_queries = [q["question"] for q in corpus_questions]
    gold_answers = [[q['answer']] for q in corpus_questions]

    # Configure HippoRAG
    config = BaseConfig(
        save_dir=os.path.join(base_dir, corpus_name),
        llm_base_url=llm_base_url,
        llm_name=model_name,
        embedding_model_name=embed_model_name,
        force_index_from_scratch=True,
        force_openie_from_scratch=True,
        graph_summary=args.graph_summary,
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=args.linking_top_k,
        max_qa_steps=3,
        qa_top_k=5,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=8,
        max_new_tokens=None,
        corpus_len=len(docs),
        openie_mode="online",
        rag_mode=args.rag_mode,
        include_passage_nodes_in_qa_input=args.include_passage_nodes
    )

    # Initialize HippoRAG
    hipporag = HippoRAG(global_config=config)

    # Index the corpus content
    index_start_time = time()
    hipporag.index(docs)
    index_time = time() - index_start_time
    logging.info(f"✅ Indexed corpus: {corpus_name}")

    # Process questions
    results = []
    qa_start_time = time()
    queries_solutions, _, _, _, _ = hipporag.rag_qa(queries=all_queries, gold_docs=None, gold_answers=gold_answers)
    qa_time = time() - qa_start_time
    solutions = [query.to_dict() for query in queries_solutions]

    for question in corpus_questions:
        solution = next((sol for sol in solutions if sol['question'] == question['question']), None)
        if solution:
            results.append({
                "id": question["id"],
                "question": question["question"],
                "source": corpus_name,
                "context": solution.get("docs", ""),
                "evidence": question.get("evidence", ""),
                "question_type": question.get("question_type", ""),
                "generated_answer": solution.get("answer", ""),
                "gold_answer": question.get("answer", ""),
                "generated_graph": solution.get("generated_graph", "False")
            })
    # 打印 LLM token 消耗统计
    print("\n===== LLM Token Usage Summary =====")
    summary = hipporag.get_llm_token_usage_summary()
    print(summary)
    print(f"Indexing time: {index_time:.2f}s, QA time: {qa_time:.2f}s for corpus {corpus_name}")
    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"💾 Saved {len(results)} predictions to: {output_path}")
    return summary, index_time, qa_time


def main():
    global global_index_time, global_qa_time
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

    logging.info(f"🚀 Starting HippoRAG processing for subset: {args.subset}")

    # Validate subset
    if args.subset not in SUBSET_PATHS:
        logging.error(f"❌ Invalid subset: {args.subset}. Valid options: {list(SUBSET_PATHS.keys())}")
        return

    # Get file paths for this subset
    corpus_path = SUBSET_PATHS[args.subset]["corpus"]
    questions_path = SUBSET_PATHS[args.subset]["questions"]

    # Handle API key security
    api_key = args.llm_api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key and 'gpt' in args.model_name.lower():
        logging.warning("⚠️ No API key provided! Requests may fail.")

    # Create workspace directory
    # os.makedirs(args.base_dir, exist_ok=True)

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
        logging.info(f"Sample corpus data, only read 1 sample")

    # Load question data
    try:
        with open(questions_path, "r", encoding="utf-8") as f:
            question_data = json.load(f)
        grouped_questions = group_questions_by_source(question_data)
        logging.info(f"❓ Loaded questions with {len(question_data)} entries from {questions_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load questions: {e}")
        return
    embeder_name = args.embed_model_name.split("/")[-1]
    # Process each corpus in the subset
    for item in corpus_data:
        corpus_name = item["corpus_name"]
        context = item["context"]
        summary, index_time, qa_time = process_corpus(
            corpus_name=corpus_name,
            context=context,
            base_dir=f'{args.base_dir}_{embeder_name}',
            model_name=args.model_name,
            embed_model_name=args.embed_model_name,
            llm_base_url=args.llm_base_url,
            llm_api_key=api_key,
            questions=grouped_questions,
            sample=args.sample
        )
        merge_token_usage(global_token_usage, summary)
        global_index_time += index_time
        global_qa_time += qa_time

    # 输出全局统计
    print('\n===== GLOBAL LLM Token Usage Summary =====')
    total_prompt = total_completion = total_calls = 0
    for step, usage in global_token_usage.items():
        print(f'{step}: prompt={usage["prompt"]}, completion={usage["completion"]}, calls={usage["calls"]}')
        total_prompt += usage["prompt"]
        total_completion += usage["completion"]
        total_calls += usage["calls"]
    print(f'Total prompt tokens: {total_prompt}')
    print(f'Total completion tokens: {total_completion}')
    print(f'Total LLM calls: {total_calls}')
    print(f'Total indexing time: {global_index_time:.2f}s, total QA time: {global_qa_time:.2f}s')


if __name__ == "__main__":
    main()
