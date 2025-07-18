import asyncio
import argparse
import json
import numpy as np
import os
import tqdm
from typing import Dict, List
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceBgeEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from metrics import compute_answer_correctness, compute_coverage_score, compute_faithfulness_score, compute_rouge_score


async def retry_with_backoff(func, max_retries=3, base_delay=1.0):
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    print(f"Rate limit hit, waiting {delay} seconds before retry {attempt + 1}/{max_retries + 1}")
                    await asyncio.sleep(delay)
                    continue
            raise e


async def evaluate_dataset(
        dataset: Dataset,
        metrics: List[str],
        llm: BaseLanguageModel,
        embeddings: Embeddings,
        batch_size: int = 20,
        delay_between_batches: float = 0
) -> Dict[str, float]:
    results = {metric: [] for metric in metrics}
    questions = dataset["question"]
    answers = dataset["answer"]
    contexts_list = dataset["contexts"]
    ground_truths = dataset["ground_truth"]
    total_samples = len(questions)
    print(f"\nStarting evaluation of {total_samples} samples...")
    sample_results = []
    for i in tqdm.tqdm(range(0, total_samples, batch_size), desc='Evaluating'):
        batch_end = min(i + batch_size, total_samples)
        batch_tasks = []
        for j in range(i, batch_end):
            batch_tasks.append(
                evaluate_sample(
                    question=questions[j],
                    answer=answers[j],
                    contexts=contexts_list[j],
                    ground_truth=ground_truths[j],
                    metrics=metrics,
                    llm=llm,
                    embeddings=embeddings
                )
            )

        async def process_batch():
            return await asyncio.gather(*batch_tasks, return_exceptions=True)

        try:
            batch_results = await retry_with_backoff(process_batch, max_retries=3, base_delay=2.0)
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    print(f"Error processing sample {i + j + 1}: {result}")
                    default_result = {metric: np.nan for metric in metrics}
                    sample_results.append(default_result)
                else:
                    sample_results.append(result)
            if batch_end < total_samples and delay_between_batches > 0:
                await asyncio.sleep(delay_between_batches)
        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {e}")
            for j in range(i, batch_end):
                default_result = {metric: np.nan for metric in metrics}
                sample_results.append(default_result)
    for sample in sample_results:
        for metric, score in sample.items():
            if isinstance(score, (int, float)) and not np.isnan(score):
                results[metric].append(score)
    return {metric: np.nanmean(scores) for metric, scores in results.items()}


async def evaluate_sample(
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
        metrics: List[str],
        llm: BaseLanguageModel,
        embeddings: Embeddings
) -> Dict[str, float]:
    results = {}
    tasks = {}
    if "rouge_score" in metrics:
        tasks["rouge_score"] = compute_rouge_score(answer, ground_truth)
    if "answer_correctness" in metrics:
        tasks["answer_correctness"] = compute_answer_correctness(
            question, answer, ground_truth, llm, embeddings
        )
    if "coverage_score" in metrics:
        tasks["coverage_score"] = compute_coverage_score(
            question, ground_truth, answer, llm
        )
    if "faithfulness" in metrics:
        tasks["faithfulness"] = compute_faithfulness_score(
            question, answer, contexts, llm
        )
    try:
        task_results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for i, (metric, result) in enumerate(zip(tasks.keys(), task_results)):
            if isinstance(result, Exception):
                print(f"Error computing {metric}: {result}")
                results[metric] = np.nan
            else:
                results[metric] = result
    except Exception as e:
        print(f"Error in evaluate_sample: {e}")
        for metric in metrics:
            results[metric] = np.nan
    return results


async def main(args: argparse.Namespace):
    llm = ChatOpenAI(
        model=args.model,
        base_url=args.base_url,
        api_key="sk-local-test",  # 这里不要用空字符串
        temperature=0.0,
        max_retries=5,
        timeout=60
    )
    bge_embeddings = HuggingFaceBgeEmbeddings(model_name=args.bge_model)
    embedding = LangchainEmbeddingsWrapper(embeddings=bge_embeddings)
    print(f"Loading evaluation data from {args.data_file}...")
    with open(args.data_file, 'r') as f:
        file_data = json.load(f)
    metric_config = {
        'Fact Retrieval': ["rouge_score",
                           "answer_correctness"
                           ],
        'Complex Reasoning': ["rouge_score",
                              "answer_correctness"
                              ],
        'Contextual Summarize': ["answer_correctness",
                                 "coverage_score"
                                 ],
        'Creative Generation': ["answer_correctness",
                                "coverage_score",
                                "faithfulness"]
    }
    grouped_data = {}
    for item in file_data:
        q_type = item.get("question_type", "Uncategorized")
        if q_type not in grouped_data:
            grouped_data[q_type] = []
        grouped_data[q_type].append(item)
    all_results = {}
    for question_type in list(grouped_data.keys()):
        if question_type not in metric_config:
            print(f"Skipping undefined question type: {question_type}")
            continue
        print(f"\n{'=' * 50}")
        print(f"Evaluating question type: {question_type}")
        print(f"{'=' * 50}")
        group_items = grouped_data[question_type]
        questions = [item['question'] for item in group_items]
        ground_truths = [item['gold_answer'] for item in group_items]
        answers = [item['generated_answer'] for item in group_items]
        contexts = [item['context']['compressed_prompt'] if isinstance(item['context'], dict) else item['context'] for
                    item in group_items]
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        dataset = Dataset.from_dict(data)
        results = await evaluate_dataset(
            dataset=dataset,
            metrics=metric_config[question_type],
            llm=llm,
            embeddings=embedding,
            batch_size=args.batch_size,
            delay_between_batches=args.delay_between_batches
        )
        all_results[question_type] = results
        print(f"\nResults for {question_type}:")
        for metric, score in results.items():
            print(f"  {metric}: {score:.4f}")
    if args.output_file:
        print(f"\nSaving results to {args.output_file}...")
        with open(args.output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
    print("\nFinal Evaluation Summary:")
    print("=" * 50)
    for q_type, metrics in all_results.items():
        print(f"\nQuestion Type: {q_type}")
        for metric, score in metrics.items():
            print(f"  {metric}: {score:.4f}")
    print('\nEvaluation complete.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate RAG performance using various metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-14B-Instruct",
        help="vLLM model name (must match your vllm deployment)"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8001/v1",
        help="Base URL for the vLLM OpenAI-compatible API"
    )
    parser.add_argument(
        "--bge_model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="HuggingFace model for BGE embeddings"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to JSON file containing evaluation data"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_results_vllm.json",
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of samples to process simultaneously (increase for local vllm)"
    )
    parser.add_argument(
        "--delay_between_batches",
        type=float,
        default=0,
        help="Delay in seconds between batches (set 0 for local vllm)"
    )
    args = parser.parse_args()
    asyncio.run(main(args))
