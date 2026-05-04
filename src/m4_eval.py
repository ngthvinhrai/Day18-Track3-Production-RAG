"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from statistics import mean
from dataclasses import dataclass
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:

    """Run RAGAS evaluation."""
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset

    # 1. Build dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # 2. Run RAGAS
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY")) 
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        embeddings=embeddings
    )

    # 3. Convert to DataFrame
    df = result.to_pandas()

    # 4. Per-question results
    per_question = []

    for i, row in df.iterrows():
        per_question.append(
            EvalResult(
                question=questions[i],
                answer=answers[i],
                contexts=contexts[i],
                ground_truth=ground_truths[i],
                faithfulness=float(row["faithfulness"]),
                answer_relevancy=float(row["answer_relevancy"]),
                context_precision=float(row["context_precision"]),
                context_recall=float(row["context_recall"]),
            )
        )


    # 5. Aggregate scores
    return {
        "faithfulness": float(df["faithfulness"].mean()),
        "answer_relevancy": float(df["answer_relevancy"].mean()),
        "context_precision": float(df["context_precision"].mean()),
        "context_recall": float(df["context_recall"].mean()),
        "per_question": per_question,
    }



def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:

    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    analyzed = []

    for r in eval_results:
        avg_score = mean([
            r.faithfulness,
            r.answer_relevancy,
            r.context_precision,
            r.context_recall,
        ])

        metrics = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }

        worst_metric = min(metrics, key=metrics.get)
        worst_score = metrics[worst_metric]

        # Diagnostic rules
        if worst_metric == "faithfulness" and worst_score < 0.85:
            diagnosis = "LLM hallucinating"
            fix = "Tighten prompt, lower temperature, add citation grounding"
        elif worst_metric == "context_recall" and worst_score < 0.75:
            diagnosis = "Missing relevant chunks"
            fix = "Improve chunking, add BM25 or increase retrieval top_k"
        elif worst_metric == "context_precision" and worst_score < 0.75:
            diagnosis = "Too many irrelevant chunks"
            fix = "Add cross-encoder reranking or metadata filtering"
        elif worst_metric == "answer_relevancy" and worst_score < 0.80:
            diagnosis = "Answer does not match the question"
            fix = "Improve prompt template or question conditioning"
        else:
            diagnosis = "General quality issue"
            fix = "Inspect retrieval + prompt jointly"

        analyzed.append({
            "question": r.question,
            "avg_score": avg_score,
            "worst_metric": worst_metric,
            "score": worst_score,
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })

    # 2. Bottom-N worst by avg score
    analyzed.sort(key=lambda x: x["avg_score"])
    return analyzed[:bottom_n]



def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
