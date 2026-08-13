"""
Runs the full RAG-Bench experiment sweep.

Usage:
    python -m src.runner                     # full sweep, all configs
    python -m src.runner --sample-size 2      # smoke test: 2 questions per config
    python -m src.runner --dry-run            # print the config matrix, run nothing

Resumability: results are written incrementally to results/<config_name>.json
after every query. If interrupted, re-running skips configs/questions that
already have results, so you don't pay to re-run a sweep from question 1
after a crash on question 40.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.pipeline import RAGPipeline
from src.evaluation import compute_retrieval_metrics, LLMJudge

load_dotenv()


def load_config(path: str = "config/experiments.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_corpus(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_sweep_matrix(cfg: dict) -> list[dict]:
    """Cross chunking strategies x retrieval depths into the full config list."""
    matrix = []
    for strategy in cfg["chunking_strategies"]:
        for top_k in cfg["retrieval_depths"]:
            config_name = f"{strategy['name']}_top{top_k}"
            matrix.append(
                {
                    "config_name": config_name,
                    "chunking_strategy": strategy["name"],
                    "chunking_params": strategy["params"],
                    "top_k": top_k,
                }
            )
    return matrix


def load_existing_results(results_path: Path) -> dict:
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return {"config_name": results_path.stem, "queries": {}}


def run_config(sweep_cfg: dict, cfg: dict, corpus: dict, sample_size: int | None) -> None:
    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / f"{sweep_cfg['config_name']}.json"

    existing = load_existing_results(results_path)
    already_done = set(existing["queries"].keys())

    questions = corpus["questions"]
    if sample_size:
        questions = questions[:sample_size]

    remaining = [q for q in questions if q["question_id"] not in already_done]
    if not remaining:
        print(f"[{sweep_cfg['config_name']}] already complete ({len(already_done)} queries). Skipping.")
        return

    print(f"[{sweep_cfg['config_name']}] running {len(remaining)} queries "
          f"({len(already_done)} already done)...")

    pipeline = RAGPipeline(
        config_name=sweep_cfg["config_name"],
        chunking_strategy=sweep_cfg["chunking_strategy"],
        chunking_params=sweep_cfg["chunking_params"],
        top_k=sweep_cfg["top_k"],
        generation_cfg=cfg["generation"],
        embedding_model=cfg["embedding"]["model"],
        pricing=cfg["pricing"],
    )
    pipeline.index_documents(corpus["documents"])

    judge = LLMJudge(
        model=cfg["judge"]["model"],
        max_tokens=cfg["judge"]["max_tokens"],
        temperature=cfg["judge"]["temperature"],
        input_price_per_million=cfg["pricing"]["input_per_million"],
        output_price_per_million=cfg["pricing"]["output_per_million"],
    )

    for n, q in enumerate(remaining, start=1):
        # per-query progress, printed BEFORE the call - long throttle waits
        # otherwise look like a hang with no output for minutes at a time
        print(f"  [{sweep_cfg['config_name']}] {n}/{len(remaining)} {q['question_id']}...",
              end="", flush=True)

        result = pipeline.answer_question(q["question_id"], q["question"])
        retrieval_metrics = compute_retrieval_metrics(
            result.retrieved_chunks, q["relevant_doc_ids"]
        )
        judge_result = judge.score(q["question"], q["ground_truth_answer"], result.answer)

        existing["queries"][q["question_id"]] = {
            "question": result.question,
            "generated_answer": result.answer,
            "ground_truth_answer": q["ground_truth_answer"],
            "retrieved_chunk_ids": [c["chunk_id"] for c in result.retrieved_chunks],
            "retrieval": {
                "hit_at_k": retrieval_metrics.hit_at_k,
                "precision_at_k": retrieval_metrics.precision_at_k,
                "recall_at_k": retrieval_metrics.recall_at_k,
            },
            "judge_score": judge_result["score"],
            "judge_reasoning": judge_result["reasoning"],
            # judge cost is tracked separately - it's an eval-harness cost,
            # not something a production RAG system would pay
            "judge_cost_usd": judge_result["judge_cost_usd"],
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_seconds": result.latency_seconds,
            "estimated_cost_usd": result.estimated_cost_usd,
        }

        # write after every query - this is what makes the run resumable
        with open(results_path, "w") as f:
            json.dump(existing, f, indent=2)

        score = judge_result["score"]
        print(f" score={score if score is not None else 'UNPARSED'}"
              f" hit={retrieval_metrics.hit_at_k}", flush=True)

    print(f"[{sweep_cfg['config_name']}] done. Results at {results_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=None,
                         help="Only run this many questions per config (smoke test)")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the sweep matrix and exit without calling the API")
    parser.add_argument("--config", type=str, default="config/experiments.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # .env is the single source of truth for which model/provider to use,
    # so switching providers never means editing the experiment config.
    env_model = os.environ.get("LLM_MODEL")
    if env_model:
        cfg["generation"]["model"] = env_model
        cfg["judge"]["model"] = env_model

    corpus = load_corpus(cfg["corpus"]["path"])
    matrix = build_sweep_matrix(cfg)

    print(f"Sweep matrix: {len(matrix)} configurations")
    for m in matrix:
        print(f"  - {m['config_name']}")

    if args.dry_run:
        print("\n--dry-run set, exiting without calling the API.")
        return

    for sweep_cfg in matrix:
        run_config(sweep_cfg, cfg, corpus, args.sample_size)


if __name__ == "__main__":
    main()
