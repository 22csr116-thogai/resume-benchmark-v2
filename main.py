"""
main.py
─────────────────────────────────────────────────────────────────────────────
Entry point for the Resume Parser & LLM Benchmarking pipeline.

Usage (local / Google Colab)
─────────────────────────────
  # Full run (all enabled models)
  python main.py

  # Test with a single model (by name, case-insensitive substring)
  python main.py --model "Llama 3 8B"

  # Skip the benchmarking step and re-evaluate previously saved raw results
  python main.py --eval-only

  # Specify custom directories
  python main.py --resume-dir my_resumes/ --ground-truth my_gt.json

Colab quick-start
─────────────────
  !python main.py --model "Llama 3 8B"       # fast smoke-test
  !python main.py                             # full 30-model run
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import logging
import os
import sys

# ── Make project root importable when running from subdirectory ───────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    RESUME_DIR,
    GROUND_TRUTH_FILE,
    OUTPUT_DIR,
    ENABLED_MODELS,
)
from parsers.resume_parser        import load_all_resumes
from benchmarking.benchmark_runner import run_benchmark
from evaluation.evaluator          import load_ground_truth, evaluate_results, aggregate_by_model
from reports.report_generator      import generate_report


# ─────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────

def _setup_logging(level: str = "INFO"):
    fmt = "%(asctime)s [%(levelname)s] %(name)s – %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(OUTPUT_DIR, "run.log"), mode="a"),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Resume Parser LLM Benchmarking Pipeline"
    )
    parser.add_argument(
        "--resume-dir", default=RESUME_DIR,
        help=f"Folder containing resume files (default: {RESUME_DIR})"
    )
    parser.add_argument(
        "--ground-truth", default=GROUND_TRUTH_FILE,
        help=f"Path to ground-truth JSON (default: {GROUND_TRUTH_FILE})"
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help=f"Folder for output files (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--model", default=None,
        help="Filter: run only models whose name contains this substring"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip benchmarking; re-evaluate existing raw_results.json"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    _setup_logging(args.log_level)
    logger = logging.getLogger("main")

    logger.info("=" * 60)
    logger.info("  Resume Parser — LLM Benchmarking Pipeline")
    logger.info("=" * 60)

    raw_results_path = os.path.join(args.output_dir, "raw_results.json")

    # ── Step 1: Parse resumes ─────────────────────────────────────────────
    logger.info(f"\n[Step 1] Loading resumes from: {args.resume_dir}")
    resumes = load_all_resumes(args.resume_dir)

    if not resumes:
        logger.error(
            "No resumes found. Place PDF/DOCX/TXT files in: "
            f"{args.resume_dir}"
        )
        sys.exit(1)

    logger.info(f"  → {len(resumes)} resume(s) loaded.")

    # ── Step 2: Benchmark ─────────────────────────────────────────────────
    if args.eval_only:
        logger.info("\n[Step 2] --eval-only: loading existing raw results …")
        if not os.path.isfile(raw_results_path):
            logger.error(f"No raw_results.json found at {raw_results_path}")
            sys.exit(1)
        with open(raw_results_path, "r", encoding="utf-8") as fh:
            raw_results = json.load(fh)
        logger.info(f"  → {len(raw_results)} result record(s) loaded.")
    else:
        # Filter models if --model flag was given
        models_to_run = ENABLED_MODELS
        if args.model:
            query = args.model.lower()
            models_to_run = [
                m for m in ENABLED_MODELS
                if query in m["name"].lower()
            ]
            if not models_to_run:
                logger.error(
                    f"No models matched '--model {args.model}'. "
                    f"Available: {[m['name'] for m in ENABLED_MODELS]}"
                )
                sys.exit(1)
            logger.info(
                f"\n[Step 2] Running {len(models_to_run)} model(s): "
                f"{[m['name'] for m in models_to_run]}"
            )
        else:
            logger.info(
                f"\n[Step 2] Running all {len(models_to_run)} enabled models …"
            )

        raw_results = run_benchmark(
            resumes=resumes,
            models=models_to_run,
            save_path=raw_results_path,
        )

    # ── Step 3: Evaluate ──────────────────────────────────────────────────
    logger.info(f"\n[Step 3] Evaluating against ground truth: {args.ground_truth}")

    if not os.path.isfile(args.ground_truth):
        logger.warning(
            "Ground-truth file not found. Scores will be zero. "
            "Create: data/ground_truth/ground_truth.json"
        )
        ground_truth = {}
    else:
        ground_truth = load_ground_truth(args.ground_truth)
        logger.info(f"  → Ground truth loaded for {len(ground_truth)} resume(s).")

    per_resume_scores = evaluate_results(raw_results, ground_truth)
    aggregated_scores = aggregate_by_model(per_resume_scores)

    # Save scores JSON
    scores_path = os.path.join(args.output_dir, "scores.json")
    with open(scores_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"per_resume": per_resume_scores, "aggregated": aggregated_scores},
            fh, indent=2
        )
    logger.info(f"  → Scores saved: {scores_path}")

    # ── Step 4: Generate report ───────────────────────────────────────────
    logger.info(f"\n[Step 4] Generating reports …")
    html_path = generate_report(
        per_resume_scores=per_resume_scores,
        aggregated_scores=aggregated_scores,
        output_dir=args.output_dir,
    )

    logger.info(f"\n{'='*60}")
    logger.info("  Pipeline complete. Output files:")
    logger.info(f"    Raw results  : {raw_results_path}")
    logger.info(f"    Scores       : {scores_path}")
    logger.info(f"    CSV report   : {os.path.join(args.output_dir, 'benchmark_report.csv')}")
    logger.info(f"    JSON report  : {os.path.join(args.output_dir, 'benchmark_report.json')}")
    logger.info(f"    HTML report  : {html_path}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
