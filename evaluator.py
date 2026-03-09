"""
evaluation/evaluator.py
─────────────────────────────────────────────────────────────────────────────
Scores each model's output against ground-truth data on four metrics:

  1. extraction_accuracy  – precision: fraction of extracted skills that are
                            actually in the ground truth.
  2. skill_coverage       – recall: fraction of ground-truth skills that the
                            model found.
  3. years_correctness    – how close the model's years estimates are to the
                            ground-truth values (mean relative accuracy).
  4. response_consistency – how stable the model's skill set is across the
                            CONSISTENCY_RUNS repeated calls.

A weighted composite score is calculated from these four metrics using the
weights defined in config.EVAL_WEIGHTS.

Ground-truth format  (data/ground_truth/ground_truth.json)
──────────────────────────────────────────────────────────
{
  "resume_filename.pdf": {
    "skills": [
      {"skill": "Python",  "years": 4.0},
      {"skill": "Docker",  "years": 2.0},
      ...
    ]
  },
  ...
}

Public API
──────────
  evaluate_results(raw_results, ground_truth) -> list[dict]
      Returns per-model-per-resume score dicts.

  aggregate_by_model(scores) -> list[dict]
      Returns one aggregated score dict per model (mean across resumes).
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import math
from collections import defaultdict

from config import EVAL_WEIGHTS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Text normalisation
# ─────────────────────────────────────────────────────────────────────────

def _normalise(skill_name: str) -> str:
    """Lower-case and remove common punctuation for fuzzy matching."""
    return skill_name.lower().strip().replace("-", " ").replace("_", " ")


def _skill_set(skills: list) -> set:
    """Return a set of normalised skill names from a skills list."""
    return {_normalise(s["skill"]) for s in skills if s.get("skill")}


# ─────────────────────────────────────────────────────────────────────────
# Individual metrics
# ─────────────────────────────────────────────────────────────────────────

def _extraction_accuracy(extracted: list, ground_truth: list) -> float:
    """
    Precision: of the skills the model extracted, what fraction are correct?
    Returns 0.0 if nothing was extracted.
    """
    if not extracted:
        return 0.0
    gt_set  = _skill_set(ground_truth)
    ex_set  = _skill_set(extracted)
    correct = len(ex_set & gt_set)
    return round(correct / len(ex_set), 4) if ex_set else 0.0


def _skill_coverage(extracted: list, ground_truth: list) -> float:
    """
    Recall: of the ground-truth skills, what fraction did the model find?
    Returns 0.0 if ground truth is empty.
    """
    if not ground_truth:
        return 1.0   # vacuously perfect if no ground truth
    gt_set  = _skill_set(ground_truth)
    ex_set  = _skill_set(extracted)
    found   = len(ex_set & gt_set)
    return round(found / len(gt_set), 4)


def _years_correctness(extracted: list, ground_truth: list) -> float:
    """
    Mean accuracy of years-of-experience estimates.

    For each ground-truth skill that has a non-null years value:
      • If the model found the skill AND provided a years estimate,
        accuracy = 1 - min(|predicted - actual| / actual, 1)
      • If the model found the skill but provided null years → 0.5 (partial)
      • If the model missed the skill entirely → 0.0

    Returns the mean across all such skills, or 1.0 if no GT years exist.
    """
    gt_skills = {
        _normalise(s["skill"]): s.get("years")
        for s in ground_truth
        if s.get("skill") and s.get("years") is not None
    }

    if not gt_skills:
        return 1.0   # no years data to evaluate

    ex_lookup = {}
    for s in extracted:
        name = _normalise(s.get("skill", ""))
        if name:
            ex_lookup[name] = s.get("years")

    scores = []
    for gt_skill, gt_years in gt_skills.items():
        if gt_skill not in ex_lookup:
            scores.append(0.0)                         # missed entirely
        elif ex_lookup[gt_skill] is None:
            scores.append(0.5)                         # found but no years
        else:
            diff = abs(ex_lookup[gt_skill] - gt_years)
            scores.append(1.0 - min(diff / gt_years, 1.0))

    return round(sum(scores) / len(scores), 4)


def _response_consistency(consistency_runs: list) -> float:
    """
    Measure how stable the model's skill set is across repeated calls.

    Strategy: pairwise Jaccard similarity between each run's skill set
    (including the primary run as run 0).
    Returns the mean Jaccard similarity; 1.0 = perfectly consistent.
    """
    skill_sets = [_skill_set(run.get("parsed_skills", [])) for run in consistency_runs]
    if not skill_sets:
        return 1.0

    # Remove empty sets before comparing (a failed call shouldn't tank consistency)
    non_empty = [s for s in skill_sets if s]
    if len(non_empty) < 2:
        return 1.0

    similarities = []
    for i in range(len(non_empty)):
        for j in range(i + 1, len(non_empty)):
            intersection = len(non_empty[i] & non_empty[j])
            union        = len(non_empty[i] | non_empty[j])
            similarities.append(intersection / union if union else 1.0)

    return round(sum(similarities) / len(similarities), 4)


# ─────────────────────────────────────────────────────────────────────────
# Composite score
# ─────────────────────────────────────────────────────────────────────────

def _composite(scores: dict) -> float:
    """Weighted average of the four metric scores."""
    total = sum(
        EVAL_WEIGHTS.get(metric, 0) * value
        for metric, value in scores.items()
    )
    return round(total, 4)


# ─────────────────────────────────────────────────────────────────────────
# Ground truth loader
# ─────────────────────────────────────────────────────────────────────────

def load_ground_truth(path: str) -> dict:
    """
    Load ground-truth JSON file.

    Returns a dict keyed by resume filename, each value being a list of
    skill dicts: [{"skill": ..., "years": ...}, ...]
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    # Accept both {"filename": {"skills": [...]}} and {"filename": [...]}
    normalised = {}
    for fname, content in raw.items():
        if isinstance(content, list):
            normalised[fname] = content
        elif isinstance(content, dict):
            normalised[fname] = content.get("skills", [])
        else:
            normalised[fname] = []

    return normalised


# ─────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────

def evaluate_results(raw_results: list, ground_truth: dict) -> list:
    """
    Score every (model, resume) result against ground truth.

    Parameters
    ----------
    raw_results  : list – output of benchmark_runner.run_benchmark()
    ground_truth : dict – output of load_ground_truth()

    Returns
    -------
    list of score dicts, one per (model, resume) pair:
      {
        model_name, model_id, resume_filename,
        extraction_accuracy, skill_coverage,
        years_correctness, response_consistency,
        composite_score,
        extracted_count, gt_count,
        latency_s, had_error
      }
    """
    scores = []

    for result in raw_results:
        fname    = result["resume_filename"]
        gt       = ground_truth.get(fname, [])
        primary  = result.get("primary", {})
        cons     = result.get("consistency", [])

        extracted = primary.get("parsed_skills", [])
        latency   = primary.get("latency_s", 0.0)
        had_error = primary.get("error") is not None

        # Include primary run in consistency list for calculating consistency
        all_runs = [{"parsed_skills": extracted}] + cons

        ea  = _extraction_accuracy(extracted, gt)
        sc  = _skill_coverage(extracted, gt)
        yc  = _years_correctness(extracted, gt)
        rc  = _response_consistency(all_runs)

        metric_scores = {
            "extraction_accuracy":  ea,
            "skill_coverage":       sc,
            "years_correctness":    yc,
            "response_consistency": rc,
        }
        comp = _composite(metric_scores)

        scores.append({
            "model_name":           result["model_name"],
            "model_id":             result["model_id"],
            "provider":             result.get("provider", "ollama"),
            "resume_filename":      fname,
            "extraction_accuracy":  ea,
            "skill_coverage":       sc,
            "years_correctness":    yc,
            "response_consistency": rc,
            "composite_score":      comp,
            "extracted_count":      len(extracted),
            "gt_count":             len(gt),
            "latency_s":            latency,
            "had_error":            had_error,
        })

        status = "✗ error" if had_error else "✓"
        logger.debug(
            f"  {status} [{result['model_name']}] {fname} → "
            f"EA={ea:.2f} SC={sc:.2f} YC={yc:.2f} RC={rc:.2f} → {comp:.2f}"
        )

    return scores


def aggregate_by_model(scores: list) -> list:
    """
    Average per-resume scores into one row per model.

    Returns list sorted by composite_score descending (best model first).
    """
    grouped: dict = defaultdict(list)
    for s in scores:
        grouped[s["model_name"]].append(s)

    aggregated = []
    for model_name, rows in grouped.items():
        n = len(rows)

        def mean(key):
            vals = [r[key] for r in rows if not r["had_error"]]
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        aggregated.append({
            "model_name":           model_name,
            "model_id":             rows[0]["model_id"],
            "provider":             rows[0]["provider"],
            "resumes_tested":       n,
            "error_count":          sum(1 for r in rows if r["had_error"]),
            "extraction_accuracy":  mean("extraction_accuracy"),
            "skill_coverage":       mean("skill_coverage"),
            "years_correctness":    mean("years_correctness"),
            "response_consistency": mean("response_consistency"),
            "composite_score":      mean("composite_score"),
            "avg_latency_s":        mean("latency_s"),
            "avg_extracted_count":  mean("extracted_count"),
        })

    # Sort best → worst
    aggregated.sort(key=lambda x: x["composite_score"], reverse=True)
    for rank, row in enumerate(aggregated, 1):
        row["rank"] = rank

    return aggregated
