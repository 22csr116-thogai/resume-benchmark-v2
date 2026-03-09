"""
benchmarking/benchmark_runner.py
─────────────────────────────────────────────────────────────────────────────
Orchestrates the full benchmarking loop:

  For every enabled model × every resume:
    1. Build the prompt from the resume text.
    2. Call the LLM (via llm_client.call_llm).
    3. Parse the JSON response into a structured skills list.
    4. Record timing, raw response, parsed output, and any errors.
    5. Run CONSISTENCY_RUNS additional calls (same model + resume) to
       measure response stability.

Results are saved to:
  outputs/raw_results.json   – full log of every call
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import os
import re
import time
from datetime import datetime

from config import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    CONSISTENCY_RUNS,
    OUTPUT_DIR,
    ENABLED_MODELS,
)
from benchmarking.llm_client import call_llm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# JSON extraction helpers
# ─────────────────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> dict | None:
    """
    Attempt to extract a JSON object from the raw model response.

    Models sometimes wrap JSON in markdown fences (```json ... ```) or add
    prose before/after.  We try three strategies in order:
      1. Direct json.loads on the stripped string.
      2. Strip ```json … ``` fences then parse.
      3. Find the first {...} block with a regex and parse that.

    Returns None if none of the strategies succeed.
    """
    # Strategy 1 – direct parse
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2 – strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3 – grab first {...} block
    match = re.search(r"\{[\s\S]+\}", raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def _normalise_skills(parsed: dict | None) -> list:
    """
    Extract the 'skills' list from the parsed JSON, normalising edge cases:
      - Skills list might be under 'skills', 'technical_skills', etc.
      - Each item should have 'skill' (str) and 'years' (float or None).
      - Deduplicates skills (case-insensitive), keeping highest years value.
      - Filters out noise entries (vague categories, company names, achievements).
    """
    if not parsed:
        return []

    # Find the skills list regardless of key name
    skills_raw = (
        parsed.get("skills")
        or parsed.get("technical_skills")
        or parsed.get("extracted_skills")
        or []
    )

    # ── Noise blocklist: vague categories / non-skills to strip out ────────
    NOISE_TERMS = {
        "version control", "monitoring", "containerization", "infrastructure as code",
        "iac", "vm", "h", "ci/cd", "cloud security", "operating systems",
        "security tools", "security frameworks", "scripting", "networking",
        "cloud deployment", "app debugging", "performance optimization",
        "performance monitoring", "config management", "troubleshooting",
        "automation scripts", "monitoring & logging", "app security",
        "location tracking", "bluetooth connectivity", "google play store",
        "ml model development", "deep learning systems", "distributed computing frameworks",
        "real-time monitoring pipelines", "anomaly detection models",
        "collaborative filtering", "time-series analysis", "iot data streams",
        "cloud infrastructure automation", "devops automation scripts",
        "ui/ux", "agile", "agile development", "agile dev",
        "version ctrl", "programming / scripting", "operating systems",
        "security frameworks", "security tools", "network protocols",
        "fraud detection", "open-source ml project contribution",
    }

    # ── Parse all items ────────────────────────────────────────────────────
    raw_list = []
    for item in skills_raw:
        if isinstance(item, str):
            raw_list.append({"skill": item.strip(), "years": None})
            continue
        if not isinstance(item, dict):
            continue

        skill_name = (
            item.get("skill")
            or item.get("name")
            or item.get("technology")
            or ""
        ).strip()

        years_raw = item.get("years") or item.get("experience") or item.get("duration")
        try:
            years = float(years_raw) if years_raw is not None else None
            # Enforce minimum of 0.5 for any real skill
            if years is not None and years < 0.5:
                years = 0.5
        except (TypeError, ValueError):
            years = None

        if skill_name:
            raw_list.append({"skill": skill_name, "years": years})

    # ── Filter noise ───────────────────────────────────────────────────────
    filtered = [
        s for s in raw_list
        if s["skill"].lower() not in NOISE_TERMS
        and len(s["skill"]) > 1          # remove single-char entries like "H"
        and not s["skill"][0].isdigit()  # remove entries starting with digits
    ]

    # ── Deduplicate: keep entry with highest years (case-insensitive key) ──
    seen: dict = {}   # normalised_name → best entry so far
    for item in filtered:
        key = item["skill"].lower().strip()
        if key not in seen:
            seen[key] = item
        else:
            # Keep the one with a non-null, higher years value
            existing_years = seen[key]["years"]
            new_years      = item["years"]
            if new_years is not None:
                if existing_years is None or new_years > existing_years:
                    seen[key] = item

    return list(seen.values())


# ─────────────────────────────────────────────────────────────────────────
# Core benchmark function
# ─────────────────────────────────────────────────────────────────────────

def run_single(model_cfg: dict, resume: dict) -> dict:
    """
    Run one model against one resume (primary call + consistency runs).

    Returns
    -------
    dict with keys:
      model_name, resume_filename,
      primary   : {raw, parsed_skills, latency_s, error},
      consistency: list of {raw, parsed_skills, latency_s, error}
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(resume_text=resume["text"])
    result = {
        "model_name":      model_cfg["name"],
        "model_id":        model_cfg["model_id"],
        "provider":        model_cfg.get("provider", "ollama"),
        "resume_filename": resume["filename"],
        "timestamp":       datetime.utcnow().isoformat(),
        "primary":         {},
        "consistency":     [],
    }

    # ── Primary call ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        raw = call_llm(model_cfg, SYSTEM_PROMPT, user_prompt)
        latency = round(time.perf_counter() - t0, 3)
        parsed  = _extract_json(raw)
        skills  = _normalise_skills(parsed)
        result["primary"] = {
            "raw":           raw,
            "parsed_skills": skills,
            "latency_s":     latency,
            "error":         None,
        }
        logger.info(
            f"  [{model_cfg['name']}] → {len(skills)} skills | {latency}s"
        )
    except Exception as exc:
        latency = round(time.perf_counter() - t0, 3)
        result["primary"] = {
            "raw":           "",
            "parsed_skills": [],
            "latency_s":     latency,
            "error":         str(exc),
        }
        logger.warning(f"  [{model_cfg['name']}] primary call failed: {exc}")

    # ── Consistency runs ──────────────────────────────────────────────────
    for run_idx in range(CONSISTENCY_RUNS):
        t0 = time.perf_counter()
        try:
            raw = call_llm(model_cfg, SYSTEM_PROMPT, user_prompt)
            latency = round(time.perf_counter() - t0, 3)
            parsed  = _extract_json(raw)
            skills  = _normalise_skills(parsed)
            result["consistency"].append({
                "run":           run_idx + 1,
                "raw":           raw,
                "parsed_skills": skills,
                "latency_s":     latency,
                "error":         None,
            })
        except Exception as exc:
            latency = round(time.perf_counter() - t0, 3)
            result["consistency"].append({
                "run":           run_idx + 1,
                "raw":           "",
                "parsed_skills": [],
                "latency_s":     latency,
                "error":         str(exc),
            })

    return result


def run_benchmark(
    resumes:     list,
    models:      list | None = None,
    save_path:   str  | None = None,
) -> list:
    """
    Full benchmarking loop: every model × every resume.

    Parameters
    ----------
    resumes   : list of dicts from parsers.resume_parser.load_all_resumes()
    models    : list of model config dicts (defaults to config.ENABLED_MODELS)
    save_path : path to write raw_results.json (defaults to OUTPUT_DIR/raw_results.json)

    Returns
    -------
    list of result dicts (one per model–resume pair)
    """
    if models is None:
        models = ENABLED_MODELS

    if save_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, "raw_results.json")

    all_results = []
    total = len(models) * len(resumes)
    done  = 0

    logger.info(
        f"Starting benchmark: {len(models)} models × {len(resumes)} resumes "
        f"= {total} primary calls  (+{CONSISTENCY_RUNS} consistency runs each)"
    )

    for model_cfg in models:
        logger.info(f"\n▶  Model: {model_cfg['name']}  ({model_cfg['model_id']})")
        for resume in resumes:
            done += 1
            logger.info(
                f"  ({done}/{total}) Resume: {resume['filename']}"
            )
            result = run_single(model_cfg, resume)
            all_results.append(result)

            # Persist after every call so partial results are not lost
            with open(save_path, "w", encoding="utf-8") as fh:
                json.dump(all_results, fh, indent=2, ensure_ascii=False)

    logger.info(f"\nBenchmark complete. Results saved → {save_path}")
    return all_results
