"""
reports/report_generator.py
─────────────────────────────────────────────────────────────────────────────
Generates three output artefacts from the evaluated benchmark scores:

  1. outputs/benchmark_report.csv   – flat table, easy to open in Excel / Sheets
  2. outputs/benchmark_report.json  – structured JSON for downstream use
  3. outputs/benchmark_report.html  – rich, colour-coded HTML report with:
       • Leaderboard table (all 30 models ranked)
       • Per-metric bar chart (via Chart.js CDN)
       • Per-resume breakdown table

Public API
──────────
  generate_report(per_resume_scores, aggregated_scores, output_dir) -> str
      Writes all three files; returns the path to the HTML report.
─────────────────────────────────────────────────────────────────────────────
"""

import csv
import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

METRICS = [
    "extraction_accuracy",
    "skill_coverage",
    "years_correctness",
    "response_consistency",
    "composite_score",
]

METRIC_LABELS = {
    "extraction_accuracy":  "Extraction Accuracy",
    "skill_coverage":       "Skill Coverage",
    "years_correctness":    "Years Correctness",
    "response_consistency": "Response Consistency",
    "composite_score":      "Composite Score",
}


# ─────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────

def _write_csv(aggregated: list, path: str):
    fields = [
        "rank", "model_name", "model_id", "provider",
        "composite_score", "extraction_accuracy", "skill_coverage",
        "years_correctness", "response_consistency",
        "avg_latency_s", "avg_extracted_count",
        "resumes_tested", "error_count",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(aggregated)
    logger.info(f"CSV report → {path}")


# ─────────────────────────────────────────────────────────────────────────
# JSON
# ─────────────────────────────────────────────────────────────────────────

def _write_json(aggregated: list, per_resume: list, path: str):
    payload = {
        "generated_at":   datetime.utcnow().isoformat(),
        "leaderboard":    aggregated,
        "per_resume":     per_resume,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logger.info(f"JSON report → {path}")


# ─────────────────────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────────────────────

def _score_colour(value: float) -> str:
    """Return a CSS background colour based on the score (0–1 scale)."""
    if value >= 0.80:
        return "#d4edda"   # green
    if value >= 0.60:
        return "#fff3cd"   # yellow
    if value >= 0.40:
        return "#fde8d8"   # orange
    return "#f8d7da"       # red


def _fmt(value, pct=True) -> str:
    if value is None:
        return "–"
    if pct:
        return f"{value * 100:.1f}%"
    return f"{value:.2f}"


def _leaderboard_rows(aggregated: list) -> str:
    rows = []
    for row in aggregated:
        rank      = row["rank"]
        medal     = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, str(rank))
        comp      = row["composite_score"]
        bg        = _score_colour(comp)
        error_txt = (
            f'<span style="color:#dc3545">({row["error_count"]} errors)</span>'
            if row["error_count"] else ""
        )
        cells = "".join(
            f'<td style="background:{_score_colour(row[m])};text-align:center">'
            f'{_fmt(row[m])}</td>'
            for m in METRICS
        )
        rows.append(
            f"<tr>"
            f'<td style="text-align:center">{medal}</td>'
            f"<td><strong>{row['model_name']}</strong><br>"
            f'<small style="color:#666">{row["model_id"]}</small> {error_txt}</td>'
            f'<td style="text-align:center">{row["provider"]}</td>'
            f'<td style="text-align:center">{row["resumes_tested"]}</td>'
            f"{cells}"
            f'<td style="text-align:center">{row["avg_latency_s"]:.2f}s</td>'
            f"</tr>"
        )
    return "\n".join(rows)


def _per_resume_rows(per_resume: list) -> str:
    rows = []
    for row in per_resume:
        cells = "".join(
            f'<td style="background:{_score_colour(row[m])};text-align:center">'
            f'{_fmt(row[m])}</td>'
            for m in METRICS
        )
        err_icon = "❌" if row["had_error"] else "✅"
        rows.append(
            f"<tr>"
            f"<td>{row['model_name']}</td>"
            f"<td>{row['resume_filename']}</td>"
            f'<td style="text-align:center">{err_icon}</td>'
            f'<td style="text-align:center">{row["extracted_count"]}</td>'
            f"{cells}"
            f'<td style="text-align:center">{row["latency_s"]:.2f}s</td>'
            f"</tr>"
        )
    return "\n".join(rows)


def _chart_data(aggregated: list) -> str:
    """Return a JS object literal for Chart.js."""
    labels  = json.dumps([r["model_name"] for r in aggregated])
    datasets = []
    palette = [
        "#4e79a7","#f28e2b","#e15759","#76b7b2",
        "#59a14f","#edc948","#b07aa1","#ff9da7",
        "#9c755f","#bab0ac",
    ]
    for idx, metric in enumerate(METRICS):
        colour = palette[idx % len(palette)]
        values = json.dumps([r[metric] for r in aggregated])
        datasets.append(
            f'{{"label":"{METRIC_LABELS[metric]}",'
            f'"data":{values},'
            f'"backgroundColor":"{colour}88",'
            f'"borderColor":"{colour}",'
            f'"borderWidth":1}}'
        )
    return f'{{"labels":{labels},"datasets":[{",".join(datasets)}]}}'


def _write_html(aggregated: list, per_resume: list, path: str):
    """Write a self-contained HTML report with leaderboard + bar chart."""
    top_model = aggregated[0]["model_name"] if aggregated else "N/A"
    top_score = aggregated[0]["composite_score"] if aggregated else 0.0
    generated = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    chart_data = _chart_data(aggregated)

    header_cols = "".join(
        f"<th>{METRIC_LABELS[m]}</th>" for m in METRICS
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLM Resume Parser Benchmark Report</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body   {{ font-family: Arial, sans-serif; background: #f0f2f5; color: #212529; }}
    header {{ background: linear-gradient(135deg,#1f3864,#2e75b6); color:#fff;
              padding: 28px 40px; }}
    header h1 {{ font-size: 1.8rem; }}
    header p  {{ opacity: .8; margin-top: 6px; font-size: .9rem; }}
    .badge {{ display:inline-block; background:#fff; color:#1f3864;
              font-weight:700; border-radius:8px; padding:4px 14px;
              margin-top:12px; font-size:1rem; }}
    .container {{ max-width: 1400px; margin: 30px auto; padding: 0 20px; }}
    section {{ background:#fff; border-radius:10px; box-shadow:0 2px 8px #0001;
               margin-bottom:30px; padding:24px; }}
    h2  {{ font-size:1.2rem; margin-bottom:16px; color:#1f3864; }}
    table {{ width:100%; border-collapse:collapse; font-size:.85rem; }}
    th   {{ background:#1f3864; color:#fff; padding:10px 8px;
            position:sticky; top:0; white-space:nowrap; }}
    td   {{ padding:8px; border-bottom:1px solid #e9ecef; vertical-align:middle; }}
    tr:hover td {{ filter:brightness(.96); }}
    .chart-wrap {{ position:relative; height:420px; }}
    .tabs {{ display:flex; gap:8px; margin-bottom:16px; }}
    .tab  {{ padding:7px 18px; border-radius:6px; cursor:pointer;
             background:#e9ecef; border:none; font-size:.85rem; }}
    .tab.active {{ background:#1f3864; color:#fff; }}
    .tab-content {{ display:none; }}
    .tab-content.active {{ display:block; overflow-x:auto; }}
    small {{ color:#888; }}
  </style>
</head>
<body>
<header>
  <h1>🤖 LLM Resume Parser — Benchmark Report</h1>
  <p>Generated: {generated} &nbsp;|&nbsp; {len(aggregated)} models tested</p>
  <div class="badge">🏆 Best model: {top_model} &nbsp;({_fmt(top_score)} composite)</div>
</header>

<div class="container">

  <!-- Leaderboard -->
  <section>
    <h2>📊 Model Leaderboard</h2>
    <div style="overflow-x:auto">
      <table>
        <thead>
          <tr>
            <th>#</th><th>Model</th><th>Provider</th><th>Resumes</th>
            {header_cols}
            <th>Avg Latency</th>
          </tr>
        </thead>
        <tbody>
          {_leaderboard_rows(aggregated)}
        </tbody>
      </table>
    </div>
  </section>

  <!-- Bar chart -->
  <section>
    <h2>📈 Score Comparison by Metric</h2>
    <div class="chart-wrap">
      <canvas id="chartMain"></canvas>
    </div>
  </section>

  <!-- Per-resume details (tabbed) -->
  <section>
    <h2>📄 Per-Resume Details</h2>
    <div style="overflow-x:auto">
      <table>
        <thead>
          <tr>
            <th>Model</th><th>Resume</th><th>OK?</th><th>Skills Found</th>
            {header_cols}
            <th>Latency</th>
          </tr>
        </thead>
        <tbody>
          {_per_resume_rows(per_resume)}
        </tbody>
      </table>
    </div>
  </section>

</div>

<script>
const ctx = document.getElementById("chartMain").getContext("2d");
new Chart(ctx, {{
  type: "bar",
  data: {chart_data},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ position:"top" }} }},
    scales: {{
      y: {{ min:0, max:1, ticks:{{ callback: v => (v*100).toFixed(0)+"%" }} }},
      x: {{ ticks: {{ maxRotation:45, minRotation:30 }} }}
    }}
  }}
}});
</script>
</body>
</html>
"""

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    logger.info(f"HTML report → {path}")


# ─────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────

def generate_report(
    per_resume_scores: list,
    aggregated_scores: list,
    output_dir: str = "outputs",
) -> str:
    """
    Write CSV, JSON, and HTML benchmark reports.

    Parameters
    ----------
    per_resume_scores : list – from evaluator.evaluate_results()
    aggregated_scores : list – from evaluator.aggregate_by_model()
    output_dir        : str  – folder to write output files

    Returns
    -------
    str – path to the HTML report
    """
    os.makedirs(output_dir, exist_ok=True)

    csv_path  = os.path.join(output_dir, "benchmark_report.csv")
    json_path = os.path.join(output_dir, "benchmark_report.json")
    html_path = os.path.join(output_dir, "benchmark_report.html")

    _write_csv(aggregated_scores, csv_path)
    _write_json(aggregated_scores, per_resume_scores, json_path)
    _write_html(aggregated_scores, per_resume_scores, html_path)

    if aggregated_scores:
        best = aggregated_scores[0]
        logger.info(
            f"\n{'='*60}\n"
            f"  🏆  WINNER: {best['model_name']}\n"
            f"       Composite Score : {best['composite_score']*100:.1f}%\n"
            f"       Extraction Acc  : {best['extraction_accuracy']*100:.1f}%\n"
            f"       Skill Coverage  : {best['skill_coverage']*100:.1f}%\n"
            f"       Years Correct.  : {best['years_correctness']*100:.1f}%\n"
            f"       Consistency     : {best['response_consistency']*100:.1f}%\n"
            f"{'='*60}"
        )

    return html_path
