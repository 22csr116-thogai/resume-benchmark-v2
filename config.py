"""
config.py
─────────────────────────────────────────────────────────────────────────────
Central configuration for the Resume Parser & LLM Benchmarking pipeline.

All 30 models are listed here. Each model entry specifies:
  - provider   : "ollama" | "openai" | "anthropic" | "huggingface"
  - model_id   : identifier used in API calls or HuggingFace hub
  - api_base   : base URL for the API (Ollama runs locally)
  - max_tokens : upper bound on generated tokens
  - temperature: sampling temperature (lower = more deterministic)
  - enabled    : set False to skip a model in a run
─────────────────────────────────────────────────────────────────────────────
"""

import os

# ── Ollama server settings ────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── Paths ─────────────────────────────────────────────────────────────────
RESUME_DIR        = "data/resumes"          # folder holding PDF / DOCX / TXT resumes
GROUND_TRUTH_FILE = "data/ground_truth/ground_truth.json"
OUTPUT_DIR        = "outputs"

# ── Evaluation weights ────────────────────────────────────────────────────
EVAL_WEIGHTS = {
    "extraction_accuracy":      0.30,   # % of expected skills correctly found
    "skill_coverage":           0.25,   # breadth of skills found vs ground truth
    "years_correctness":        0.30,   # years-of-experience accuracy
    "response_consistency":     0.15,   # same model, same resume → stable output
}

# ── Number of consistency runs per model per resume ───────────────────────
CONSISTENCY_RUNS = 3

# ── Prompt sent to every LLM ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert resume parser specialising in technical skill extraction.

TASK
────
Extract every specific technical skill from the resume and calculate years of
experience for each one.

WHAT TO EXTRACT  ✅
───────────────────
Include ONLY concrete, named technologies such as:
  • Programming languages  : Python, Java, Kotlin, Go, SQL, Bash, etc.
  • Frameworks / libraries : React.js, Django, Spring Boot, Flutter, TensorFlow, etc.
  • Databases              : PostgreSQL, MongoDB, MySQL, Redis, Snowflake, etc.
  • Cloud platforms        : AWS, Azure, GCP and their specific services (EC2, S3, Lambda…)
  • DevOps / infra tools   : Docker, Kubernetes, Terraform, Jenkins, Ansible, GitHub Actions…
  • Security tools         : Splunk, Wireshark, Nessus, Metasploit, Burp Suite, etc.
  • Testing tools          : Selenium, JUnit, Pytest, Postman, JMeter, Cypress, etc.
  • Data / ML tools        : Pandas, Spark, Airflow, MLflow, Tableau, Power BI, etc.
  • Protocols / standards  : TCP/IP, REST APIs, GraphQL, OAuth2, JWT, etc.

WHAT TO EXCLUDE  ❌
───────────────────
Do NOT include any of the following — these are NOT skills:
  • Soft skills            : communication, teamwork, leadership, problem-solving
  • Vague category headings: "Version Control", "Monitoring", "Containerization",
                             "Infrastructure as Code", "Cloud Security", "Operating Systems",
                             "Security Tools", "Security Frameworks", "Scripting"
  • Job titles or roles    : "Backend Developer", "DevOps Engineer"
  • Company / institute names: "DataCore Technologies", "Insight Analytics"
  • Achievement text       : "CTF competition", "ISSA member", "Open-source contributor"
  • Generic abbreviations  : "IaC", "VM", "H", "CI/CD" (use the full tool name instead)
  • Competency phrases     : "ML Model Development", "App Debugging", "Performance Optimization",
                             "Threat Detection", "Deep Learning Systems"

YEARS OF EXPERIENCE — HOW TO CALCULATE
───────────────────────────────────────
1. Find all job roles with date ranges in the resume (e.g. "2021 – 2023" = 2 years,
   "2023 – Present" = calculate to current year 2025, so 2 years).
2. For each skill, identify WHICH roles mention or use that skill.
3. Sum the durations of those roles to get total years for that skill.
4. If a skill appears in the Technical Skills section but is NOT mentioned in any
   specific role, assign years = total career length as an approximation.
5. Only set years = null if there are NO date ranges anywhere in the resume at all.
6. Round years to 1 decimal place. Minimum value is 0.5 (never use 0.1 for a real skill).

DEDUPLICATION
─────────────
Each skill must appear ONLY ONCE in your output.
If you find the same skill listed multiple times, keep only ONE entry with the
highest or most accurate years value.

OUTPUT FORMAT
─────────────
Return ONLY valid JSON — no prose, no markdown fences, no explanation:
{
  "skills": [
    {"skill": "Python",     "years": 4.0},
    {"skill": "Docker",     "years": 2.0},
    {"skill": "PostgreSQL", "years": 4.0}
  ]
}"""

USER_PROMPT_TEMPLATE = """Extract technical skills from the resume below.
Remember:
- Specific tools and technologies ONLY (no vague categories, no soft skills, no company names)
- Each skill appears exactly ONCE — no duplicates
- Calculate years from job history dates
- Return JSON only, no extra text

RESUME:
{resume_text}"""

# ── Model registry ────────────────────────────────────────────────────────
# provider options:
#   "ollama"       → local Ollama REST API
#   "openai"       → OpenAI-compatible API (set OPENAI_API_KEY env var)
#   "anthropic"    → Anthropic API  (set ANTHROPIC_API_KEY env var)
#   "huggingface"  → HuggingFace Transformers (loaded locally / Colab GPU)

MODELS = [
    # ── Ollama-hosted models (all 30 mapped to Ollama provider) ──────────
    {
        "name":        "Gemma 2 9B",
        "provider":    "ollama",
        "model_id":    "gemma2:9b",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
    {
        "name":        "Pixtral 12B",
        "provider":    "ollama",
        "model_id":    "pixtral:12b",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
    {
        "name":        "Gemma 3n E4B",
        "provider":    "ollama",
        "model_id":    "gemma3n:e4b",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
    {
        "name":        "OLMo 2 7B Instruct",
        "provider":    "ollama",
        "model_id":    "olmo2:7b",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
    {
        "name":        "WizardLM-2 7B",
        "provider":    "ollama",
        "model_id":    "wizardlm2:7b",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
    {
        "name":        "Jamba Mini 52B",
        "provider":    "ollama",
        "model_id":    "jamba-mini:latest",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
    {
        "name":        "DeepSeek-V3.2",
        "provider":    "ollama",
        "model_id":    "deepseek-v3:latest",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
    {
        "name":        "Llama 4 Maverick",
        "provider":    "ollama",
        "model_id":    "llama4:maverick",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
    {
        "name":        "Phi-4 14B",
        "provider":    "ollama",
        "model_id":    "phi4:14b",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
    {
        "name":        "Nous-Hermes-2-Mistral-7B-DPO",
        "provider":    "ollama",
        "model_id":    "nous-hermes2-mistral:7b-dpo",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
    {
        "name":        "Llama 3 8B",
        "provider":    "ollama",
        "model_id":    "llama3:8b",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
    {
        "name":        "Apertus 8B",
        "provider":    "ollama",
        "model_id":    "apertus:8b",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
    {
        "name":        "Dolphin 2.9 Mistral",
        "provider":    "ollama",
        "model_id":    "dolphin-mistral:latest",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
    {
        "name":        "Cogito v1 8B",
        "provider":    "ollama",
        "model_id":    "cogito:8b",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
    {
        "name":        "GLM-4.6",
        "provider":    "ollama",
        "model_id":    "glm4:latest",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
    {
        "name":        "MiniCPM4.1 8B",
        "provider":    "ollama",
        "model_id":    "minicpm:8b",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
    {
        "name":        "Mistral 7B v0.3",
        "provider":    "ollama",
        "model_id":    "mistral:7b-instruct-v0.3",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
    {
        "name":        "Gemma 3 4B",
        "provider":    "ollama",
        "model_id":    "gemma3:4b",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
    {
        "name":        "Falcon Mamba 7B",
        "provider":    "ollama",
        "model_id":    "falcon-mamba:7b",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
    {
        "name":        "Gemma 3 12B",
        "provider":    "ollama",
        "model_id":    "gemma3:12b",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
    {
        "name":        "DeepSeek-R1-0528-Qwen3-8B",
        "provider":    "ollama",
        "model_id":    "deepseek-r1:8b",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
    {
        "name":        "Qwen3-14B",
        "provider":    "ollama",
        "model_id":    "qwen3:14b",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
    {
        "name":        "XGen-7B",
        "provider":    "ollama",
        "model_id":    "xgen:7b",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
    {
        "name":        "DeepSeek-R1-Distill-Qwen-7B",
        "provider":    "ollama",
        "model_id":    "deepseek-r1:7b",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
    {
        "name":        "Apriel-1.5-15B-Thinker",
        "provider":    "ollama",
        "model_id":    "apriel:15b",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
    {
        "name":        "Llama 4 Scout",
        "provider":    "ollama",
        "model_id":    "llama4:scout",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
    {
        "name":        "InternLM2 Chat 20B",
        "provider":    "ollama",
        "model_id":    "internlm2:20b",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
    {
        "name":        "Hermes 3 Llama 3.1 8B",
        "provider":    "ollama",
        "model_id":    "hermes3:8b",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
    {
        "name":        "Zephyr 7B",
        "provider":    "ollama",
        "model_id":    "zephyr:7b",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
    {
        "name":        "GPT-OSS-20B",
        "provider":    "ollama",
        "model_id":    "gpt-oss:20b",
        "max_tokens":  1024,
        "temperature": 0.1,
        "enabled":     True,
    },
]

# ── Quick lookup: enabled models only ─────────────────────────────────────
ENABLED_MODELS = [m for m in MODELS if m.get("enabled", True)]
