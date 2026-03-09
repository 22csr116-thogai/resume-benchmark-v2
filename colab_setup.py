"""
colab_setup.py
─────────────────────────────────────────────────────────────────────────────
Run this script once at the start of a Colab session to:
  1. Install all required Python packages.
  2. Install and start Ollama inside the Colab runtime.
  3. Pull all 30 models from the Ollama registry.
  4. Upload your resume files to data/resumes/.

Usage in Colab
──────────────
  !python colab_setup.py             # install packages + start Ollama
  !python colab_setup.py --pull      # also pull all 30 models (slow!)
  !python colab_setup.py --pull --model "llama3:8b"   # pull one model
─────────────────────────────────────────────────────────────────────────────
"""

import subprocess
import sys
import os
import argparse
import time

# ── 1. Python packages ────────────────────────────────────────────────────
PACKAGES = [
    "pdfplumber",
    "python-docx",
    "requests",
    "tqdm",
    # Optional: uncomment if using HuggingFace models
    # "transformers",
    # "accelerate",
    # "torch",
]

def install_packages():
    print("Installing Python packages …")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet"] + PACKAGES
    )
    print("  ✓ Packages installed.")


# ── 2. Ollama installation ────────────────────────────────────────────────
def install_ollama():
    """Download and install Ollama in the Colab environment."""
    print("Installing Ollama …")
    result = subprocess.run(
        ["which", "ollama"], capture_output=True, text=True
    )
    if result.returncode == 0:
        print("  ✓ Ollama already installed.")
        return

    # Official Ollama install script
    subprocess.run(
        "curl -fsSL https://ollama.com/install.sh | sh",
        shell=True, check=True
    )
    print("  ✓ Ollama installed.")


def start_ollama():
    """Start the Ollama server as a background process."""
    print("Starting Ollama server …")
    # Check if already running
    result = subprocess.run(
        ["pgrep", "-x", "ollama"], capture_output=True
    )
    if result.returncode == 0:
        print("  ✓ Ollama already running.")
        return

    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3)   # give the server time to start

    # Verify
    import requests
    for attempt in range(10):
        try:
            r = requests.get("http://localhost:11434", timeout=2)
            if r.status_code == 200:
                print("  ✓ Ollama server is running on http://localhost:11434")
                return
        except Exception:
            time.sleep(2)
    print("  ⚠ Ollama server may not have started correctly. Check manually.")


# ── 3. Pull models ────────────────────────────────────────────────────────

# Map from config model_id to the exact Ollama pull tag.
# Adjust if Ollama's model registry uses different names.
ALL_MODEL_PULL_TAGS = [
    "gemma2:9b",
    "pixtral:12b",
    "gemma3n:e4b",
    "olmo2:7b",
    "wizardlm2:7b",
    "jamba-mini:latest",
    "deepseek-v3:latest",
    "llama4:maverick",
    "phi4:14b",
    "nous-hermes2-mistral:7b-dpo",
    "llama3:8b",
    "apertus:8b",
    "dolphin-mistral:latest",
    "cogito:8b",
    "glm4:latest",
    "minicpm:8b",
    "mistral:7b-instruct-v0.3",
    "gemma3:4b",
    "falcon-mamba:7b",
    "gemma3:12b",
    "deepseek-r1:8b",
    "qwen3:14b",
    "xgen:7b",
    "deepseek-r1:7b",
    "apriel:15b",
    "llama4:scout",
    "internlm2:20b",
    "hermes3:8b",
    "zephyr:7b",
    "gpt-oss:20b",
]


def pull_models(model_filter: str | None = None):
    """Pull models from Ollama registry."""
    tags = ALL_MODEL_PULL_TAGS
    if model_filter:
        tags = [t for t in tags if model_filter.lower() in t.lower()]

    print(f"Pulling {len(tags)} Ollama model(s) …")
    for tag in tags:
        print(f"  pulling {tag} …", end=" ", flush=True)
        result = subprocess.run(
            ["ollama", "pull", tag],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("✓")
        else:
            print(f"✗  {result.stderr.strip()[:80]}")


# ── 4. Directory setup ────────────────────────────────────────────────────
def setup_directories():
    dirs = [
        "data/resumes",
        "data/ground_truth",
        "outputs",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("  ✓ Directories created:", dirs)


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pull", action="store_true",
                        help="Pull all models from Ollama registry")
    parser.add_argument("--model", default=None,
                        help="Only pull models matching this string")
    args = parser.parse_args()

    install_packages()
    install_ollama()
    start_ollama()
    setup_directories()

    if args.pull:
        pull_models(args.model)

    print("\n✅  Setup complete. You can now run:  python main.py")
