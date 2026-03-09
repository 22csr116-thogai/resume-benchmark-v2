"""
benchmarking/llm_client.py
─────────────────────────────────────────────────────────────────────────────
Unified LLM client that abstracts provider differences.

Supported providers
───────────────────
  "ollama"      → local Ollama REST API  (/api/chat)
  "openai"      → OpenAI ChatCompletion  (requires OPENAI_API_KEY env var)
  "anthropic"   → Anthropic Messages API (requires ANTHROPIC_API_KEY env var)
  "huggingface" → HuggingFace pipeline   (local GPU inference via transformers)

Public API
──────────
  call_llm(model_cfg: dict, system_prompt: str, user_prompt: str) -> str
      Sends prompts to the correct backend and returns the raw response string.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import logging
import requests
import time

logger = logging.getLogger(__name__)

# ── Optional heavyweight imports (only loaded when needed) ────────────────
_hf_pipeline_cache: dict = {}   # model_id → pipeline object


def _get_hf_pipeline(model_id: str, max_tokens: int):
    """Load (and cache) a HuggingFace text-generation pipeline."""
    if model_id not in _hf_pipeline_cache:
        try:
            from transformers import pipeline
            import torch
            device = 0 if torch.cuda.is_available() else -1
            logger.info(f"Loading HuggingFace model '{model_id}' on device={device} …")
            _hf_pipeline_cache[model_id] = pipeline(
                "text-generation",
                model=model_id,
                device=device,
                max_new_tokens=max_tokens,
                truncation=True,
            )
            logger.info(f"Model '{model_id}' loaded.")
        except ImportError:
            raise RuntimeError(
                "transformers / torch not installed. "
                "Run: pip install transformers accelerate torch"
            )
    return _hf_pipeline_cache[model_id]


# ─────────────────────────────────────────────────────────────────────────
# Provider implementations
# ─────────────────────────────────────────────────────────────────────────

def _call_ollama(model_cfg: dict, system_prompt: str, user_prompt: str) -> str:
    """
    Call a local Ollama server using its /api/chat endpoint.
    Ollama must be running: `ollama serve`
    Pull the model first: `ollama pull <model_id>`
    """
    from config import OLLAMA_BASE_URL

    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model":    model_cfg["model_id"],
        "messages": [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature":   model_cfg.get("temperature", 0.1),
            "num_predict":   model_cfg.get("max_tokens",  1024),
        },
    }

    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"].strip()


def _call_openai(model_cfg: dict, system_prompt: str, user_prompt: str) -> str:
    """Call the OpenAI ChatCompletion API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model_cfg["model_id"],
        messages=[
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_prompt},
        ],
        max_tokens=model_cfg.get("max_tokens",  1024),
        temperature=model_cfg.get("temperature", 0.1),
    )
    return response.choices[0].message.content.strip()


def _call_anthropic(model_cfg: dict, system_prompt: str, user_prompt: str) -> str:
    """Call the Anthropic Messages API."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model=model_cfg["model_id"],
        max_tokens=model_cfg.get("max_tokens", 1024),
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text.strip()


def _call_huggingface(model_cfg: dict, system_prompt: str, user_prompt: str) -> str:
    """Run inference with a locally loaded HuggingFace model."""
    pipe = _get_hf_pipeline(model_cfg["model_id"], model_cfg.get("max_tokens", 1024))
    # Combine system + user for models that don't support a separate system role
    combined_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>"
    output = pipe(combined_prompt)[0]["generated_text"]
    # Strip the prompt prefix so we return only the model's continuation
    if combined_prompt in output:
        output = output.replace(combined_prompt, "").strip()
    return output


# ─────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────

PROVIDER_MAP = {
    "ollama":      _call_ollama,
    "openai":      _call_openai,
    "anthropic":   _call_anthropic,
    "huggingface": _call_huggingface,
}


def call_llm(
    model_cfg:     dict,
    system_prompt: str,
    user_prompt:   str,
    retries:       int = 2,
    retry_delay:   float = 3.0,
) -> str:
    """
    Send prompts to the appropriate LLM backend and return the response.

    Parameters
    ----------
    model_cfg     : dict  — one entry from config.MODELS
    system_prompt : str   — role/instruction context
    user_prompt   : str   — the actual resume text wrapped in a template
    retries       : int   — number of retry attempts on transient failure
    retry_delay   : float — seconds between retries

    Returns
    -------
    str
        Raw model response (expected to be JSON, but may not be on error).

    Raises
    ------
    ValueError  if the provider is not recognised.
    RuntimeError if all retry attempts fail.
    """
    provider = model_cfg.get("provider", "ollama").lower()
    if provider not in PROVIDER_MAP:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported: {list(PROVIDER_MAP.keys())}"
        )

    caller = PROVIDER_MAP[provider]
    last_exc = None

    for attempt in range(1, retries + 2):   # +2: first try + retries
        try:
            logger.debug(
                f"[{model_cfg['name']}] attempt {attempt} via '{provider}'"
            )
            return caller(model_cfg, system_prompt, user_prompt)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                f"[{model_cfg['name']}] attempt {attempt} failed: {exc}"
            )
            if attempt <= retries:
                time.sleep(retry_delay)

    raise RuntimeError(
        f"All {retries + 1} attempts failed for model '{model_cfg['name']}': {last_exc}"
    )
