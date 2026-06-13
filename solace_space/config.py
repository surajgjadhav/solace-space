"""Configuration and shared runtime helpers for Solace Space."""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any


APP_TITLE = "Solace Space"
APP_SUBTITLE = "Powered by SolaceLLM — Your Inside Out Emotional Companion"

MODEL_PATH = os.getenv("SOLACE_MODEL_PATH", "").strip()
MODEL_REPO = os.getenv("SOLACE_MODEL_REPO", "build-small-hackathon/solace-llm-GGUF").strip()
MODEL_FILE_NAME = os.getenv("SOLACE_MODEL_FILE", "*Q4_K_M.gguf").strip()
N_CTX = int(os.getenv("SOLACE_N_CTX", "2048"))
MAX_TOKENS = int(os.getenv("SOLACE_MAX_TOKENS", "220"))
TEMPERATURE = float(os.getenv("SOLACE_TEMPERATURE", "0.72"))
TOP_P = float(os.getenv("SOLACE_TOP_P", "0.92"))
REPEAT_PENALTY = float(os.getenv("SOLACE_REPEAT_PENALTY", "1.15"))
FREQUENCY_PENALTY = float(os.getenv("SOLACE_FREQUENCY_PENALTY", "0.15"))
PRESENCE_PENALTY = float(os.getenv("SOLACE_PRESENCE_PENALTY", "0.05"))
MAX_REPEATED_SENTENCES = int(os.getenv("SOLACE_MAX_REPEATED_SENTENCES", "2"))
HISTORY_TOKEN_BUFFER = int(os.getenv("SOLACE_HISTORY_TOKEN_BUFFER", "128"))

MALFORMED_OUTPUT_PATTERNS = [
    r"\\input\s*\{",
    r"\\begin\s*\{",
    r"\\end\s*\{",
    r"^\s*\\item\b",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("solace-space")


def supports_parameter(callable_obj: Any, parameter_name: str) -> bool:
    """Return whether a callable exposes a specific keyword parameter."""
    try:
        return parameter_name in inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False

