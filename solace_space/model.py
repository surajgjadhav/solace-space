"""Model loading, history shaping, and streaming inference helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .config import (
    FREQUENCY_PENALTY,
    HISTORY_TOKEN_BUFFER,
    LOGGER,
    MALFORMED_OUTPUT_PATTERNS,
    MAX_REPEATED_SENTENCES,
    MAX_TOKENS,
    MODEL_FILE_NAME,
    MODEL_PATH,
    MODEL_REPO,
    N_CTX,
    PRESENCE_PENALTY,
    REPEAT_PENALTY,
    TEMPERATURE,
    TOP_P,
)
from .prompts import SYSTEM_PROMPT

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover - surfaced in the UI at runtime.
    Llama = None  # type: ignore[assignment]


MODEL: Optional["Llama"] = None


def get_model() -> "Llama":
    """Load the local GGUF model lazily and reuse it for future turns."""
    global MODEL

    if MODEL is not None:
        return MODEL

    if Llama is None:
        raise RuntimeError(
            "llama-cpp-python is not installed. Install dependencies with "
            "`pip install -r requirements.txt`."
        )

    if MODEL_PATH:
        local_model_path = Path(MODEL_PATH).expanduser()
        if not local_model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at `{local_model_path}`. Put your GGUF model "
                "there or set SOLACE_MODEL_PATH to the correct local file."
            )

        LOGGER.info("Loading SolaceLLM model from local file %s", local_model_path)
        MODEL = Llama(
            model_path=str(local_model_path),
            n_ctx=N_CTX,
            verbose=False,
        )
        return MODEL

    if not MODEL_REPO:
        raise RuntimeError(
            "No model source configured. Set SOLACE_MODEL_PATH to a local GGUF file "
            "or SOLACE_MODEL_REPO to a Hugging Face GGUF repository."
        )

    LOGGER.info(
        "Loading SolaceLLM model from Hugging Face repo %s (%s)",
        MODEL_REPO,
        MODEL_FILE_NAME,
    )
    MODEL = Llama.from_pretrained(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE_NAME,
        n_ctx=N_CTX,
        verbose=False,
    )
    return MODEL


def estimate_token_count(text: str) -> int:
    """Cheap token estimate used before the model tokenizer is loaded."""
    return max(1, (len(text) + 3) // 4)


def count_text_tokens(text: str, model: Optional["Llama"] = None) -> int:
    """Count tokens with llama-cpp when available, otherwise estimate."""
    if model is not None and hasattr(model, "tokenize"):
        try:
            return max(1, len(model.tokenize(text.encode("utf-8"), add_bos=False)))
        except Exception:
            LOGGER.debug("Falling back to estimated token count", exc_info=True)

    return estimate_token_count(text)


def message_token_cost(message: Dict[str, str], model: Optional["Llama"] = None) -> int:
    """Estimate token cost for a role-tagged chat message."""
    return count_text_tokens(message["content"], model) + 8


def available_history_tokens() -> int:
    """Reserve room for generation and formatting inside the model context."""
    return max(256, N_CTX - MAX_TOKENS - HISTORY_TOKEN_BUFFER)


def trim_history_to_context(
    history: Sequence[Dict[str, str]],
    model: Optional["Llama"] = None,
) -> List[Dict[str, str]]:
    """Keep as much recent history as fits instead of slicing by message count."""
    budget = available_history_tokens()
    selected_reversed: List[Dict[str, str]] = []
    used_tokens = count_text_tokens(SYSTEM_PROMPT, model)

    for message in reversed(list(history)):
        cost = message_token_cost(message, model)
        if selected_reversed and used_tokens + cost > budget:
            break

        selected_reversed.append(message)
        used_tokens += cost

    selected = list(reversed(selected_reversed))
    while len(selected) > 1 and selected[0]["role"] != "user":
        selected.pop(0)

    return selected


def build_messages(
    history: Sequence[Dict[str, str]],
    model: Optional["Llama"] = None,
) -> List[Dict[str, str]]:
    """Build llama-cpp chat messages with a bounded conversational window."""
    recent_history = trim_history_to_context(history, model)
    return [{"role": "system", "content": SYSTEM_PROMPT}, *recent_history]


def extract_stream_delta(chunk: Dict[str, Any]) -> str:
    """Read text from llama-cpp's OpenAI-compatible streaming chunks."""
    choices = chunk.get("choices") or []
    if not choices:
        return ""

    choice = choices[0]
    delta = choice.get("delta") or {}
    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
        return delta["content"]

    message = choice.get("message") or {}
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]

    text = choice.get("text")
    return text if isinstance(text, str) else ""


def normalize_sentence_for_repeat_check(sentence: str) -> str:
    """Normalize generated text to catch low-value repetition loops."""
    normalized = re.sub(r"\s+", " ", sentence.strip().lower())
    return normalized.strip(" .!?,;:\"'`")


def should_stop_for_repetition(text: str) -> bool:
    """Detect repeated sentence loops during streaming generation."""
    if MAX_REPEATED_SENTENCES <= 0:
        return False

    sentences = re.findall(r"[^.!?\n]+[.!?]", text)
    counts: Dict[str, int] = {}

    for sentence in sentences:
        normalized = normalize_sentence_for_repeat_check(sentence)
        if len(normalized) < 12:
            continue

        counts[normalized] = counts.get(normalized, 0) + 1
        if counts[normalized] > MAX_REPEATED_SENTENCES:
            return True

    lines = [normalize_sentence_for_repeat_check(line) for line in text.splitlines()]
    repeated_lines = 0
    previous_line = ""
    for line in lines:
        if len(line) < 12:
            continue
        if line == previous_line:
            repeated_lines += 1
            if repeated_lines >= MAX_REPEATED_SENTENCES:
                return True
        else:
            previous_line = line
            repeated_lines = 0

    return False


def contains_malformed_output(text: str) -> bool:
    """Detect formatting/control artifacts that should never appear in this chat."""
    return any(
        re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        for pattern in MALFORMED_OUTPUT_PATTERNS
    )


def stream_model_reply(history: Sequence[Dict[str, str]]) -> Iterable[str]:
    """Yield SolaceLLM response text chunks for the current chat history."""
    model = get_model()
    stream = model.create_chat_completion(
        messages=build_messages(history, model),
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        repeat_penalty=REPEAT_PENALTY,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
        stream=True,
    )

    generated_text = ""
    for chunk in stream:
        delta = extract_stream_delta(chunk)
        if delta:
            candidate_text = generated_text + delta
            if contains_malformed_output(candidate_text):
                LOGGER.warning("Stopping generation because malformed output was detected")
                break
            if should_stop_for_repetition(candidate_text):
                LOGGER.warning("Stopping generation because repeated text was detected")
                break
            generated_text = candidate_text
            yield delta

