"""Gradio event handlers and chat history normalization."""

from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

from .config import LOGGER
from .model import contains_malformed_output, stream_model_reply
from .prompts import CRISIS_MESSAGE, LOADING_MESSAGE, contains_crisis_language


def extract_message_text(content: Any) -> str:
    """Read text from plain and Gradio-normalized Chatbot message content."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
            elif hasattr(item, "text") and isinstance(item.text, str):
                text_parts.append(item.text)
        return "\n".join(text_parts)

    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return content["text"]

    if hasattr(content, "text") and isinstance(content.text, str):
        return content.text

    return ""


def normalize_history(history: Optional[Sequence[Any]]) -> List[Dict[str, str]]:
    """Accept modern messages or legacy tuple history and return chat messages."""
    messages: List[Dict[str, str]] = []

    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role")
            content = extract_message_text(item.get("content"))
            if role in {"user", "assistant"} and content.strip():
                if role == "assistant" and contains_malformed_output(content):
                    continue
                messages.append({"role": role, "content": content})
            continue

        if hasattr(item, "role") and hasattr(item, "content"):
            role = getattr(item, "role")
            content = extract_message_text(getattr(item, "content"))
            if role in {"user", "assistant"} and content.strip():
                if role == "assistant" and contains_malformed_output(content):
                    continue
                messages.append({"role": role, "content": content})
            continue

        if isinstance(item, (list, tuple)) and len(item) == 2:
            user_text, assistant_text = item
            if isinstance(user_text, str) and user_text.strip():
                messages.append({"role": "user", "content": user_text})
            if isinstance(assistant_text, str) and assistant_text.strip():
                messages.append({"role": "assistant", "content": assistant_text})

    return messages


def add_user_message(
    message: str,
    history: Optional[Sequence[Any]],
) -> Tuple[List[Dict[str, str]], str]:
    clean_message = (message or "").strip()
    messages = normalize_history(history)

    if not clean_message:
        return messages, ""

    messages.append({"role": "user", "content": clean_message})
    return messages, ""


def add_quick_prompt(
    prompt: str,
    history: Optional[Sequence[Any]],
) -> Tuple[List[Dict[str, str]], str]:
    return add_user_message(prompt, history)


def generate_assistant_reply(
    history: Optional[Sequence[Any]],
) -> Generator[List[Dict[str, str]], None, None]:
    messages = normalize_history(history)

    if not messages or messages[-1]["role"] != "user":
        yield messages
        return

    latest_user_message = messages[-1]["content"]

    if contains_crisis_language(latest_user_message):
        messages.append({"role": "assistant", "content": CRISIS_MESSAGE})
        yield messages
        return

    messages.append({"role": "assistant", "content": LOADING_MESSAGE})
    yield messages

    try:
        for token in stream_model_reply(messages[:-1]):
            if messages[-1]["content"] == LOADING_MESSAGE:
                messages[-1]["content"] = ""
            messages[-1]["content"] += token
            yield messages
    except (FileNotFoundError, RuntimeError) as exc:
        LOGGER.warning("SolaceLLM is not ready: %s", exc)
        messages[-1]["content"] = (
            "I couldn't reach SolaceLLM locally yet. "
            f"{exc}\n\nCheck that `llama-cpp-python` is installed and either "
            "`SOLACE_MODEL_PATH` points to a local GGUF file or `SOLACE_MODEL_REPO` "
            "points to a Hugging Face GGUF repo."
        )
        yield messages
    except Exception as exc:  # pragma: no cover - depends on local runtime setup.
        LOGGER.exception("Unable to generate a SolaceLLM response")
        messages[-1]["content"] = (
            "I couldn't reach SolaceLLM locally yet. "
            f"{exc}\n\nCheck that `llama-cpp-python` is installed and either "
            "`SOLACE_MODEL_PATH` points to a local GGUF file or `SOLACE_MODEL_REPO` "
            "points to a Hugging Face GGUF repo."
        )
        yield messages

