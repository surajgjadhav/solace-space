"""Solace Space: a local emotional companion chat app powered by llama-cpp."""

from __future__ import annotations

import inspect
import logging
import os
import re
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence, Tuple

warnings.filterwarnings(
    "ignore",
    message=r"'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated.*",
    category=Warning,
    module=r"gradio\.routes",
)

import gradio as gr

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover - surfaced in the UI at runtime.
    Llama = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

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

MODEL: Optional["Llama"] = None


def supports_parameter(callable_obj: Any, parameter_name: str) -> bool:
    """Return whether a callable exposes a specific keyword parameter."""
    try:
        return parameter_name in inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Emotional companion behavior and guardrails
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are SolaceLLM, the local model behind Solace Space.

You are a compassionate emotional-support companion with a counseling-style
tone. You are not a licensed therapist, doctor, crisis worker, or diagnostic
tool. Do not diagnose the user, prescribe treatment, or claim to replace
professional care. You may provide reflective support, coping strategies,
communication guidance, and practical next steps.

Your goal is to understand the user's situation and feeling, then help them
work with the emotion in a grounded way. Respond like a calm consultation:
1. Name and validate the likely feeling in plain language.
2. Reflect the concrete situation the user described so they feel understood.
3. Ask at most one gentle clarifying question only if it would help.
4. Offer 1-3 practical next steps, exercises, or reframes the user can try now.
5. End with a small, doable action or supportive check-in.

Adapt dynamically to the user's emotional state:
- Fear or anxiety: slow the pace, reduce uncertainty, suggest grounding,
  breathing, sensory orientation, planning the next controllable step, or
  separating facts from feared possibilities.
- Anger: validate the boundary or hurt underneath the anger, discourage
  impulsive escalation, suggest pausing, naming the need, and choosing a
  values-aligned response.
- Sadness or grief: be gentle and unhurried. Validate loss, disappointment, or
  heaviness. Offer reflection, self-compassion, reaching out, rest, or one small
  care task without forcing positivity.
- Shame, disgust, or overwhelm: respond without judgment. Normalize the body
  reaction, help the user find safety, boundaries, cleanup/reset steps, or a way
  to talk to themselves more kindly.
- Joy, pride, relief, or happiness: participate warmly. Help the user savor the
  moment, name what it says about their values or effort, share it with someone
  safe, and channel the energy into gratitude, creativity, connection, or a next
  meaningful step.

For pet, family, health, work, or relationship stress, do not pretend to know
outcomes. Acknowledge uncertainty and focus on what the user can do while
waiting: breathing, asking useful questions, preparing, resting, contacting a
support person, or taking the next practical step.

Keep responses conversational, specific, and concise. Avoid repeated phrases,
generic endings, excessive disclaimers, bullet overload, or long lectures. Do
not include math, code, LaTeX, templates, roleplay markup, or hidden reasoning.

Important safety rule: if the user may be in immediate danger or mentions
self-harm, suicide, harming others, overdose, or being unable to stay safe, the
application will bypass you and show crisis resources instead."""


CRISIS_PATTERNS = [
    r"\bkill myself\b",
    r"\bkill me\b",
    r"\bsuicide\b",
    r"\bsuicidal\b",
    r"\bself[-\s]?harm\b",
    r"\bhurt myself\b",
    r"\bcut myself\b",
    r"\bend my life\b",
    r"\bwant to die\b",
    r"\bi'?m going to die\b",
    r"\boverdose\b",
    r"\bcan't stay safe\b",
    r"\bcannot stay safe\b",
    r"\bnot safe with myself\b",
    r"\bharm someone\b",
    r"\bhurt someone\b",
    r"\bkill someone\b",
]

CRISIS_MESSAGE = """I'm really sorry you're carrying this. I can't safely handle this as an ordinary chat.

If you might act on this soon, call emergency services now or go to the nearest emergency room. If you can, move near another person and tell them plainly: "I may not be safe alone right now."

You can also contact a crisis line:
- US and Canada: call or text 988
- UK and Republic of Ireland: Samaritans at 116 123
- India: KIRAN at 1800-599-0019, or emergency services at 112

If you're not in immediate danger, please still reach out to a trusted person or local mental health professional right now. You deserve real support in this moment."""


QUICK_PROMPTS = {
    "Deep Breathing Exercise": (
        "Guide me through a short deep breathing exercise for the emotion I am feeling right now."
    ),
    "Journal Prompt": (
        "Give me one gentle journal prompt to understand what I am feeling without judging it."
    ),
    "Share a Win": (
        "I want to share a small win. Help me savor it and notice why it matters."
    ),
}


def contains_crisis_language(text: str) -> bool:
    """Detect crisis language before any model call is made."""
    normalized = text.lower()
    return any(re.search(pattern, normalized) for pattern in CRISIS_PATTERNS)


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


# ---------------------------------------------------------------------------
# Model initialization and inference helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Gradio event handlers
# ---------------------------------------------------------------------------

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
    messages.append({"role": "assistant", "content": ""})

    if contains_crisis_language(latest_user_message):
        messages[-1]["content"] = CRISIS_MESSAGE
        yield messages
        return

    try:
        for token in stream_model_reply(messages[:-1]):
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


# ---------------------------------------------------------------------------
# UI theme
# ---------------------------------------------------------------------------

CSS = """
:root {
  --solace-deep: #160f2c;
  --solace-room: #26184a;
  --solace-console: #f7f0dc;
  --solace-console-edge: #d8c8a7;
  --solace-panel: rgba(37, 24, 72, 0.82);
  --solace-panel-strong: rgba(24, 17, 48, 0.96);
  --solace-line: rgba(245, 235, 207, 0.18);
  --solace-text: #fffaf0;
  --solace-muted: #d8cde8;
  --joy: #ffd84d;
  --sadness: #5ca9ff;
  --fear: #b892ff;
  --anger: #ff5b56;
  --disgust: #55d487;
  --memory-cyan: #73e2ff;
  --memory-pink: #ff8acf;
}

body,
.gradio-container {
  background:
    linear-gradient(90deg, rgba(255, 216, 77, 0.16) 0 1px, transparent 1px 72px),
    linear-gradient(0deg, rgba(115, 226, 255, 0.10) 0 1px, transparent 1px 88px),
    linear-gradient(140deg, #17102f 0%, #342264 42%, #1d143e 68%, #0d1023 100%) !important;
  color: var(--solace-text);
  min-height: 100vh;
}

.solace-app {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 18px 24px;
}

.solace-header {
  padding: 26px 0 16px;
  border-bottom: 1px solid var(--solace-line);
  position: relative;
}

.solace-header::after {
  background: linear-gradient(
    90deg,
    var(--joy),
    var(--sadness),
    var(--fear),
    var(--anger),
    var(--disgust)
  );
  border-radius: 999px;
  bottom: -2px;
  content: "";
  height: 3px;
  left: 0;
  position: absolute;
  width: min(420px, 100%);
}

.solace-title {
  color: var(--solace-text);
  font-size: 38px;
  font-weight: 800;
  line-height: 1.05;
  letter-spacing: 0;
  margin: 0;
  text-shadow: 0 0 18px rgba(255, 216, 77, 0.26);
}

.solace-subtitle {
  color: #e6dcf5;
  font-size: 15px;
  margin: 8px 0 0;
}

.emotion-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(120px, 1fr));
  gap: 12px;
  margin: 18px 0 16px;
}

.emotion-card {
  align-items: center;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.10), rgba(255, 255, 255, 0.035)),
    var(--solace-panel);
  border: 1px solid color-mix(in srgb, var(--emotion) 58%, rgba(255, 255, 255, 0.2));
  border-radius: 8px;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08), 0 10px 28px rgba(0, 0, 0, 0.16);
  display: flex;
  gap: 10px;
  min-height: 66px;
  padding: 12px;
}

.emotion-core {
  border-radius: 50%;
  background: currentColor;
  box-shadow: 0 0 18px currentColor, inset 0 0 8px rgba(255, 255, 255, 0.52);
  color: var(--emotion);
  flex: 0 0 24px;
  height: 24px;
  width: 24px;
}

.emotion-label {
  color: var(--solace-text);
  font-size: 14px;
  font-weight: 700;
  line-height: 1.15;
}

.emotion-tone {
  color: #d9d0e8;
  font-size: 12px;
  line-height: 1.2;
  margin-top: 3px;
}

#solace-chatbot {
  background:
    linear-gradient(180deg, rgba(247, 240, 220, 0.10), rgba(247, 240, 220, 0.035)),
    rgba(20, 15, 42, 0.86);
  border: 1px solid rgba(247, 240, 220, 0.28);
  border-radius: 8px;
  box-shadow:
    0 20px 70px rgba(0, 0, 0, 0.32),
    inset 0 0 0 1px rgba(255, 255, 255, 0.05),
    inset 0 -18px 50px rgba(255, 216, 77, 0.055);
  min-height: 540px;
}

#solace-chatbot .message,
#solace-chatbot .message-content,
#solace-chatbot .user-message,
#solace-chatbot .bot-message {
  border-radius: 8px !important;
  line-height: 1.55;
}

#solace-chatbot .message.user,
#solace-chatbot .user-message {
  background: linear-gradient(180deg, #4c347e 0%, #35245e 100%) !important;
  border: 1px solid rgba(255, 216, 77, 0.72) !important;
  box-shadow: 0 8px 28px rgba(255, 216, 77, 0.14) !important;
  color: #fffaf0 !important;
}

#solace-chatbot .message.bot,
#solace-chatbot .message.assistant,
#solace-chatbot .bot-message {
  background: linear-gradient(180deg, #173256 0%, #102440 100%) !important;
  border: 1px solid rgba(92, 169, 255, 0.56) !important;
  box-shadow: 0 8px 28px rgba(92, 169, 255, 0.12) !important;
  color: #f4f7ff !important;
}

#solace-chatbot .message.user *,
#solace-chatbot .user-message * {
  color: #fffaf0 !important;
}

#solace-chatbot .message.bot *,
#solace-chatbot .message.assistant *,
#solace-chatbot .bot-message * {
  color: #f4f7ff !important;
}

#solace-chatbot .message-content,
#solace-chatbot .message-content *,
#solace-chatbot .prose,
#solace-chatbot .prose *,
#solace-chatbot .md,
#solace-chatbot .md *,
#solace-chatbot p {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  outline: 0 !important;
}

#solace-chatbot .message p,
#solace-chatbot .user-message p,
#solace-chatbot .bot-message p {
  margin: 0 !important;
}

.quick-row {
  margin-top: 10px;
}

.quick-tool button,
#send-button {
  border-radius: 8px !important;
  font-weight: 760 !important;
}

#send-button {
  background: linear-gradient(180deg, #fff0a3 0%, #ffd84d 100%) !important;
  border: 1px solid rgba(255, 216, 77, 0.82) !important;
  color: #1f1a03 !important;
  box-shadow: 0 0 20px rgba(255, 216, 77, 0.20) !important;
}

.quick-tool button {
  background: rgba(42, 28, 82, 0.94) !important;
  border: 1px solid rgba(184, 146, 255, 0.44) !important;
  color: var(--solace-text) !important;
  min-height: 42px;
}

.quick-tool button:hover {
  background: rgba(58, 38, 108, 0.98) !important;
  border-color: rgba(115, 226, 255, 0.58) !important;
}

#message-box textarea {
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.02)),
    var(--solace-panel-strong) !important;
  color: var(--solace-text) !important;
  border: 1px solid rgba(247, 240, 220, 0.24) !important;
  border-radius: 8px !important;
}

#message-box textarea::placeholder {
  color: #aeb6d4 !important;
}

footer {
  display: none !important;
}

@media (max-width: 820px) {
  .solace-title {
    font-size: 30px;
  }

  .emotion-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  #solace-chatbot {
    min-height: 460px;
  }
}

@media (max-width: 520px) {
  .emotion-grid {
    grid-template-columns: 1fr;
  }
}
"""


def emotion_indicator_html() -> str:
    emotions = [
        ("Joy", "Savor", "var(--joy)"),
        ("Sadness", "Reflect", "var(--sadness)"),
        ("Fear/Anxiety", "Ground", "var(--fear)"),
        ("Anger", "Pause", "var(--anger)"),
        ("Disgust", "Reset", "var(--disgust)"),
    ]

    cards = []
    for label, tone, color in emotions:
        cards.append(
            f"""
            <div class="emotion-card" style="--emotion: {color};">
              <div class="emotion-core"></div>
              <div>
                <div class="emotion-label">{label}</div>
                <div class="emotion-tone">{tone}</div>
              </div>
            </div>
            """
        )
    return f'<div class="emotion-grid">{"".join(cards)}</div>'


def build_demo() -> gr.Blocks:
    blocks_kwargs: Dict[str, Any] = {"title": APP_TITLE}
    if supports_parameter(gr.Blocks, "css"):
        blocks_kwargs["css"] = CSS
    if supports_parameter(gr.Blocks, "theme"):
        blocks_kwargs["theme"] = gr.themes.Base()

    with gr.Blocks(**blocks_kwargs) as demo:
        with gr.Column(elem_classes="solace-app"):
            gr.HTML(
                f"""
                <header class="solace-header">
                  <h1 class="solace-title">{APP_TITLE}</h1>
                  <p class="solace-subtitle">{APP_SUBTITLE}</p>
                </header>
                """
            )
            gr.HTML(emotion_indicator_html())

            chatbot_kwargs: Dict[str, Any] = {
                "label": "SolaceLLM",
                "elem_id": "solace-chatbot",
                "height": 540,
            }
            if supports_parameter(gr.Chatbot, "type"):
                chatbot_kwargs["type"] = "messages"
            if supports_parameter(gr.Chatbot, "show_copy_button"):
                chatbot_kwargs["show_copy_button"] = True
            elif supports_parameter(gr.Chatbot, "buttons"):
                chatbot_kwargs["buttons"] = ["copy", "copy_all"]

            chatbot = gr.Chatbot(**chatbot_kwargs)

            with gr.Row(equal_height=True):
                message_box = gr.Textbox(
                    elem_id="message-box",
                    placeholder="Tell Solace Space what is happening inside right now...",
                    show_label=False,
                    scale=7,
                    lines=2,
                    max_lines=5,
                )
                send_button = gr.Button("Send", elem_id="send-button", scale=1)

            with gr.Row(elem_classes="quick-row"):
                quick_buttons = [
                    gr.Button(label, elem_classes="quick-tool")
                    for label in QUICK_PROMPTS
                ]

            clear_button = gr.ClearButton(
                [message_box, chatbot],
                value="Clear conversation",
                variant="secondary",
            )

            submit_event = message_box.submit(
                add_user_message,
                inputs=[message_box, chatbot],
                outputs=[chatbot, message_box],
                queue=False,
            )
            submit_event.then(generate_assistant_reply, inputs=chatbot, outputs=chatbot)

            click_event = send_button.click(
                add_user_message,
                inputs=[message_box, chatbot],
                outputs=[chatbot, message_box],
                queue=False,
            )
            click_event.then(generate_assistant_reply, inputs=chatbot, outputs=chatbot)

            for button, prompt in zip(quick_buttons, QUICK_PROMPTS.values()):
                quick_event = button.click(
                    partial(add_quick_prompt, prompt),
                    inputs=chatbot,
                    outputs=[chatbot, message_box],
                    queue=False,
                    show_progress="hidden",
                )
                quick_event.then(generate_assistant_reply, inputs=chatbot, outputs=chatbot)

    return demo


demo = build_demo()


if __name__ == "__main__":
    launch_kwargs: Dict[str, Any] = {}
    if supports_parameter(demo.launch, "css"):
        launch_kwargs["css"] = CSS
    if supports_parameter(demo.launch, "theme"):
        launch_kwargs["theme"] = gr.themes.Base()

    demo.queue().launch(**launch_kwargs)
