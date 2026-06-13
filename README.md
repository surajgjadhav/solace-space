# Solace Space

Solace Space is a local Gradio chat app for an emotional companion experience powered by a GGUF model through `llama-cpp-python`.

## Setup

1. Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. By default, the app loads the GGUF model from Hugging Face:

```text
build-small-hackathon/solace-llm-GGUF
```

You can also point to a local GGUF file:

```bash
export SOLACE_MODEL_PATH=/absolute/path/to/your-model.gguf
```

## Run

```bash
python app.py
```

The app launches with `demo.queue().launch()` and prints a local Gradio URL in the terminal.

## Configuration

Optional environment variables:

- `SOLACE_MODEL_PATH`: optional local GGUF model path. When set, this takes priority over the Hugging Face repo.
- `SOLACE_MODEL_REPO`: Hugging Face GGUF repo. Defaults to `build-small-hackathon/solace-llm-GGUF`.
- `SOLACE_MODEL_FILE`: GGUF filename or glob inside the repo. Defaults to `*Q4_K_M.gguf`.
- `SOLACE_N_CTX`: context window. Defaults to `2048`.
- `SOLACE_MAX_TOKENS`: maximum generated tokens per reply. Defaults to `220`.
- `SOLACE_TEMPERATURE`: sampling temperature. Defaults to `0.72`.
- `SOLACE_TOP_P`: nucleus sampling value. Defaults to `0.92`.
- `SOLACE_REPEAT_PENALTY`: llama-cpp repeat penalty. Defaults to `1.15`.
- `SOLACE_FREQUENCY_PENALTY`: discourages repeated tokens. Defaults to `0.15`.
- `SOLACE_PRESENCE_PENALTY`: lightly encourages new wording. Defaults to `0.05`.
- `SOLACE_MAX_REPEATED_SENTENCES`: app-side repeated sentence cutoff. Defaults to `2`; set to `0` to disable.
- `SOLACE_HISTORY_TOKEN_BUFFER`: token estimate reserved for prompt formatting and safety margin. Defaults to `128`.

Conversation history is trimmed by estimated context budget, not by a fixed
message count. Increase `SOLACE_N_CTX` to keep more prior conversation in the
model prompt.

If llama.cpp prints a line like this:

```text
llama_context: n_ctx_seq (2048) < n_ctx_train (131072) -- the full capacity of the model will not be utilized
```

that is informational, not a crash. It means the app is using a smaller context
window than the model's training maximum. Increase `SOLACE_N_CTX` if you need
more conversation history and have enough RAM/VRAM, for example:

```bash
export SOLACE_N_CTX=8192
```

Using the full `131072` context is usually very memory intensive.

## Safety Note

Solace Space is not a medical, therapy, or crisis service. Its system prompt is designed for counseling-style emotional support, coping strategies, and reflection, but it should not be treated as a substitute for professional help. The app includes a hard-coded crisis bypass for self-harm and immediate danger language.
