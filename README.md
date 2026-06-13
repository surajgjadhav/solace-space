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
- `SOLACE_MAX_TOKENS`: maximum generated tokens per reply. Defaults to `512`.
- `SOLACE_TEMPERATURE`: sampling temperature. Defaults to `0.72`.
- `SOLACE_TOP_P`: nucleus sampling value. Defaults to `0.92`.
- `SOLACE_MAX_HISTORY_MESSAGES`: number of recent chat messages sent to the model. Defaults to `16`.

## Safety Note

Solace Space is not a medical, therapy, or crisis service. The app includes a hard-coded crisis bypass for self-harm and immediate danger language, but it should not be treated as a substitute for professional help.
