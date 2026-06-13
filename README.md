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

3. Place your local model at the default path:

```text
models/solace_llm_q4.gguf
```

You can also point to another GGUF file:

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

- `SOLACE_MODEL_PATH`: local GGUF model path. Defaults to `models/solace_llm_q4.gguf`.
- `SOLACE_N_CTX`: context window. Defaults to `2048`.
- `SOLACE_MAX_TOKENS`: maximum generated tokens per reply. Defaults to `512`.
- `SOLACE_TEMPERATURE`: sampling temperature. Defaults to `0.72`.
- `SOLACE_TOP_P`: nucleus sampling value. Defaults to `0.92`.
- `SOLACE_MAX_HISTORY_MESSAGES`: number of recent chat messages sent to the model. Defaults to `16`.

## Safety Note

Solace Space is not a medical, therapy, or crisis service. The app includes a hard-coded crisis bypass for self-harm and immediate danger language, but it should not be treated as a substitute for professional help.
