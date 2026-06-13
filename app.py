"""Solace Space: a local emotional companion chat app powered by llama-cpp."""

from __future__ import annotations

import warnings
from typing import Any, Dict

warnings.filterwarnings(
    "ignore",
    message=r"'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated.*",
    category=Warning,
    module=r"gradio\.routes",
)


import gradio as gr

from solace_space.config import supports_parameter
from solace_space.theme import CSS
from solace_space.ui import build_demo


demo = build_demo()


if __name__ == "__main__":
    launch_kwargs: Dict[str, Any] = {}
    if supports_parameter(demo.launch, "css"):
        launch_kwargs["css"] = CSS
    if supports_parameter(demo.launch, "theme"):
        launch_kwargs["theme"] = gr.themes.Base()

    demo.queue().launch(**launch_kwargs)
