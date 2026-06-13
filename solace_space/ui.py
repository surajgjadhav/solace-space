"""Gradio UI assembly for Solace Space."""

from __future__ import annotations

from functools import partial
from typing import Any, Dict

import gradio as gr

from .chat import add_quick_prompt, add_user_message, generate_assistant_reply
from .config import APP_SUBTITLE, APP_TITLE, supports_parameter
from .prompts import QUICK_PROMPTS
from .theme import CSS


def emotion_indicator_html() -> str:
    emotions = [
        ("Joy", "Savor and share", "J", "var(--joy)"),
        ("Sadness", "Name and soften", "S", "var(--sadness)"),
        ("Fear", "Ground and plan", "F", "var(--fear)"),
        ("Anger", "Pause and protect", "A", "var(--anger)"),
        ("Disgust", "Reset boundaries", "D", "var(--disgust)"),
    ]

    cards = []
    for label, tone, initial, color in emotions:
        cards.append(
            f"""
            <div class="emotion-card" style="--emotion: {color};">
              <div class="emotion-avatar">{initial}</div>
              <div>
                <div class="emotion-label">{label}</div>
                <div class="emotion-tone">{tone}</div>
              </div>
            </div>
            """
        )
    orbs = "".join(
        f'<div class="memory-orb" style="--orb: {color};"></div>'
        for _, _, _, color in emotions
    )
    return f"""
    <section class="side-console">
      <div class="panel-title">
        <span>Emotion Console</span>
        <span class="pulse-dot"></span>
      </div>
      <div class="emotion-deck">{"".join(cards)}</div>
      <div class="memory-rail">{orbs}</div>
    </section>
    """


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
                  <div class="brand-mark">
                    <div class="memory-logo"></div>
                    <div>
                      <h1 class="solace-title">{APP_TITLE}</h1>
                      <p class="solace-subtitle">{APP_SUBTITLE}</p>
                    </div>
                  </div>
                  <div class="system-strip">
                    <span class="system-chip">Local LLM</span>
                    <span class="system-chip">Emotion-aware</span>
                    <span class="system-chip">Private session</span>
                  </div>
                </header>
                """
            )
            with gr.Row(elem_classes="console-grid", equal_height=True):
                with gr.Column(scale=3, min_width=260):
                    gr.HTML(emotion_indicator_html())

                    with gr.Column(elem_classes="quick-row"):
                        quick_buttons = [
                            gr.Button(label, elem_classes="quick-tool")
                            for label in QUICK_PROMPTS
                        ]

                with gr.Column(scale=8, elem_classes="chat-shell"):
                    gr.HTML(
                        """
                        <div class="chat-topbar">
                          <div>
                            <div class="chat-title">SolaceLLM Session</div>
                            <div class="chat-meta">Guided emotional support</div>
                          </div>
                          <div class="system-chip">Streaming</div>
                        </div>
                        """
                    )

                    chatbot_kwargs: Dict[str, Any] = {
                        "label": "SolaceLLM",
                        "elem_id": "solace-chatbot",
                        "height": 560,
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
                            scale=8,
                            lines=2,
                            max_lines=5,
                        )
                        send_button = gr.Button("Send", elem_id="send-button", scale=1)

                    gr.ClearButton(
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

