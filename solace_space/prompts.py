"""System prompt, quick prompts, and safety guardrails."""

from __future__ import annotations

import re


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

CRISIS_MESSAGE = (
    "It sounds like you are going through an incredibly difficult time, "
    "but please know that you do not have to carry this alone. I am an AI companion, "
    "not a clinical tool, and I cannot provide the professional crisis support you deserve.\n\n"
    "Please reach out to an official, free, and confidential helpline immediately:\n\n"
    "**INDIA (Tele-MANAS - Govt. Initiative):**\n"
    "• Dial **14416** or **1800-891-4416** (Available 24/7)\n\n"
    "**UNITED STATES (Suicide & Crisis Lifeline):**\n"
    "• Dial or Text **988** (Available 24/7)\n\n"
    "**UNITED KINGDOM (NHS & Samaritans):**\n"
    "• Dial **111** and select the Mental Health Option (NHS 24/7)\n"
    "• Call **116 123** to talk to Samaritans\n\n"
    "Free, human support is available right now. Please connect with them or a trusted professional."
)

QUICK_PROMPTS = {
    "Ground Fear": (
        "I am feeling anxious or afraid. Help me name what I am feeling, "
        "separate facts from feared possibilities, and choose one grounding "
        "step I can do right now."
    ),
    "Soften Sadness": (
        "I am feeling sad or heavy. Help me reflect on what hurts, respond "
        "with self-compassion, and choose one small care action."
    ),
    "Pause Anger": (
        "I am feeling angry. Help me understand what boundary, hurt, or need "
        "is underneath it, pause before reacting, and choose a values-aligned "
        "next step."
    ),
    "Reset Overwhelm": (
        "I feel overwhelmed, ashamed, or disgusted with the situation. Help "
        "me settle my body, reduce judgment, and find one reset or boundary "
        "step."
    ),
    "Channel Joy": (
        "I am feeling happy, proud, or relieved. Help me savor this moment, "
        "name what it says about my values or effort, and channel it into "
        "gratitude, connection, or a meaningful next step."
    ),
}


LOADING_MESSAGE = """
<div class="typing-loader" role="status" aria-live="polite" aria-label="SolaceLLM is thinking">
  <span></span><span></span><span></span>
</div>
"""


def contains_crisis_language(text: str) -> bool:
    """Detect crisis language before any model call is made."""
    normalized = text.lower()
    return any(re.search(pattern, normalized) for pattern in CRISIS_PATTERNS)

