"""CSS theme for the Solace Space Gradio interface."""

CSS = """
:root {
  --ink: #29314a;
  --chat-ink: #1f2328;
  --muted: #68708d;
  --surface: rgba(255, 255, 255, 0.82);
  --surface-strong: #ffffff;
  --soft-blue: #eef7ff;
  --soft-lavender: #f4f0ff;
  --line: rgba(91, 107, 143, 0.18);
  --joy: #f6c84f;
  --sadness: #6aa6df;
  --fear: #a78bd9;
  --anger: #e5746c;
  --disgust: #6dbf8b;
  --memory: #62b8d6;
  --dream: #df8fbf;
}

body,
.gradio-container {
  background:
    radial-gradient(circle at 16% 8%, rgba(255, 226, 126, 0.34), transparent 23rem),
    radial-gradient(circle at 88% 14%, rgba(183, 216, 255, 0.44), transparent 28rem),
    radial-gradient(circle at 54% 92%, rgba(213, 201, 255, 0.38), transparent 28rem),
    linear-gradient(135deg, #fff8ea 0%, #eef7ff 44%, #f6f1ff 100%) !important;
  color: var(--ink);
  min-height: 100vh;
}

.gradio-container {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
}

.solace-app {
  max-width: 1280px;
  margin: 0 auto;
  padding: 20px;
}

.solace-header {
  align-items: end;
  display: flex;
  justify-content: space-between;
  gap: 18px;
  padding: 8px 0 18px;
  position: relative;
}

.brand-mark {
  align-items: center;
  display: flex;
  gap: 14px;
}

.memory-logo {
  background:
    radial-gradient(circle at 34% 28%, #fff9cf 0 18%, transparent 19%),
    radial-gradient(circle at 50% 50%, #ffd86a, #f3a95f 72%);
  border: 1px solid rgba(255, 255, 255, 0.86);
  border-radius: 50%;
  box-shadow: 0 14px 34px rgba(235, 179, 68, 0.22), inset 0 -12px 18px rgba(143, 74, 18, 0.10);
  flex: 0 0 54px;
  height: 54px;
  width: 54px;
}

.solace-title {
  color: var(--ink);
  font-size: 34px;
  font-weight: 800;
  line-height: 1.05;
  letter-spacing: 0;
  margin: 0;
  text-shadow: 0 1px 0 rgba(255, 255, 255, 0.92);
}

.solace-subtitle {
  color: var(--muted);
  font-size: 15px;
  margin: 6px 0 0;
}

.system-strip {
  align-items: center;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: flex-end;
}

.system-chip {
  background: rgba(255, 255, 255, 0.70);
  border: 1px solid var(--line);
  border-radius: 999px;
  color: #56607c;
  font-size: 12px;
  font-weight: 700;
  padding: 7px 10px;
  box-shadow: 0 8px 24px rgba(91, 107, 143, 0.08);
}

.console-grid {
  align-items: stretch;
  display: grid;
  gap: 18px;
  grid-template-columns: minmax(260px, 330px) minmax(0, 1fr);
}

.side-console,
.chat-shell {
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.92), rgba(255, 255, 255, 0.76)),
    var(--surface);
  border: 1px solid var(--line);
  border-radius: 18px;
  box-shadow: 0 22px 60px rgba(83, 102, 145, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.82);
}

.side-console {
  padding: 16px;
}

.panel-title {
  align-items: center;
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid rgba(91, 107, 143, 0.12);
  border-radius: 12px;
  color: #1f2328 !important;
  display: flex;
  font-size: 13px;
  font-weight: 800;
  justify-content: space-between;
  letter-spacing: 0;
  margin-bottom: 12px;
  padding: 9px 10px;
}

.side-console .panel-title span {
  color: #1f2328 !important;
}

.pulse-dot {
  animation: solacePulse 1.8s ease-in-out infinite;
  background: var(--memory);
  border-radius: 50%;
  box-shadow: 0 0 16px rgba(98, 184, 214, 0.46);
  height: 9px;
  width: 9px;
}

.emotion-deck {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.emotion-card {
  align-items: center;
  background:
    linear-gradient(90deg, color-mix(in srgb, var(--emotion) 16%, #ffffff), rgba(255, 255, 255, 0.94)),
    #ffffff;
  border: 1px solid color-mix(in srgb, var(--emotion) 32%, rgba(91, 107, 143, 0.14));
  border-radius: 14px;
  box-shadow: inset 4px 0 0 var(--emotion);
  display: flex;
  gap: 12px;
  min-height: 74px;
  padding: 12px;
}

.emotion-avatar {
  align-items: center;
  background:
    radial-gradient(circle at 34% 28%, rgba(255, 255, 255, 0.82), transparent 16%),
    radial-gradient(circle at 50% 62%, color-mix(in srgb, var(--emotion) 76%, #ffffff), var(--emotion));
  border: 1px solid rgba(255, 255, 255, 0.92);
  border-radius: 50%;
  box-shadow: 0 10px 24px color-mix(in srgb, var(--emotion) 28%, transparent);
  color: #30405b;
  display: flex;
  flex: 0 0 48px;
  font-size: 23px;
  font-weight: 900;
  height: 48px;
  justify-content: center;
  width: 48px;
}

.emotion-label {
  color: color-mix(in srgb, var(--emotion) 48%, #24304c);
  font-size: 14px;
  font-weight: 800;
  line-height: 1.15;
}

.emotion-tone {
  color: color-mix(in srgb, var(--emotion) 28%, #5d6681);
  font-size: 12px;
  font-weight: 680;
  line-height: 1.35;
  margin-top: 5px;
}

.memory-rail {
  display: grid;
  gap: 9px;
  grid-template-columns: repeat(5, 1fr);
  margin: 16px 0 4px;
}

.memory-orb {
  aspect-ratio: 1;
  background:
    radial-gradient(circle at 34% 28%, rgba(255, 255, 255, 0.82), transparent 18%),
    radial-gradient(circle, color-mix(in srgb, var(--orb) 48%, #ffffff), color-mix(in srgb, var(--orb) 78%, #ffffff));
  border-radius: 50%;
  box-shadow: 0 10px 20px color-mix(in srgb, var(--orb) 20%, transparent);
}

.chat-shell {
  padding: 14px;
}

.chat-topbar {
  align-items: center;
  display: flex;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}

.chat-title {
  color: var(--ink);
  font-size: 15px;
  font-weight: 800;
}

.chat-meta {
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
}

#solace-chatbot {
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.86), rgba(246, 250, 255, 0.96)),
    var(--soft-blue);
  border: 1px solid rgba(120, 148, 190, 0.22);
  border-radius: 14px;
  box-shadow:
    inset 0 0 0 1px rgba(255, 255, 255, 0.72),
    inset 0 -24px 70px rgba(106, 166, 223, 0.08);
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
  background: linear-gradient(180deg, #fff5c7 0%, #ffe69d 100%) !important;
  border: 1px solid rgba(224, 174, 58, 0.38) !important;
  box-shadow: 0 10px 26px rgba(224, 174, 58, 0.16) !important;
  color: var(--chat-ink) !important;
}

#solace-chatbot .message.bot,
#solace-chatbot .message.assistant,
#solace-chatbot .bot-message {
  background: linear-gradient(180deg, #ffffff 0%, #edf6ff 100%) !important;
  border: 1px solid rgba(106, 166, 223, 0.30) !important;
  box-shadow: 0 10px 26px rgba(94, 134, 187, 0.13) !important;
  color: var(--chat-ink) !important;
}

#solace-chatbot .message.user *,
#solace-chatbot .user-message * {
  color: var(--chat-ink) !important;
}

#solace-chatbot .message.bot *,
#solace-chatbot .message.assistant *,
#solace-chatbot .bot-message * {
  color: var(--chat-ink) !important;
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

#solace-chatbot .typing-loader {
  align-items: center;
  display: inline-flex;
  gap: 7px;
  min-height: 26px;
  padding: 3px 1px;
}

#solace-chatbot .typing-loader span {
  animation: solaceTyping 1.1s ease-in-out infinite;
  background: #6aa6df !important;
  border: 0 !important;
  border-radius: 50% !important;
  box-shadow: 0 0 10px rgba(106, 166, 223, 0.38) !important;
  display: inline-block;
  height: 7px;
  opacity: 0.38;
  width: 7px;
}

#solace-chatbot .typing-loader span:nth-child(2) {
  animation-delay: 0.16s;
}

#solace-chatbot .typing-loader span:nth-child(3) {
  animation-delay: 0.32s;
}

@keyframes solaceTyping {
  0%,
  80%,
  100% {
    opacity: 0.38;
    transform: translateY(0);
  }

  40% {
    opacity: 1;
    transform: translateY(-4px);
  }
}

@keyframes solacePulse {
  0%,
  100% {
    box-shadow: 0 0 0 0 rgba(98, 184, 214, 0.34), 0 0 12px rgba(98, 184, 214, 0.34);
    opacity: 0.76;
    transform: scale(0.92);
  }

  50% {
    box-shadow: 0 0 0 7px rgba(98, 184, 214, 0), 0 0 20px rgba(98, 184, 214, 0.62);
    opacity: 1;
    transform: scale(1);
  }
}

.quick-row {
  margin-top: 12px;
}

.quick-tool button,
#send-button {
  border-radius: 12px !important;
  font-weight: 760 !important;
}

#send-button {
  background: linear-gradient(180deg, #5e9fd8 0%, #4b87c5 100%) !important;
  border: 1px solid rgba(75, 135, 197, 0.54) !important;
  color: #ffffff !important;
  box-shadow: 0 12px 26px rgba(75, 135, 197, 0.22) !important;
}

.quick-tool button {
  background: rgba(255, 255, 255, 0.74) !important;
  border: 1px solid rgba(148, 131, 192, 0.24) !important;
  color: var(--ink) !important;
  min-height: 44px;
  width: 100%;
  box-shadow: 0 8px 20px rgba(91, 107, 143, 0.08) !important;
}

.quick-tool button:hover {
  background: #f5f0ff !important;
  border-color: rgba(167, 139, 217, 0.42) !important;
}

#message-box textarea {
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.94), rgba(248, 251, 255, 0.96)),
    var(--surface-strong) !important;
  color: var(--ink) !important;
  border: 1px solid rgba(120, 148, 190, 0.28) !important;
  border-radius: 14px !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.82) !important;
}

#message-box textarea::placeholder {
  color: #7c849e !important;
}

footer {
  display: none !important;
}

@media (max-width: 820px) {
  .solace-title {
    font-size: 30px;
  }

  .solace-header {
    align-items: start;
    flex-direction: column;
  }

  .system-strip {
    justify-content: flex-start;
  }

  .console-grid {
    grid-template-columns: 1fr;
  }

  #solace-chatbot {
    min-height: 460px;
  }
}

@media (max-width: 520px) {
  .solace-app {
    padding: 12px;
  }

  .memory-logo {
    flex-basis: 46px;
    height: 46px;
    width: 46px;
  }
}
"""
