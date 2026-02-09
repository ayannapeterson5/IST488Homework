import streamlit as st
from openai import OpenAI
import requests
import time
import re
from html import unescape

# -----------------------------
# MODELS
# -----------------------------
OPENAI_MODEL = "gpt-4o"
ANTHROPIC_MODEL = "claude-3-opus-20240229"

MAX_TOKENS = 2000
MAX_OUTPUT_TOKENS = 800

st.title("HW 3 – Streaming Chatbot w/ URL Context + Conversation Buffer")

# -----------------------------
# SIDEBAR OPTIONS
# -----------------------------
st.sidebar.header("Options")

url_1 = st.sidebar.text_input("URL 1 (optional)")
url_2 = st.sidebar.text_input("URL 2 (optional)")

vendor_choice = st.sidebar.selectbox(
    "Choose LLM (2 vendors)",
    [
        f"OpenAI ({OPENAI_MODEL})",
        f"Anthropic ({ANTHROPIC_MODEL})",
    ],
)

st.write(
    """
**How this chatbot works**
- You can input up to **two URLs** in the sidebar.
- The app reads text from those pages and inserts it into the **system prompt** (permanent context).
- Conversation memory uses a **token-buffer**: the system prompt is always kept, plus as many recent messages as fit.
- The assistant response is displayed with a **streaming/typewriter** effect.
"""
)

# -----------------------------
# SIMPLE HTML → TEXT (no bs4 needed)
# -----------------------------
def html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", html)
    html = re.sub(r"(?is)<br\s*/?>", "\n", html)
    html = re.sub(r"(?is)</p\s*>", "\n", html)
    html = re.sub(r"(?is)<.*?>", " ", html)
    text = unescape(html)
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)

def read_url_content(url: str) -> str:
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        text = html_to_text(r.text)
        # prevent massive prompts
        return text[:12000] if len(text) > 12000 else text
    except Exception as e:
        return f"[Could not load {url}. Error: {e}]"

# -----------------------------
# TOKEN BUFFER HELPERS
# -----------------------------
def rough_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def rough_tokens_messages(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        total += rough_tokens(m.get("role", ""))
        total += rough_tokens(m.get("content", ""))
    return total

def build_token_buffer(all_messages: list[dict], max_tokens: int) -> list[dict]:
    if not all_messages:
        return []

    system_msg = all_messages[0]
    kept = [system_msg]
    used = rough_tokens_messages(kept)

    for msg in reversed(all_messages[1:]):
        msg_tokens = rough_tokens_messages([msg])
        if used + msg_tokens > max_tokens:
            break
        kept.insert(1, msg)
        used += msg_tokens

    return kept

# -----------------------------
# CLIENTS (OpenAI + Anthropic) — safe init
# -----------------------------
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")
ANTHROPIC_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")

if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

if "anthropic_client" not in st.session_state:
    st.session_state.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY) if ANTHROPIC_KEY else None

# -----------------------------
# SYSTEM PROMPT (URLs as permanent context)
# -----------------------------
def build_system_prompt(u1: str, u2: str) -> str:
    docs = []
    if u1.strip():
        docs.append(f"URL 1: {u1}\n\n{read_url_content(u1)}")
    if u2.strip():
        docs.append(f"URL 2: {u2}\n\n{read_url_content(u2)}")

    docs_text = "\n\n---\n\n".join(docs) if docs else "No URL context provided."

    return (
        "You are a helpful chatbot. Explain things so a 10-year-old can understand. "
        "Be clear, simple, and friendly.\n\n"
        "Use the following URL documents as context when answering.\n"
        "If the answer is not in the URLs, say so clearly.\n\n"
        f"{docs_text}"
    )

current_urls = (url_1.strip(), url_2.strip())
if "last_urls" not in st.session_state:
    st.session_state.last_urls = None

if ("messages" not in st.session_state) or (st.session_state.last_urls != current_urls):
    st.session_state.messages = [{"role": "system", "content": build_system_prompt(*current_urls)}]
    st.session_state.last_urls = current_urls
    st.session_state.waiting_for_more_info = False

if "waiting_for_more_info" not in st.session_state:
    st.session_state.waiting_for_more_info = False

# -----------------------------
# DISPLAY CHAT HISTORY
# -----------------------------
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -----------------------------
# STREAMING
# -----------------------------
def stream_text(text: str):
    for ch in text:
        yield ch
        time.sleep(0.01)

# -----------------------------
# CALL MODELS
# -----------------------------
def call_openai(messages_for_model: list[dict]) -> str:
    if st.session_state.openai_client is None:
        return "OpenAI key not found. Add OPENAI_API_KEY to Streamlit secrets."
    completion = st.session_state.openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages_for_model,
        temperature=0,
    )
    return completion.choices[0].message.content

def call_anthropic(messages_for_model: list[dict]) -> str:
    if st.session_state.anthropic_client is None:
        return "Anthropic key not found. Add ANTHROPIC_API_KEY to Streamlit secrets."

    system_prompt = messages_for_model[0]["content"]
    convo_msgs = [
        {"role": m["role"], "content": m["content"]}
        for m in messages_for_model[1:]
        if m["role"] in ["user", "assistant"]
    ]

    resp = st.session_state.anthropic_client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0,
        system=system_prompt,
        messages=convo_msgs,
    )
    return resp.content[0].text

# -----------------------------
# CHAT INPUT + LOGIC
# -----------------------------
prompt = st.chat_input("Say something...")

if prompt:
    user_text = prompt.strip()
    user_lower = user_text.lower()

    if st.session_state.waiting_for_more_info:
        if user_lower in ["yes", "y"]:
            st.session_state.messages.append({"role": "user", "content": "Yes, please give me more info."})
            with st.chat_message("user"):
                st.write("Yes")
        elif user_lower in ["no", "n"]:
            st.session_state.waiting_for_more_info = False
            msg = "Okay! What can I help you with?"
            with st.chat_message("assistant"):
                st.write(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.stop()
        else:
            msg = "Please type Yes or No. Do you want more info?"
            with st.chat_message("assistant"):
                st.write(msg)
            st.stop()
    else:
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.write(user_text)

    messages_for_model = build_token_buffer(st.session_state.messages, MAX_TOKENS)
    # HARD CLAMP (prevents Anthropic crash)
while rough_tokens_messages(messages_for_model) > MAX_TOKENS:
    messages_for_model = messages_for_model[:1] + messages_for_model[-1:]

    tokens_sent = rough_tokens_messages(messages_for_model)
    st.sidebar.write(f"Estimated tokens sent: {tokens_sent} / {MAX_TOKENS}")

    if vendor_choice.startswith("OpenAI"):
        reply = call_openai(messages_for_model)
    else:
        reply = call_anthropic(messages_for_model)

    reply = reply + "\n\nDo you want more info?"

    with st.chat_message("assistant"):
        st.write_stream(stream_text(reply))

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.waiting_for_more_info = True

