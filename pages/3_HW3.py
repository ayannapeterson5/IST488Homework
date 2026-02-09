import streamlit as st
import requests
import time
import json
from html.parser import HTMLParser
from typing import List, Dict, Optional

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="HW 3 – Streaming URL Chatbot", layout="centered")
st.title("Homework 03 – A Streaming Chatbot that Discusses a URL")

st.write(
    "This chatbot loads up to **two URLs** as permanent context in a system prompt (never discarded). "
    "Conversation memory: **buffer of last 6 messages** (3 user–assistant exchanges). "
    "Responses stream as the model generates them."
)

# -----------------------------
# CONSTANTS
# -----------------------------
BUFFER_SIZE = 6  # last 6 user/assistant messages; system always kept
OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini"]

# Claude model candidates (we will auto-fallback if your key doesn't support some)
CLAUDE_MODEL_CANDIDATES = [
    "claude-3-haiku-20240307",   # most likely to be available
    "claude-3-sonnet-20240229",
    "claude-3-opus-20240229",
]

# -----------------------------
# HTML -> TEXT (no BeautifulSoup)
# -----------------------------
class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return " ".join(self.fed)


def strip_tags(html: str) -> str:
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def read_url_content(url: str, timeout: int = 12, max_chars: int = 12000) -> str:
    url = (url or "").strip()
    if not url:
        return ""

    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("URL must start with http:// or https://")

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()

    text = strip_tags(r.text)
    text = " ".join(text.split())

    if len(text) > max_chars:
        text = text[:max_chars] + " ...[TRUNCATED]"

    return text


# -----------------------------
# SESSION STATE
# -----------------------------
if "url_context" not in st.session_state:
    st.session_state.url_context = ""

if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []

if "waiting_for_more_info" not in st.session_state:
    st.session_state.waiting_for_more_info = False


# -----------------------------
# SIDEBAR OPTIONS
# -----------------------------
st.sidebar.header("Options")

url1 = st.sidebar.text_input("URL 1 (required for context)", value="https://www.howbaseballworks.com/TheBasics.htm")
url2 = st.sidebar.text_input("URL 2 (optional)", value="https://www.pbs.org/kenburns/baseball/baseball-for-beginners")

vendor = st.sidebar.radio("Choose LLM vendor", ["OpenAI", "Anthropic (Claude)"], index=0)

if vendor == "OpenAI":
    model_choice = st.sidebar.selectbox("OpenAI model", OPENAI_MODELS, index=0)
else:
    model_choice = st.sidebar.selectbox("Claude model", CLAUDE_MODEL_CANDIDATES, index=0)

load_clicked = st.sidebar.button("Load URLs")


# -----------------------------
# LOAD URLS INTO PERMANENT CONTEXT
# -----------------------------
if load_clicked:
    parts = []
    errs = []

    if url1.strip():
        try:
            t1 = read_url_content(url1)
            parts.append(f"URL 1: {url1}\nCONTENT:\n{t1}")
        except Exception as e:
            errs.append(f"URL 1 failed: {e}")

    if url2.strip():
        try:
            t2 = read_url_content(url2)
            parts.append(f"URL 2: {url2}\nCONTENT:\n{t2}")
        except Exception as e:
            errs.append(f"URL 2 failed: {e}")

    st.session_state.url_context = "\n\n---\n\n".join(parts) if parts else ""

    if parts:
        st.sidebar.success("Loaded URL content into system context!")
    else:
        st.sidebar.warning("No URL content loaded.")

    if errs:
        st.sidebar.error("\n".join(errs))


# -----------------------------
# SYSTEM PROMPT (NEVER DISCARDED)
# -----------------------------
def build_system_prompt() -> Dict[str, str]:
    sys_text = (
        "You are a helpful assistant. Be clear, structured, and accurate.\n"
        "Use ONLY the URL context below as your source. If the answer is not in the context, say that.\n"
    )
    if st.session_state.url_context.strip():
        sys_text += "\nURL CONTEXT (permanent):\n" + st.session_state.url_context
    else:
        sys_text += "\n(No URL content loaded yet. Ask the user to click 'Load URLs' in the sidebar.)"
    return {"role": "system", "content": sys_text}


def apply_buffer_memory(chat_msgs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    system_msg = build_system_prompt()
    recent = chat_msgs[-BUFFER_SIZE:]
    return [system_msg] + recent


# -----------------------------
# DISPLAY CHAT HISTORY
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# -----------------------------
# OPENAI STREAMING
# -----------------------------
def stream_openai(messages_for_model: List[Dict[str, str]], model: str):
    from openai import OpenAI

    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in secrets.")

    client = OpenAI(api_key=api_key)

    stream = client.chat.completions.create(
        model=model,
        messages=messages_for_model,
        temperature=0,
        stream=True,
    )

    for event in stream:
        delta = event.choices[0].delta
        if delta and getattr(delta, "content", None):
            yield delta.content


# -----------------------------
# ANTHROPIC (CLAUDE) STREAMING VIA REQUESTS (NO PACKAGE REQUIRED)
# Includes auto-fallback across models if one is not available (404).
# -----------------------------
def _anthropic_stream_once(messages_for_model: List[Dict[str, str]], model: str):
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY in secrets.")

    system_text = messages_for_model[0]["content"]
    chat_msgs = messages_for_model[1:]

    anth_msgs = []
    for m in chat_msgs:
        if m["role"] not in ["user", "assistant"]:
            continue
        anth_msgs.append(
            {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
        )

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 800,
        "temperature": 0,
        "system": system_text,
        "messages": anth_msgs,
        "stream": True,
    }

    with requests.post(url, headers=headers, json=payload, stream=True, timeout=60) as r:
        if r.status_code != 200:
            try:
                err = r.json()
            except Exception:
                err = {"error": r.text}
            # Raise with full info so we can fallback if it's 404
            raise RuntimeError(f"Anthropic API error ({r.status_code}): {err}")

        for raw_line in r.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            if raw_line.startswith("data: "):
                data = raw_line[len("data: "):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                if obj.get("type") == "content_block_delta":
                    delta = obj.get("delta", {})
                    text = delta.get("text")
                    if text:
                        yield text


def stream_anthropic_with_fallback(messages_for_model: List[Dict[str, str]], preferred_model: str):
    """
    Try preferred model first. If it 404s, try other models automatically.
    If ALL fail, raise a clean error.
    """
    tried = []
    models_to_try = [preferred_model] + [m for m in CLAUDE_MODEL_CANDIDATES if m != preferred_model]

    last_err = None
    for m in models_to_try:
        tried.append(m)
        try:
            yield from _anthropic_stream_once(messages_for_model, m)
            return  # success
        except Exception as e:
            last_err = e
            # Only fallback for "model not found" / 404 cases
            msg = str(e).lower()
            if "404" in msg or "not_found" in msg or "model" in msg and "not found" in msg:
                continue
            # If it's not a model availability issue (e.g., 401 invalid key), stop immediately
            raise RuntimeError(f"Claude failed (not a model-name issue): {e}")

    raise RuntimeError(
        "Claude models not available for this API key. "
        f"Tried: {', '.join(tried)}. Last error: {last_err}"
    )


# -----------------------------
# MAIN CHAT INPUT + YES/NO LOOP
# -----------------------------
prompt = st.chat_input("Ask about the URL(s)...")

if prompt:
    user_text = prompt.strip()
    user_lower = user_text.lower()

    if st.session_state.waiting_for_more_info:
        if user_lower in ["yes", "y"]:
            st.session_state.messages.append(
                {"role": "user", "content": "Yes — please provide more detail using the URL context."}
            )
        elif user_lower in ["no", "n"]:
            st.session_state.waiting_for_more_info = False
            msg = "Okay. Ask me another question about the URL(s)."
            with st.chat_message("assistant"):
                st.write(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.stop()
        else:
            msg = "Please type **Yes** or **No**. Do you want more info?"
            with st.chat_message("assistant"):
                st.write(msg)
            st.stop()
    else:
        with st.chat_message("user"):
            st.write(user_text)
        st.session_state.messages.append({"role": "user", "content": user_text})

    messages_for_model = apply_buffer_memory(st.session_state.messages)

    assistant_text = ""
    with st.chat_message("assistant"):
        placeholder = st.empty()

        def render():
            placeholder.markdown(assistant_text)

        try:
            if vendor == "OpenAI":
                for chunk in stream_openai(messages_for_model, model_choice):
                    assistant_text += chunk
                    render()
            else:
                for chunk in stream_anthropic_with_fallback(messages_for_model, model_choice):
                    assistant_text += chunk
                    render()

        except Exception as e:
            assistant_text = f"**Error:** {e}\n\nTry switching to OpenAI if Claude is unavailable for your API key."
            render()

    # Add follow-up question exactly once
    assistant_text = assistant_text.rstrip() + "\n\n**Do you want more info?**"

    # Show final with follow-up
    with st.chat_message("assistant"):
        st.write(assistant_text)

    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
    st.session_state.waiting_for_more_info = True



