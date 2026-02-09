import streamlit as st
import time
import requests
from html.parser import HTMLParser

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="HW 3 – Streaming URL Chatbot", layout="centered")
st.title("Homework 03 – A Streaming Chatbot that Discusses a URL")

st.write(
    """
**How this chatbot works**
- You can paste up to **two URLs** in the sidebar and click **Load URLs**.
- The app downloads the pages, extracts readable text, and stores that as **permanent context** inside the **system prompt** (the system prompt is never discarded).
- **Conversation memory implemented:** **Buffer of 6 messages** (3 user–assistant exchanges).  
  That means the bot keeps the system prompt + the most recent 6 chat messages (excluding system).
- Responses stream to the screen as the model generates them.
"""
)

# -----------------------------
# CONFIG
# -----------------------------
OPENAI_PREMIUM_MODEL = "gpt-4o"  # premium OpenAI choice (good default)
ANTHROPIC_PREMIUM_MODEL = "claude-3-5-sonnet-latest"  # premium Anthropic alias

# If your HW requires a token-budget buffer instead, you can switch memory mode.
# But per the prompt, we implement ONE option: buffer of 6 messages.
BUFFER_SIZE = 6  # 3 user-assistant exchanges

# -----------------------------
# SIMPLE HTML -> TEXT STRIPPER
# (no BeautifulSoup needed)
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
    """
    Re-use idea from HW2: fetch URL and return readable text.
    This version avoids BeautifulSoup; it strips HTML tags using html.parser.
    """
    if not url:
        return ""

    # Basic safety/cleanup
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("URL must start with http:// or https://")

    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()

    text = strip_tags(r.text)

    # Clean up whitespace
    text = " ".join(text.split())

    # Truncate so you don’t blow up context
    if len(text) > max_chars:
        text = text[:max_chars] + " ...[TRUNCATED]"

    return text


# -----------------------------
# SESSION STATE INIT
# -----------------------------
if "url_context" not in st.session_state:
    st.session_state.url_context = ""  # permanent context inserted into system prompt

if "messages" not in st.session_state:
    st.session_state.messages = []  # we will build system prompt dynamically each run

if "waiting_for_more_info" not in st.session_state:
    st.session_state.waiting_for_more_info = False

# -----------------------------
# SIDEBAR OPTIONS (HW REQUIREMENTS 2 + 3)
# -----------------------------
st.sidebar.header("Options")

url1 = st.sidebar.text_input("URL 1 (required for context)", value="")
url2 = st.sidebar.text_input("URL 2 (optional)", value="")

vendor = st.sidebar.radio("Choose LLM vendor", ["OpenAI", "Anthropic"], index=0)

# “Pick the specific LLM to use” (2 vendors + their premium model)
if vendor == "OpenAI":
    model_choice = st.sidebar.selectbox(
        "OpenAI premium model",
        options=[OPENAI_PREMIUM_MODEL, "gpt-4o-mini"],
        index=0,
    )
else:
    model_choice = st.sidebar.selectbox(
        "Anthropic premium model",
        options=[ANTHROPIC_PREMIUM_MODEL, "claude-3-5-haiku-latest"],
        index=0,
    )

load_clicked = st.sidebar.button("Load URLs")

# -----------------------------
# LOAD URLS INTO PERMANENT CONTEXT (HW REQUIREMENT 4)
# -----------------------------
if load_clicked:
    parts = []
    errors = []

    if url1.strip():
        try:
            text1 = read_url_content(url1)
            parts.append(f"URL 1: {url1}\nCONTENT:\n{text1}")
        except Exception as e:
            errors.append(f"URL 1 error: {e}")

    if url2.strip():
        try:
            text2 = read_url_content(url2)
            parts.append(f"URL 2: {url2}\nCONTENT:\n{text2}")
        except Exception as e:
            errors.append(f"URL 2 error: {e}")

    if parts:
        st.session_state.url_context = "\n\n---\n\n".join(parts)
        st.sidebar.success("Loaded URL content into system context!")
    else:
        st.session_state.url_context = ""
        st.sidebar.warning("No URL content loaded.")

    if errors:
        st.sidebar.error("\n".join(errors))

# -----------------------------
# SYSTEM PROMPT THAT IS NEVER DISCARDED (HW REQUIREMENT 4)
# -----------------------------
def build_system_prompt() -> dict:
    base = (
        "You are a helpful chatbot. Explain things so a 10-year-old can understand: clear, simple, friendly.\n"
        "You MUST use the URL content below as your main context. If the answer is not in the URL content, say so.\n"
        "If the user asks to compare the two URLs, do it directly.\n"
    )
    if st.session_state.url_context.strip():
        base += "\nURL CONTEXT (permanent):\n" + st.session_state.url_context
    else:
        base += "\n(No URL content loaded yet. Ask the user to load URLs in the sidebar.)"
    return {"role": "system", "content": base}


# -----------------------------
# MEMORY: BUFFER OF 6 MESSAGES (HW REQUIREMENT 5 OPTION 1)
# Keep system + last 6 non-system messages
# -----------------------------
def apply_buffer_memory(all_messages: list[dict]) -> list[dict]:
    """
    all_messages already excludes system.
    Return system + last BUFFER_SIZE messages.
    """
    system = build_system_prompt()
    recent = all_messages[-BUFFER_SIZE:]
    return [system] + recent


# -----------------------------
# CLIENT SETUP
# -----------------------------
def get_openai_client():
    from openai import OpenAI

    if "openai_client" not in st.session_state:
        st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    return st.session_state.openai_client


def get_anthropic_client():
    # Anthropic needs `pip install anthropic`
    # and st.secrets["ANTHROPIC_API_KEY"]
    import anthropic

    if "anthropic_client" not in st.session_state:
        st.session_state.anthropic_client = anthropic.Anthropic(
            api_key=st.secrets["ANTHROPIC_API_KEY"]
        )
    return st.session_state.anthropic_client


# -----------------------------
# DISPLAY CHAT HISTORY
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask about the URL(s)...")

# -----------------------------
# MAIN CHAT LOGIC
# -----------------------------
if prompt:
    user_text = prompt.strip()
    user_lower = user_text.lower()

    # Handle yes/no loop (kept from your Lab 3)
    if st.session_state.waiting_for_more_info:
        if user_lower in ["yes", "y"]:
            st.session_state.messages.append(
                {"role": "user", "content": "Yes, please give me more info."}
            )
        elif user_lower in ["no", "n"]:
            st.session_state.waiting_for_more_info = False
            msg = "Okay! Ask me anything else about the URL(s)."
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

    # Build messages for model using buffer memory
    messages_for_model = apply_buffer_memory(st.session_state.messages)

    # -----------------------------
    # STREAMING RESPONSE
    # -----------------------------
    assistant_full = ""

    with st.chat_message("assistant"):
        placeholder = st.empty()

        def update_stream(text_so_far: str):
            placeholder.markdown(text_so_far)

        # OpenAI streaming
        if vendor == "OpenAI":
            client = get_openai_client()
            stream = client.chat.completions.create(
                model=model_choice,
                messages=messages_for_model,
                temperature=0,
                stream=True,
            )

            for event in stream:
                delta = event.choices[0].delta
                if delta and getattr(delta, "content", None):
                    assistant_full += delta.content
                    update_stream(assistant_full)

        # Anthropic streaming (if installed + key exists)
        else:
            try:
                client = get_anthropic_client()

                # Anthropic wants system separately; we already built a system msg, so split it out.
                sys_msg = messages_for_model[0]["content"]
                non_system = messages_for_model[1:]

                # Convert to Anthropic format: list of {"role": "user"/"assistant", "content": "..."}
                # (Anthropic also supports richer content blocks, but we’ll keep it simple.)
                with client.messages.stream(
                    model=model_choice,
                    system=sys_msg,
                    max_tokens=800,
                    temperature=0,
                    messages=non_system,
                ) as stream:
                    for text in stream.text_stream:
                        assistant_full += text
                        update_stream(assistant_full)

            except Exception as e:
                assistant_full = (
                    "Anthropic setup error. Make sure you installed `anthropic` and set "
                    "`ANTHROPIC_API_KEY` in .streamlit/secrets.toml.\n\n"
                    f"Error details: {e}"
                )
                update_stream(assistant_full)

        # Always ask your Lab 3 follow-up question
        assistant_full += "\n\nDo you want more info?"
        update_stream(assistant_full)

    # Save assistant response and set waiting flag
    st.session_state.messages.append({"role": "assistant", "content": assistant_full})
    st.session_state.waiting_for_more_info = True


