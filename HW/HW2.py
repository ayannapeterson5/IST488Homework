import streamlit as st
import requests
import re

from openai import OpenAI, RateLimitError, AuthenticationError


# -----------------------------
# PAGE HEADER
# -----------------------------
st.title("HW 2: URL Summarizer")
st.write("Enter a URL and generate a summary using an LLM (OpenAI or Claude).")


# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
llm_provider = st.sidebar.selectbox(
    "Select LLM to use",
    ["OpenAI", "Claude"]
)

use_advanced = st.sidebar.checkbox("Use advanced model")

summary_type = st.sidebar.radio(
    "Summary type",
    ["100 words", "2 connecting paragraphs", "5 bullet points"]
)

language = st.sidebar.selectbox(
    "Select output language",
    ["English", "French", "Spanish"]
)


# -----------------------------
# HELPERS
# -----------------------------
def get_page_text(url: str) -> str:
    """Fetch HTML and convert to plain-ish text (no bs4 needed)."""
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()

    html = r.text
    html = re.sub(r"<script.*?>.*?</script>", "", html, flags=re.S | re.I)
    html = re.sub(r"<style.*?>.*?</style>", "", html, flags=re.S | re.I)

    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()

    # keep prompt reasonable
    return text[:12000]


def build_prompt(text: str, summary_type: str, language: str) -> str:
    return f"""
Summarize the following webpage content.

Requirements:
- Summary format: {summary_type}
- Output language: {language}
- Be clear, accurate, and concise.
- Do not add info not in the content.

Webpage content:
{text}
""".strip()


def summarize_with_openai(prompt: str, use_advanced: bool) -> str:
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY in .streamlit/secrets.toml")
        st.stop()

    client = OpenAI(api_key=api_key)

    # cheaper vs advanced
    model = "gpt-4o-mini" if not use_advanced else "gpt-4o"

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content

    except RateLimitError:
        st.error("OpenAI quota/rate limit hit. Check billing/limits.")
        st.stop()
    except AuthenticationError:
        st.error("OpenAI authentication failed. Check your API key.")
        st.stop()
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        st.stop()


def summarize_with_claude(prompt: str, use_advanced: bool) -> str:
    """
    Claude via direct HTTP request (NO anthropic python package needed).
    """
    api_key = st.secrets.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("Missing ANTHROPIC_API_KEY in .streamlit/secrets.toml")
        st.stop()

    model = "claude-3-haiku-20240307" if not use_advanced else "claude-3-opus-20240229"

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    payload = {
        "model": model,
        "max_tokens": 600,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()

        # Claude returns content as a list of blocks
        return data["content"][0]["text"]

    except requests.exceptions.HTTPError:
        # show Anthropic's error message if present
        try:
            st.error(f"Claude API error: {r.json()}")
        except Exception:
            st.error(f"Claude API error: {r.text}")
        st.stop()
    except Exception as e:
        st.error(f"Claude error: {e}")
        st.stop()


# -----------------------------
# MAIN UI
# -----------------------------
url = st.text_input("Enter a webpage URL")

if st.button("Generate Summary"):
    if not url:
        st.warning("Please enter a URL.")
        st.stop()

    try:
        text = get_page_text(url)
    except Exception:
        st.error("Failed to retrieve webpage content. Try a different URL.")
        st.stop()

    if not text.strip():
        st.error("No readable text found on this page.")
        st.stop()

    prompt = build_prompt(text, summary_type, language)

    if llm_provider == "OpenAI":
        summary = summarize_with_openai(prompt, use_advanced)
    else:
        summary = summarize_with_claude(prompt, use_advanced)

    st.subheader("Summary")
    st.write(summary)








