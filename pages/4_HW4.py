# pages/4_HW4.py

import sys

# ---- Chroma + sqlite fix (works even if pysqlite3 isn't available yet) ----
try:
    import pysqlite3  # from pysqlite3-binary
    sys.modules["sqlite3"] = pysqlite3
except (ModuleNotFoundError, ImportError):
    pass

import streamlit as st
from openai import OpenAI
import time
from pathlib import Path
from bs4 import BeautifulSoup
import chromadb
import shutil

# -----------------------------
# SETTINGS
# -----------------------------
MODEL_NAME = "gpt-4o-mini"              # any LLM you want
EMBED_MODEL = "text-embedding-3-small"  # embeddings model

# Put ALL provided HTML files here:
DATA_FOLDER = "HW/HW4-Data"


# Persistent local Chroma folder:
CHROMA_PATH = "./ChromaDB_for_HW4"
COLLECTION_NAME = "HW4Collection"

st.title("HW 4 – RAG Chatbot (HTML → ChromaDB) + 5-Interaction Memory")

# -----------------------------
# OpenAI client
# -----------------------------
if "client" not in st.session_state:
    st.session_state.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -----------------------------
# Chat memory (store last 5 interactions)
# last 5 interactions = last 10 messages (user+assistant pairs)
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful chatbot. Explain things simply and clearly.\n"
                "If you use retrieved course documents, say so briefly.\n"
                "If the retrieved context is missing or not relevant, say so."
            ),
        }
    ]

if "waiting_for_more_info" not in st.session_state:
    st.session_state.waiting_for_more_info = False


def keep_last_5_interactions(messages: list[dict]) -> list[dict]:
    """
    Keeps:
      - the original system message
      - last 10 non-system messages = last 5 user+assistant pairs
    """
    if not messages:
        return []
    system_msg = messages[0]
    non_system = [m for m in messages[1:] if m.get("role") != "system"]
    recent = non_system[-10:]
    return [system_msg] + recent


# -----------------------------
# HTML TEXT EXTRACTION
# -----------------------------
def extract_text_from_html(html_path: str) -> str:
    """
    Reads HTML and converts it to plain text.
    We remove script/style so the embedding is mostly real content.
    """
    raw = Path(html_path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # clean lines
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


# -----------------------------
# CHUNKING (HW4 requirement)
# Must create 2 mini-documents per HTML file.
#
# Method used: "split in half by paragraphs"
# - We break the text into lines (like paragraphs-ish)
# - Then take first half as chunk1 and second half as chunk2
#
# Why this method:
# - Simple and beginner-friendly
# - Keeps nearby sentences together (better than random character slicing)
# - Guarantees exactly TWO chunks per doc (matches HW requirement)
# -----------------------------
def chunk_into_two(text: str) -> tuple[str, str]:
    parts = [p.strip() for p in text.split("\n") if p.strip()]

    # if too small, split by characters
    if len(parts) < 8:
        mid = max(1, len(text) // 2)
        return text[:mid].strip(), text[mid:].strip()

    mid = len(parts) // 2
    c1 = "\n".join(parts[:mid]).strip()
    c2 = "\n".join(parts[mid:]).strip()
    return c1, c2


# -----------------------------
# Embeddings
# -----------------------------
def embed_text(text: str) -> list[float]:
    resp = st.session_state.client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return resp.data[0].embedding


# -----------------------------
# Create/load ChromaDB (only build if empty)
# -----------------------------
def create_or_load_vectordb():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() == 0:
        html_dir = Path(DATA_FOLDER)
        html_files = sorted(list(html_dir.glob("*.html")) + list(html_dir.glob("*.htm")))

        if not html_files:
            st.error(f"No HTML files found in {DATA_FOLDER}. Put the provided HTML files there.")
            return collection

        with st.spinner("Building HW4 Vector DB (first run only)..."):
            for f in html_files:
                file_name = f.name
                text = extract_text_from_html(str(f))

                chunk1, chunk2 = chunk_into_two(text)

                emb1 = embed_text(chunk1)
                emb2 = embed_text(chunk2)

                # Unique IDs per chunk
                id1 = f"{file_name}__chunk1"
                id2 = f"{file_name}__chunk2"

                collection.add(
                    ids=[id1, id2],
                    documents=[chunk1, chunk2],
                    embeddings=[emb1, emb2],
                    metadatas=[
                        {"source": file_name, "chunk": 1},
                        {"source": file_name, "chunk": 2},
                    ]
                )

        st.success("✅ Vector DB created and HTML documents added (2 chunks each).")

    return collection


def retrieve_context(query: str, k: int = 4) -> tuple[str, list[str]]:
    q_emb = embed_text(query)
    results = st.session_state.HW4_VectorDB.query(
        query_embeddings=[q_emb],
        n_results=k
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    context = "\n\n---\n\n".join(docs) if docs else ""

    sources = []
    for m in metas:
        if not m:
            continue
        sources.append(f"{m.get('source', '?')} (chunk {m.get('chunk', '?')})")

    return context, sources


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("HW4 Controls")

if "HW4_VectorDB" not in st.session_state:
    st.session_state.HW4_VectorDB = create_or_load_vectordb()

if st.sidebar.button("Rebuild Vector DB (re-embeds everything)"):
    if Path(CHROMA_PATH).exists():
        shutil.rmtree(CHROMA_PATH)
    st.session_state.HW4_VectorDB = create_or_load_vectordb()

test_mode = st.sidebar.checkbox("Test Retrieval Mode", value=False)

# -----------------------------
# Show chat history
# -----------------------------
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -----------------------------
# Retrieval Test Mode
# -----------------------------
if test_mode:
    st.subheader("Retrieval Test")
    q = st.text_input("Type a test query")
    if q:
        _, sources = retrieve_context(q, k=4)
        st.write("Top sources returned:")
        for i, s in enumerate(sources, start=1):
            st.write(f"{i}. {s}")
    st.info("Turn off test mode to return to the chatbot.")
    st.stop()

# -----------------------------
# Chat input
# -----------------------------
prompt = st.chat_input("Ask a question about the course content...")

if prompt:
    user_text = prompt.strip()
    user_lower = user_text.lower()

    # handle the "more info?" loop
    if st.session_state.waiting_for_more_info:
        if user_lower in ["yes", "y"]:
            st.session_state.messages.append({"role": "user", "content": "Yes, give me more info."})
        elif user_lower in ["no", "n"]:
            st.session_state.waiting_for_more_info = False
            msg = "Okay! Ask me another question whenever you want."
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

    # --- RAG retrieval ---
    context, sources = retrieve_context(user_text, k=4)

    # --- memory buffer (last 5 interactions) ---
    messages_for_model = keep_last_5_interactions(st.session_state.messages)

    # Inject retrieved context as extra system msg (not permanently saved)
    rag_system = {
        "role": "system",
        "content": (
            "You have retrieved course document context below. Use it if relevant.\n"
            "If it is not relevant or missing, say so.\n\n"
            f"RETRIEVED CONTEXT:\n{context}\n\n"
            f"SOURCES: {', '.join(sources) if sources else 'None'}"
        )
    }

    # Place after the original system message
    if messages_for_model and messages_for_model[0]["role"] == "system":
        messages_for_model = [messages_for_model[0], rag_system] + messages_for_model[1:]
    else:
        messages_for_model = [rag_system] + messages_for_model

    completion = st.session_state.client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages_for_model,
        temperature=0,
    )

    reply = completion.choices[0].message.content

    if sources:
        reply += f"\n\n(Used RAG sources: {', '.join(sources)})"

    reply += "\n\nDo you want more info?"

    def stream_text(text: str):
        for ch in text:
            yield ch
            time.sleep(0.01)

    with st.chat_message("assistant"):
        st.write_stream(stream_text(reply))

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.waiting_for_more_info = True
