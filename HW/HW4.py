import sys

try:
    import pysqlite3  
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
# HW4 settings
# -----------------------------
MODEL_NAME = "gpt-4o-mini"   # any LLM you want
EMBED_MODEL = "text-embedding-3-small"

# IMPORTANT: HW4 uses HTML files
DATA_FOLDER = "Labs/HW4-Data"

# persistent chroma folder for HW4
CHROMA_PATH = "./ChromaDB_for_HW4"
COLLECTION_NAME = "HW4Collection"

st.title("HW 4 – RAG Chatbot with HTML Vector DB + Memory Buffer")

# -----------------------------
# OpenAI client
# -----------------------------
if "client" not in st.session_state:
    st.session_state.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -----------------------------
# Messages init
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful chatbot. Explain things simply and clearly.\n"
                "If you use retrieved course documents, say so briefly.\n"
                "If context is missing, say you are not sure."
            ),
        }
    ]

if "waiting_for_more_info" not in st.session_state:
    st.session_state.waiting_for_more_info = False

# -----------------------------
# HW4 requirement: store last 5 interactions
# (an interaction = user+assistant pair)
# We'll build the model messages from the LAST 10 non-system messages.
# -----------------------------
def last_5_interactions(messages: list[dict]) -> list[dict]:
    """
    Keeps:
      - the first system message
      - the last 10 non-system messages (5 user+assistant pairs)
    """
    if not messages:
        return []

    system_msg = messages[0]
    non_system = [m for m in messages[1:] if m["role"] != "system"]

    # last 10 messages = last 5 interactions (user+assistant pairs)
    recent = non_system[-10:]
    return [system_msg] + recent

# -----------------------------
# HTML -> text
# -----------------------------
def extract_text_from_html(html_path: str) -> str:
    html = Path(html_path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    # remove script/style so we embed only readable text
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # clean up whitespace
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)

# -----------------------------
# CHUNKING (HW4 requirement)
# Create TWO mini-documents for each HTML file.
#
# Method used: "paragraph-halves"
# - We split the extracted text by blank lines (paragraph-ish breaks)
# - Then we take first half paragraphs as chunk A, second half as chunk B
#
# Why this method:
# - simple, fast, beginner-friendly
# - keeps nearby sentences together (better than random character cuts)
# - satisfies “2 chunks per document” requirement clearly
# -----------------------------
def chunk_into_two(text: str) -> tuple[str, str]:
    paras = [p.strip() for p in text.split("\n") if p.strip()]

    # If file is tiny, just split by midpoint of characters
    if len(paras) < 8:
        mid = max(1, len(text) // 2)
        return text[:mid].strip(), text[mid:].strip()

    mid = len(paras) // 2
    chunk_a = "\n".join(paras[:mid]).strip()
    chunk_b = "\n".join(paras[mid:]).strip()
    return chunk_a, chunk_b

def embed_text(text: str) -> list[float]:
    resp = st.session_state.client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return resp.data[0].embedding

# -----------------------------
# Create/load vector DB (ONLY create if it doesn't exist)
# HW4 requirement: create the DB only if it does not already exist
# -----------------------------
def create_or_load_vectordb():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    # If collection is empty, build it once
    if collection.count() == 0:
        html_dir = Path(DATA_FOLDER)
        html_files = sorted(list(html_dir.glob("*.html")) + list(html_dir.glob("*.htm")))

        if not html_files:
            st.error(f"No HTML files found in {DATA_FOLDER}. Put the provided .html files there.")
            return collection

        with st.spinner("Building HW4 vector DB from HTML files (first run only)..."):
            for html_file in html_files:
                file_name = html_file.name
                text = extract_text_from_html(str(html_file))

                c1, c2 = chunk_into_two(text)

                # Embed each chunk separately
                e1 = embed_text(c1)
                e2 = embed_text(c2)

                # Unique IDs per chunk
                id1 = f"{file_name}__chunk1"
                id2 = f"{file_name}__chunk2"

                collection.add(
                    ids=[id1, id2],
                    documents=[c1, c2],
                    embeddings=[e1, e2],
                    metadatas=[
                        {"source": file_name, "chunk": 1},
                        {"source": file_name, "chunk": 2},
                    ],
                )

        st.success("HW4 Vector DB built successfully (HTML chunks added)!")
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
        sources.append(f"{m.get('source','?')} (chunk {m.get('chunk','?')})")
    return context, sources

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("HW4 Controls")

if "HW4_VectorDB" not in st.session_state:
    st.session_state.HW4_VectorDB = create_or_load_vectordb()

if st.sidebar.button("Rebuild HW4 Vector DB (re-embeds everything)"):
    if Path(CHROMA_PATH).exists():
        shutil.rmtree(CHROMA_PATH)
    st.session_state.HW4_VectorDB = create_or_load_vectordb()

test_mode = st.sidebar.checkbox("Test Retrieval Mode", value=False)

# -----------------------------
# Display chat history
# -----------------------------
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -----------------------------
# Test mode (optional)
# -----------------------------
if test_mode:
    st.subheader("Vector DB Test")
    q = st.text_input("Test search")
    if q:
        _, sources = retrieve_context(q, k=4)
        st.write("Top returned sources:")
        for i, s in enumerate(sources, start=1):
            st.write(f"{i}. {s}")
    st.info("Uncheck to return to chatbot.")
    st.stop()

# -----------------------------
# Chat input
# -----------------------------
prompt = st.chat_input("Ask a question...")

if prompt:
    user_text = prompt.strip()
    user_lower = user_text.lower()

    if st.session_state.waiting_for_more_info:
        if user_lower in ["yes", "y"]:
            st.session_state.messages.append({"role": "user", "content": "Yes, please give me more info."})
        elif user_lower in ["no", "n"]:
            st.session_state.waiting_for_more_info = False
            msg = "Okay! What can I help you with next?"
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

    # --- memory buffer: last 5 interactions ---
    messages_for_model = last_5_interactions(st.session_state.messages)

    # Inject retrieved context as an extra system message (not saved permanently)
    rag_system = {
        "role": "system",
        "content": (
            "Retrieved course document context is below. Use it if relevant.\n"
            "If it doesn't help, say so.\n\n"
            f"RETRIEVED CONTEXT:\n{context}\n\n"
            f"SOURCES: {', '.join(sources) if sources else 'None'}"
        )
    }

    # Put rag message right after the original system message
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
