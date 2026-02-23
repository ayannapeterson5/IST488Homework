import sys
import streamlit as st

# =========================================================
# SQLite FIX (MUST RUN BEFORE chromadb import)
# =========================================================
try:
    import pysqlite3  # from pysqlite3-binary
    sys.modules["sqlite3"] = pysqlite3
except Exception:
    pass

import os
import re
import time
from pathlib import Path

import streamlit as st
from openai import OpenAI
import chromadb


# =========================================================
# SETTINGS
# =========================================================
MODEL_NAME = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

# Your HW4 docs folder (you said it's called HW4-Data)
DATA_DIR = "HW/HW4-Data"

# Persistent Chroma location + collection name
CHROMA_PATH = "./ChromaDB_for_HW4"
COLLECTION_NAME = "HW4Collection"


# =========================================================
# STREAMLIT UI
# =========================================================
st.title("HW 5 — Short-Term Memory Chatbot + ChromaDB Retrieval (from HW4 Docs)")
st.caption(f"Docs folder: `{DATA_DIR}` | Chroma path: `{CHROMA_PATH}` | Collection: `{COLLECTION_NAME}`")


# =========================================================
# OPENAI CLIENT
# =========================================================
if "client" not in st.session_state:
    st.session_state.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# =========================================================
# SHORT-TERM MEMORY (keep last 5 interactions)
# =========================================================
def keep_last_5_interactions(messages: list[dict]) -> list[dict]:
    system_msg = messages[0]
    non_system = [m for m in messages[1:] if m.get("role") != "system"]
    return [system_msg] + non_system[-10:]  # last 5 user/assistant pairs


# =========================================================
# CHROMADB CONNECT
# =========================================================
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME)

collection = get_collection()


# =========================================================
# EMBEDDING HELPER
# =========================================================
def embed_text(text: str) -> list[float]:
    resp = st.session_state.client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return resp.data[0].embedding


# =========================================================
# TEXT CLEAN + CHUNKING
# =========================================================
def strip_html(html: str) -> str:
    # very simple HTML tag removal; good enough for HW
    text = re.sub(r"<script.*?>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    chunks = []
    i = 0
    n = len(text)
    step = max(1, chunk_size - overlap)
    while i < n:
        chunks.append(text[i:i + chunk_size])
        i += step
    return chunks


# =========================================================
# INGEST HW4 DATA INTO CHROMA
# =========================================================
def ingest_hw4_data(data_dir: str = DATA_DIR, rebuild: bool = False) -> dict:
    """
    Reads HW/HW4-Data files, chunks them, embeds, and adds into ChromaDB.
    If rebuild=True: deletes the chroma folder and recreates collection.
    """
    global collection

    if rebuild:
        # Stop Streamlit cache, delete db folder, recreate
        st.cache_resource.clear()
        if os.path.isdir(CHROMA_PATH):
            for root, dirs, files in os.walk(CHROMA_PATH, topdown=False):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except Exception:
                        pass
                for d in dirs:
                    try:
                        os.rmdir(os.path.join(root, d))
                    except Exception:
                        pass
            try:
                os.rmdir(CHROMA_PATH)
            except Exception:
                pass

        collection = get_collection()

    base = Path(data_dir)
    if not base.exists():
        return {"ok": False, "error": f"Data folder not found: {data_dir}", "added": 0}

    files = [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in [".html", ".txt", ".md"]]
    if not files:
        return {"ok": False, "error": f"No .html/.txt/.md files found in {data_dir}", "added": 0}

    # Progress UI
    added = 0
    skipped_empty = 0
    pbar = st.progress(0, text="Starting ingest...")

    total_steps = len(files)
    for idx, fp in enumerate(files, start=1):
        try:
            raw = fp.read_text(errors="ignore")
        except Exception:
            continue

        if fp.suffix.lower() == ".html":
            raw = strip_html(raw)

        if not raw or len(raw) < 50:
            skipped_empty += 1
            continue

        chunks = chunk_text(raw)
        for j, piece in enumerate(chunks):
            piece = piece.strip()
            if len(piece) < 50:
                continue

            emb = embed_text(piece)

            doc_id = f"{fp.name}__chunk{j}"
            collection.add(
                ids=[doc_id],
                documents=[piece],
                embeddings=[emb],
                metadatas=[{"source": str(fp)}]
            )
            added += 1

        pbar.progress(int(idx / total_steps * 100), text=f"Ingesting: {fp.name} ({idx}/{total_steps})")

    pbar.progress(100, text="Ingest complete.")
    return {"ok": True, "added": added, "skipped_empty": skipped_empty, "collection_count": collection.count()}


# =========================================================
# REQUIRED HW5 FUNCTION (3a)
# =========================================================
def relevant_course_info(query: str, k: int = 3) -> dict:
    q_emb = embed_text(query)
    results = collection.query(query_embeddings=[q_emb], n_results=k)

    docs = results.get("documents", [[]])[0]
    ids = results.get("ids", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    matches = []
    for i, doc in enumerate(docs):
        text = (doc or "").strip()
        if len(text) > 1500:
            text = text[:1500] + "..."

        matches.append({
            "rank": i + 1,
            "id": ids[i] if i < len(ids) else None,
            "metadata": metas[i] if i < len(metas) else {},
            "distance": dists[i] if i < len(dists) else None,
            "text": text
        })

    return {"query": query, "top_k": k, "matches": matches}


# =========================================================
# MESSAGES INIT
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "system",
        "content": (
            "You are a helpful chatbot for questions about the course documents. "
            "Use retrieved context when relevant. "
            "If the retrieved context doesn't contain the answer, say: "
            "'I don't see it in the retrieved documents.'"
        )
    }]


# =========================================================
# INGEST CONTROLS
# =========================================================
colA, colB, colC = st.columns([1, 1, 2])

with colA:
    st.write("**Chroma status**")
    st.write("Count:", collection.count())

with colB:
    if st.button("Build DB (if empty)"):
        if collection.count() > 0:
            st.info("Collection already has data. Use Rebuild if you want to overwrite.")
        else:
            with st.spinner("Building ChromaDB from HW/HW4-Data..."):
                out = ingest_hw4_data(DATA_DIR, rebuild=False)
            if out.get("ok"):
                st.success(f"Added {out['added']} chunks. New count: {out['collection_count']}")
            else:
                st.error(out.get("error", "Unknown ingest error."))

with colC:
    if st.button("Rebuild DB (delete + rebuild)"):
        with st.spinner("Rebuilding ChromaDB from HW/HW4-Data (this overwrites your collection)..."):
            out = ingest_hw4_data(DATA_DIR, rebuild=True)
        if out.get("ok"):
            st.success(f"Rebuilt. Added {out['added']} chunks. New count: {out['collection_count']}")
        else:
            st.error(out.get("error", "Unknown rebuild error."))

st.divider()


# =========================================================
# SHOW CHAT HISTORY
# =========================================================
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# =========================================================
# CHAT INPUT
# =========================================================
prompt = st.chat_input("Ask a question about your HW4 documents...")

if prompt:
    user_text = prompt.strip()

    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write(user_text)

    # 1) Retrieval first (3a)
    retrieval = relevant_course_info(user_text, k=3)
    st.session_state.last_retrieval = retrieval

    # Build context
    context_parts = []
    sources = []
    for m in retrieval["matches"]:
        src = m.get("metadata", {}).get("source") or m.get("id") or "unknown"
        sources.append(str(src))
        context_parts.append(f"[SOURCE: {src}]\n{m.get('text', '')}")
    context_text = "\n\n---\n\n".join(context_parts)

    # 2) Call LLM using retrieved context (3b) — no tools
    messages_for_model = keep_last_5_interactions(st.session_state.messages)

    rag_system = {
        "role": "system",
        "content": (
            "Retrieved context is below. Use it to answer.\n"
            "If it doesn't contain the answer, say: 'I don't see it in the retrieved documents.'\n\n"
            f"RETRIEVED CONTEXT:\n{context_text if context_text else '[NO MATCHES FOUND]'}\n\n"
            f"SOURCES: {', '.join(sources) if sources else 'None'}"
        )
    }

    messages_for_model = [messages_for_model[0], rag_system] + messages_for_model[1:]

    completion = st.session_state.client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages_for_model,
        temperature=0
    )

    reply = completion.choices[0].message.content

    if sources:
        reply += f"\n\n(Used RAG sources: {', '.join(sources)})"

    def stream_text(text: str):
        for ch in text:
            yield ch
            time.sleep(0.005)

    with st.chat_message("assistant"):
        st.write_stream(stream_text(reply))

    st.session_state.messages.append({"role": "assistant", "content": reply})


# =========================================================
# DEBUG
# =========================================================
with st.expander("Show retrieved results (debug)"):
    st.json(st.session_state.get("last_retrieval", {"note": "Ask a question to generate retrieval."}))