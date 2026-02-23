import sys
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "pysqlite3 is missing. Make sure 'pysqlite3-binary' is in requirements.txt."
    )

import json
import streamlit as st
from openai import OpenAI
import chromadb

# -----------------------------
# CONFIG (match your Lab 4)
# -----------------------------
MODEL_NAME = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

CHROMA_PATH = "./ChromaDB_for_Lab4"
COLLECTION_NAME = "Lab4Collection"

st.title("HW 5 — Short-Term Memory Chatbot + ChromaDB Tool")

# -----------------------------
# OpenAI client
# -----------------------------
if "client" not in st.session_state:
    st.session_state.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -----------------------------
# Connect to ChromaDB collection
# -----------------------------
@st.cache_resource
def get_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return chroma_client.get_or_create_collection(name=COLLECTION_NAME)

collection = get_collection()

# -----------------------------
# Embedding helper (same idea as Lab 4)
# -----------------------------
def embed_text(text: str) -> list[float]:
    resp = st.session_state.client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return resp.data[0].embedding

# -----------------------------
# REQUIRED HW5 FUNCTION/TOOL
# relevant_course_info(query) OR relative_club_info(query)
# -----------------------------
def relevant_course_info(query: str, k: int = 3) -> dict:
    """
    Takes input 'query' (from the LLM) and returns relevant info from ChromaDB.
    """
    q_emb = embed_text(query)

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k
    )

    docs = results.get("documents", [[]])[0]
    ids = results.get("ids", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    matches = []
    for i in range(len(docs)):
        matches.append({
            "rank": i + 1,
            "id": ids[i] if i < len(ids) else None,
            "metadata": metas[i] if i < len(metas) else {},
            "distance": dists[i] if i < len(dists) else None,
            "text": docs[i]
        })

    return {
        "query": query,
        "top_k": k,
        "matches": matches
    }

# -----------------------------
# Tool schema for OpenAI tool calling
# -----------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "relevant_course_info",
            "description": "Retrieve relevant course/syllabus/organization info from the ChromaDB vector database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for in the vector DB."},
                    "k": {"type": "integer", "description": "Number of results to return.", "default": 3}
                },
                "required": ["query"]
            }
        }
    }
]

# -----------------------------
# Short-term memory chat
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful chatbot with short-term memory. "
                "When the user asks about course/syllabus/student org info, "
                "use the retrieval tool to look it up. "
                "After retrieval is available, answer using that info."
            )
        }
    ]

# Show chat history (skip system)
for m in st.session_state.messages:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        st.write(m["content"])

user_prompt = st.chat_input("Ask a question about the course / syllabus / orgs...")

if user_prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    # -----------------------------
    # 1) First call WITH tools (tool_choice auto)
    # -----------------------------
    first = st.session_state.client.chat.completions.create(
        model=MODEL_NAME,
        messages=st.session_state.messages,
        tools=tools,
        tool_choice="auto"
    )

    msg1 = first.choices[0].message

    retrieved_payloads = []

    # If tool calls exist, run them
    if msg1.tool_calls:
        # store assistant tool call message
        st.session_state.messages.append({
            "role": "assistant",
            "content": msg1.content,
            "tool_calls": msg1.tool_calls
        })

        for tc in msg1.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            q = args.get("query", user_prompt)
            k = int(args.get("k", 3))

            tool_output = relevant_course_info(query=q, k=k)
            retrieved_payloads.append(tool_output)

            # store tool output message
            st.session_state.messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(tool_output)
            })

        # -----------------------------
        # 2) Second call WITHOUT tools
        # (so the model cannot call the function again)
        # -----------------------------
        second = st.session_state.client.chat.completions.create(
            model=MODEL_NAME,
            messages=st.session_state.messages
            # IMPORTANT: no tools parameter here
        )

        final = second.choices[0].message.content

    else:
        # If model didn’t request retrieval, just use its response
        final = msg1.content

    st.session_state.messages.append({"role": "assistant", "content": final})
    with st.chat_message("assistant"):
        st.write(final)

    # Optional grader-friendly debug
    with st.expander("Show retrieved results (debug)"):
        for p in retrieved_payloads:
            st.json(p)