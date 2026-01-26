import streamlit as st
from openai import OpenAI
import fitz  # used to allow python to import and read pdf files


def read_pdf(uploaded_file) -> str:
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = []
    for page in doc:
        text.append(page.get_text())
    return "\n".join(text)


# Show title and description.
st.title("MY Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
openai_api_key = st.text_input("OpenAI API Key", type="password")

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
    st.stop()

try:
    client = OpenAI(api_key=openai_api_key)
    client.models.list()
    st.success("API key is valid!")
except Exception:
    st.error("Invalid or blocked API key. Please check it and try again.")
    st.stop()

# Let the user upload a file
uploaded_file = st.file_uploader(
    "Upload a document (.pdf or .txt)", type=("pdf", "txt")
)

# Ask for a question
question = st.text_area(
    "Now ask a question about the document!",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

# Only proceed if both are provided
if uploaded_file and question:
    file_extension = uploaded_file.name.split(".")[-1].lower()

    if file_extension == "txt":
        document = uploaded_file.read().decode("utf-8", errors="ignore")

    elif file_extension == "pdf":
        document = read_pdf(uploaded_file)

    else:
        st.error("Unsupported file type.")
        st.stop()

    messages = [
        {
            "role": "user",
            "content": f"Here's a document:\n\n{document}\n\n---\n\n{question}",
        }
    ]

    model = st.selectbox(
    "Choose a model",
    [
        "gpt-4o-mini",      # replacement for gpt-3.5
        "gpt-4.1",
        "gpt-5-chat-latest",
        "gpt-5-nano",
    ]
)

stream = client.chat.completions.create(
    model=model,
    messages=messages,
    stream=True,
)

st.write_stream(stream)
