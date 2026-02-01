import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF

def read_pdf(uploaded_file) -> str:
    """
    Reads an uploaded PDF file and extracts all text.
    PDFs are binary files, so we use PyMuPDF (fitz) to parse them.
    """
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []
    for page in doc:
        pages.append(page.get_text())

    doc.close()
    return "\n".join(pages)


st.title("MY Document Question Answering App")
st.write("Upload a PDF and ask a question about it.")

uploaded_file = st.file_uploader("Upload a PDF", type=("pdf",))
question = st.text_area("Ask a question about the PDF:", disabled=not uploaded_file)

if uploaded_file and question:
    document_text = read_pdf(uploaded_file)

    # NOTE: this expects your key to be set in the environment / secrets
    client = OpenAI()

    messages = [
        {
            "role": "user",
            "content": f"Here is a document:\n\n{document_text}\n\nQuestion: {question}",
        }
    ]

    stream = client.chat.completions.create(
        model="gpt-5-chat-latest",
        messages=messages,
        stream=True,
    )

    st.write_stream(stream)
