import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from io import BytesIO

# Gemini API setup
API_KEY = "AIzaSyBpUdapLDCBItYx_OGbaRFPqyY6vHz_KVM"
genai.configure(api_key=API_KEY)

gemini_model = genai.GenerativeModel(
    "gemini-1.5-flash",
    generation_config={
        "temperature": 0.7,
        "max_output_tokens": 1024,
    }
)


# Load PDF and create FAISS vectorstore
@st.cache_resource
def load_vectorstore_from_bytes(pdf_bytes):
    loader = PyPDFLoader(BytesIO(pdf_bytes))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    splits = splitter.split_documents(docs)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_index = FAISS.from_documents(splits, embedding_model)
    return faiss_index

# Query function

def answer_query(faiss_index, question, k=4):
    docs = faiss_index.similarity_search(question, k=k)
    context = "\n\n".join([d.page_content for d in docs])
    
    prompt = f"""
You are a helpful assistant.
Use the context below to answer the question in detail.
Provide a structured explanation and examples if relevant.

Context:
{context}

Question: {question}

Answer (detailed):
"""
    response = gemini_model.generate_content(prompt)
    return response.text

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="PDF Q&A Bot", page_icon="📄")
st.title("PDF Q&A Bot with FAISS + Gemini")

# Upload PDF
pdf_file = st.file_uploader("Upload a PDF", type="pdf")

if pdf_file is not None:
    pdf_bytes = pdf_file.read()
    faiss_index = load_vectorstore_from_bytes(pdf_bytes)
    st.success("PDF processed successfully!")

    # User query
    question = st.text_input("Ask a question from the PDF:")
    if question:
        with st.spinner("Thinking..."):
            try:
                answer = answer_query(faiss_index, question)
                st.subheader("Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")
