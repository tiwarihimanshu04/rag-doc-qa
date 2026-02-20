import streamlit as st
import os
import shutil
from rag_pipeline import (
    load_and_split, build_vectorstore,
    load_vectorstore, get_qa_chain, ask_question
)

CHROMA_PATH = "faiss_db"
DOCS_DIR = "docs"

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄")
st.title("📄 Document Q&A with RAG")
st.markdown("Upload a PDF and ask questions about it.")

with st.sidebar:
    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    
    if uploaded_file:
        pdf_path = os.path.join(DOCS_DIR, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("⚙️ Process Document"):
            with st.spinner("Chunking and embedding..."):
                if os.path.exists(CHROMA_PATH):
                    shutil.rmtree(CHROMA_PATH)
                chunks = load_and_split(pdf_path)
                build_vectorstore(chunks)
                st.success(f"✅ Done! {len(chunks)} chunks stored.")
    
    st.markdown("---")
    st.caption("Stack: LangChain · Groq · ChromaDB · HuggingFace")

if os.path.exists(CHROMA_PATH):
    vectorstore = load_vectorstore()
    qa_chain, retriever = get_qa_chain(vectorstore)
    
    question = st.text_input("💬 Ask a question about your document:")
    
    if question:
        with st.spinner("Thinking..."):
            answer, sources = ask_question(qa_chain, retriever, question)
        
        st.markdown("### 🤖 Answer")
        st.write(answer)
        
        with st.expander("📚 Source Chunks Retrieved"):
            for i, doc in enumerate(sources):
                st.markdown(f"**Chunk {i+1}** (Page {doc.metadata.get('page', '?')})")
                st.caption(doc.page_content)
else:
    st.info("👈 Upload a PDF and click 'Process Document' to get started.")