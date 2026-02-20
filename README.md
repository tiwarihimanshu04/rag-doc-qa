# 📄 RAG Document Q&A System

An end-to-end Retrieval-Augmented Generation (RAG) application that lets users upload PDF documents and ask questions about them using natural language.

##  Demo
Upload any PDF → Process it → Ask questions → Get AI-powered answers with source references.
![IMG_20260220_144522](https://github.com/user-attachments/assets/e8fb1898-fddd-474b-89f4-b059003087ef)


##  Tech Stack
| Component | Technology |
|-----------|-----------|
| Framework | LangChain |
| LLM | Groq API (Llama 3.3 70B) |
| Vector Store | FAISS |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Frontend | Streamlit |

##  How It Works
1. **Ingest** — PDF is loaded and split into chunks
2. **Embed** — Each chunk is converted to a vector using HuggingFace embeddings
3. **Store** — Vectors are stored locally in FAISS
4. **Retrieve** — User question is embedded and top 4 matching chunks are retrieved
5. **Generate** — Groq's Llama3 generates an answer using retrieved chunks as context

## Installation
```bash
git clone https://github.com/tiwarihimanshu04/rag-doc-qa.git
cd rag-doc-qa
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

##  Setup
Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
```
I used free API key from,
 [console.groq.com](https://console.groq.com)

##  Run
```bash
streamlit run app.py
```

##  Project Structure
```
rag-doc-qa/
├── app.py              # Streamlit frontend
├── rag_pipeline.py     # Core RAG logic
├── requirements.txt    # Dependencies
└── .env                # API keys (not committed)
```
