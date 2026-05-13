# 📄 Doc QA System using Gemini + LangChain + FAISS

An AI-powered Document Question Answering (Doc QA) system built using **LangChain**, **Google Gemini**, **FAISS**, and **Streamlit**.

This project allows users to upload PDF documents and ask questions in natural language. The system uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant document chunks and generate context-aware answers using Gemini.

---

# 🚀 Features

- 📂 Upload multiple PDF documents
- ✂️ Automatic text chunking
- 🧠 Semantic search using embeddings
- 🔍 Fast similarity retrieval with FAISS
- 🤖 AI-generated answers using Gemini Pro
- 💬 Interactive Streamlit interface
- 🏗️ Modular and scalable project structure

---

# 🧠 Tech Stack

- Python
- Streamlit
- LangChain
- Google Gemini API
- FAISS Vector Database
- PyPDF2

---

# 📌 Project Architecture

```text
PDFs
  ↓
Text Extraction
  ↓
Text Chunking
  ↓
Embeddings Generation
  ↓
FAISS Vector Store
  ↓
User Question
  ↓
Similarity Search
  ↓
Gemini LLM
  ↓
Final Answer

```
---


# 📁 Project Structure
doc_qa_project/
│
├── app.py
├── requirements.txt
├── config.py
│
├── src/
│   ├── loader.py
│   ├── splitter.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   └── rag_chain.py
│
├── data/
│   └── faiss_index/
│
└── .env