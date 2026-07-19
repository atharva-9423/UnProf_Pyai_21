<div align="center">

# 🤖 Day 21 — RAG Generator
### Document-based Q&A CLI Tool

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-green?style=for-the-badge&logo=langchain&logoColor=white)](https://python.langchain.com/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-3.5%20Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://aistudio.google.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-red?style=for-the-badge&logo=facebook&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Embeddings-ff9d00?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/)

*A powerful CLI-based Retrieval-Augmented Generation (RAG) application utilizing the modern LCEL framework.*

</div>

---

## 📖 Overview

This application reads `.txt` documents from the `documents/` folder, creates dense vector embeddings, stores them in a FAISS vector database, and answers user queries using the `gemini-1.5-flash` LLM. The answers are strictly grounded on the provided document context using LangChain's modern Expression Language (LCEL).

## ✨ Key Features

- **📄 Document Loading:** Automatically reads all text files from the `documents` folder.
- **✂️ Text Chunking:** Splits large texts into 500-character chunks with a 50-character overlap to retain context using `RecursiveCharacterTextSplitter`.
- **🧠 Embeddings & Vector Store:** Uses `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) and `FAISS` to construct an efficient retrieval index.
- **⚡ Modern RAG Generation:** Connects to Google's Generative AI (`gemini-1.5-flash`) via LangChain LCEL to generate accurate answers based *only* on the retrieved context.
- **💬 Interactive CLI:** A straightforward conversational loop allowing users to repeatedly ask questions.

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python** | Core Language |
| **LangChain (LCEL)** | RAG Pipeline Orchestration |
| **FAISS** | Fast Vector Database Search |
| **Sentence Transformers** | Hugging Face Embeddings generation |
| **Google Gemini API** | Advanced LLM Generation |

## 📁 Project Structure

```text
UnProf_Pyai_21/
├── main.py               # The CLI RAG application utilizing LangChain LCEL
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── documents/
    └── sample_notes.txt  # Sample document for testing RAG
```

## 🚀 Setup & Usage

### 1. Install Dependencies
Make sure you have an active virtual environment, then install the latest dependencies:
```bash
pip install -U -r requirements.txt
```

### 2. Set the Gemini API Key
To allow the application to generate responses, provide your Google Gemini API key as an environment variable. Ensure it is a valid `AIza...` key from Google AI Studio.

**Windows (PowerShell)**
```powershell
$env:GEMINI_API_KEY="AIza..."
```
**Linux / macOS**
```bash
export GEMINI_API_KEY="AIza..."
```

### 3. Run the Application
Execute the RAG CLI directly from your terminal:
```bash
python main.py
```

### 4. Ask Questions
Once the models and embeddings are loaded, you'll be prompted to type in your question based on your documents. Type `exit` to quit the application.

---
<div align="center">
<i>Built for the 100 Days of Code challenge. Phase 3 - LLM & RAG Core.</i>
</div>
