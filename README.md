# RAG Basics

**RAG Basics** a beginner-friendly implementation of a Retrieval-Augmented Generation (RAG) pipeline using [LangChain](https://www.langchain.com/), [FAISS](https://github.com/facebookresearch/faiss), [HuggingFace Transformers](https://huggingface.co/), and [Google Gemini](https://ai.google.dev/).

With this project I wanna demonstrate how to build a simple question-answering system based on PDF documents — in this case, the official futsal rules (as I'm a futsal referee) — by retrieving relevant chunks and passing them as context to a language model.

## 🚀 Features

- ✅ Load and parse PDF documents with `PyPDFLoader`
- ✅ Chunk text using `RecursiveCharacterTextSplitter`
- ✅ Generate embeddings with HuggingFace or Google Gemini
- ✅ Store and retrieve vectors with FAISS
- ✅ Generate answers using `ChatGoogleGenerativeAI`
- ✅ Modular Python scripts for indexing and querying

---

## ⚙️ Setup

1. **Install dependencies** (preferably in a virtual environment, note that I recommend you to use uv faster option than other dependency managers):

```bash
pip install uv
```
