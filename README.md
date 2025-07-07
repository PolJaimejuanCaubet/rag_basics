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


## ⚙️ Setup

1. **Install dependencies** (preferably in a virtual environment, note that I recommend you to use uv faster option than other dependency managers):

    ```bash
    pip install uv
    ```
    ```bash
    uv init
    ```
    ```bash
    uv sync
    ```

2. **Set your API key** in a .env file (note that you have to introduce your api key without commas):

    ```bash
    cp .env.sample .env
    GOOGLE_API_KEY=your_api_key_here
    ```
> [!NOTE]
>  If you don't have your API KEY yet, click on the following link create one, copy it and paste it on your .env
>  [Create a new Google Gemini API KEY](https://aistudio.google.com/apikey)

3. **Run indexing file**

    ```bash
    cd src/
    uv run indexing_documents.py
    ```

4. **Run query + retrieval file**

    ```bash
    uv run query_retrieve.py
    ```




