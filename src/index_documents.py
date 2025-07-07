from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

def split_text_and_get_chunks(docs):

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", "", "\t"],
        chunk_size = 1500,
        chunk_overlap = 300,
        length_function = len,
        is_separator_regex = False,
    )
    
    chunks = []

    for text in docs:
        chunks.extend(text_splitter.split_text(text.page_content))

    return chunks

def get_embedding_function_Gemini():
    load_dotenv()
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key = os.getenv('GOOGLE_API_KEY')
        )

def get_embedding_function_HugginFace():
    return HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

def create_and_save_vector_store(chunks): 
    if not os.path.exists("../database/faiss_index"):
        vector_store = FAISS.from_texts(texts=chunks, embedding=get_embedding_function_Gemini())
        vector_store.save_local("../database/faiss_index")
    else:
        vector_store = FAISS.load_local(
            "../database/faiss_index",
            get_embedding_function_Gemini(), 
            allow_dangerous_deserialization=True
            )
    

def main():
    file_path = '~/rag_basics/docs/reglas_futsal_24-25.pdf'
    docs = load_pdf(file_path)
    chunks = split_text_and_get_chunks(docs)
    create_and_save_vector_store(chunks)
    

if __name__ == "__main__":
    main()