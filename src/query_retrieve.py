from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def retrieve(query, embedding_function, top_k=10):
    vector_store = FAISS.load_local(
        "faiss_index",
        embedding_function,
        allow_dangerous_deserialization=True
    )
    results = vector_store.similarity_search(query, k=top_k)
    return query, results

def build_context(docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    return context

def create_prompt(context, question):
    prompt = f"""Responde la siguiente pregunta usando el contexto proporcionado. 
Si no tienes la respuesta exacta en el contexto, simplemente di que no la sabes.

Contexto:
{context}

Pregunta:
{question}

Respuesta:
""".strip()
    return prompt
    
def load_model():
    load_dotenv()
    model = ChatGoogleGenerativeAI(
        model = "gemini-1.5-pro",
        google_api_key = os.getenv('GOOGLE_API_KEY'),
        temperature = 0.7)
    
    return model

def main():
    query, docs = retrieve(input("Ask: "), get_embedding_function())
    context = build_context(docs)
    prompt = create_prompt(context, query)
    model = load_model() 
    answer = model.invoke(prompt)
    print(answer.content)
    
if __name__ == '__main__':
    main()