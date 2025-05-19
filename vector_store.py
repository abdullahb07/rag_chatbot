from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def create_vector_store(docs):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore
