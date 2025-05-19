import streamlit as st
from loader import load_and_split
from vector_store import create_vector_store
from chatbot import build_chatbot
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Futuristic AI Chatbot")
st.title("Futuristic AI Chatbot (RAG)")

if "chatbot" not in st.session_state:
    docs = load_and_split("data/futuristic_ai_dataset.csv")
    vectorstore = create_vector_store(docs)
    st.session_state.chatbot = build_chatbot(vectorstore)

query = st.text_input("Ask a question about AI innovations:")

if query:
    response = st.session_state.chatbot.run(query)
    st.write("**Answer:**", response)
