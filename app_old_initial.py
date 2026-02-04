import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# 1. Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# 2. Initialize the Large Language Model (LLM)
# Using Llama3-8b for high speed and free tier compatibility
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.3-70b-versatile", # Try this more powerful stable model
    groq_api_key=api_key
)

# 3. Setup the Streamlit Web Interface
st.set_page_config(page_title="Locarno Pardo d'Oro Assistant", page_icon="🎬")
st.title("Locarno Golden Leopard Assistant")
st.caption("A demo to test the connection between the UI and the Groq LLM.")

# Create a container for chat input
user_input = st.chat_input("Ask me anything about the festival...")

if user_input:
    # Display user message in the chat container
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        try:
            response = llm.invoke(user_input)
            st.write(response.content)
        except Exception as e:
            st.error(f"Error: {e}")
