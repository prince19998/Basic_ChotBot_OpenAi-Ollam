import os
import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

## LangChain API Key and Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

## Prompt Template
prompt = ChatPromptTemplate(
    [
        ("user", "QUESTION:{question}"),
        ("system", "You are a helpful assistant. Please respond to the question asked")
    ]
)

## Streamlit Framework

# App title and description
st.set_page_config(page_title="LangChain Chatbot", layout="centered")
st.title("LangChain Chatbot with Gemma Model")
st.markdown("#### Powered by Ollama Llama2 'Gemma2:2b' Model")

# Sidebar for app info
with st.sidebar:
    st.header("Chatbot Info")
    st.markdown(
        """
        **About:** This chatbot is powered by the LangChain framework and the Ollama model.
        
        - Model: Gemma2:2b (Llama2)
        - Framework: LangChain + Streamlit
        - API: LangChain API
        
        Ask any question and get an intelligent response!
        """
    )
    st.info("This app is in a development stage, and responses are generated based on the Llama2 model.")

# Input field for user question
input_text = st.text_input("What question do you have in mind?")

# Ollama Llama2 model setup
llm = Ollama(model="gemma2:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Display loading spinner while processing
if input_text:
    with st.spinner("Generating response..."):
        try:
            response = chain.invoke({"question": input_text})
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.markdown("_Please enter a question in the text box above._")

# Custom CSS for better UI
st.markdown(
    """
    <style>
        .reportview-container {
            background-color: #f0f2f6;
            padding: 20px;
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
        }
        .stTextInput {
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)
