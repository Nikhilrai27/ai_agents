from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import streamlit as st

# Configure the Streamlit page
st.set_page_config(
    page_title="LangChain Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Initialize the LLM
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

llm = get_llm()

# Creating chatbot chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please provide your answer to user queries."),
    ("human", "Question: {question}"),
])

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Streamlit UI
st.title("🤖 ChatBot with LangChain & Groq")
st.markdown("Welcome! I'm a helpful AI assistant powered by **Llama 3.1**. Ask me anything!")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if user_input := st.chat_input("What is on your mind?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(user_input)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke({"question": user_input})
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
