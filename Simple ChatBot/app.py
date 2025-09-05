from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader  

load_dotenv()

# Streamlit app title
st.title("PDF Q&A with Google Gemini 1.5 Flash")

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if "response" not in st.session_state:
    st.session_state.response = ""

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

style = st.selectbox(
    "Choose response style:",
    ["Normal Answer", "Short Summary"]
)

pdf_text = ""
if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()

input_text = st.text_input("Ask me anything about the PDF:", key="input_text")

def clear_chat():
    st.session_state.input_text = ""
    st.session_state.response = ""

st.button("Clear Chat", on_click=clear_chat)

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Use the given PDF content to answer."),
        ("user", "Context: {context}\n\nQuestion: {question}\nAnswer in style: {style}")
    ]
)

# Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text and pdf_text:
    response = chain.invoke({'question': input_text, 'context': pdf_text, 'style': style})
    st.session_state.response = response
    st.write(response)
elif st.session_state.response:
    st.write(st.session_state.response)
elif input_text and not pdf_text:
    st.warning("⚠️ Please upload a PDF first.")
