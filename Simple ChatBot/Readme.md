# LangChain + Google Gemini PDF Q&A (with Streamlit)

A simple Streamlit app that lets you **chat with your PDFs** using Google’s **Gemini 1.5 Flash** model and LangChain.  

Features:
- Upload a **text-based PDF**
- Ask natural language questions about its content
- Choose between a **Normal Answer** or a **Short Summary**
- Clear the entire chat with a single button

It’s like having an **AI assistant that can read your PDF and explain it in plain English**.  

---

## ⚙️ How It Works

1. Upload a PDF (currently supports text-based PDFs, not scanned images).
2. Text is extracted using **pypdf**.
3. Your **question + extracted text** are passed to Gemini 1.5 Flash via LangChain.
4. The model generates an answer, displayed in the Streamlit app.
5. You can reset everything (input + output) using the **Clear Chat** button.

---
