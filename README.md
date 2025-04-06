# 🧠 Advanced RAG System using Streamlit + LangChain + OpenAI Embeddings

This project is a **Retrieval-Augmented Generation (RAG)** system that allows users to upload PDFs, automatically extract information, index it using OpenAI Embeddings and FAISS, and query it using OpenAI’s GPT models — all from a simple **Streamlit interface**.

---

## 📌 Features

- ✅ Upload and process any PDF file
- ✅ Text chunking
- ✅ Dense vector indexing using FAISS
- ✅ Embedding powered by OpenAI Embeddings
- ✅ Conversational QA using `ChatOpenAI`
- ✅ Metadata retention including filename and page numbers
- ✅ Streamlit caching for improved performance
- ✅ Supports OpenAI API v1.0+

---

## 🧱 Tech Stack

- **Frontend**: Streamlit
- **LLM**: OpenAI `gpt-3.5-turbo` or `gpt-4`
- **Embedding Model**: `OpenAIEmbeddings` from `langchain-openai`
- **Vector Store**: FAISS
- **Text Splitter**: RecursiveCharacterTextSplitter (LangChain)
- **PDF Parsing**: PyPDF2
- **RAG Framework**: LangChain

---

## 📂 Folder Structure

```bash
.
├── advancedRAG.py            # Main Streamlit app
├── requirements.txt          # Python dependencies
├── README.md                 # You're reading it!
