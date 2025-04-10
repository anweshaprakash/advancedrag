import os
import streamlit as st
import numpy as np
import faiss
import PyPDF2
import openai

from dotenv import load_dotenv
from FlagEmbedding import FlagReranker
import ragas.metrics as rm
from ragas.llms import LangchainLLMWrapper
from langchain_community.chat_models import ChatOpenAI


from openai import OpenAI
from langchain_openai import OpenAIEmbeddings

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

if "messages" not in st.session_state:
    st.session_state.messages = {}



@st.cache_data(show_spinner=True)
def load_pdf(file_path):
    pages_text = []
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_number, page in enumerate(reader.pages):
            extracted_text = page.extract_text()
            if extracted_text:
                pages_text.append((extracted_text, page_number))
    return pages_text

@st.cache_data(show_spinner=True)
def split_text(text_with_pageinfo, chunk_size=150, filename="document"):
    """Split text from each page into chunks and attach metadata."""
    chunks = []
    for text, page_number in text_with_pageinfo:
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i:i + chunk_size])
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "page_number": page_number + 1,
                    "chunk_index": i // chunk_size,
                    "filename": filename
                }
            })
    return chunks

@st.cache_data(show_spinner=True)
def build_dense_index(chunks, _model):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = _model.embed_documents(texts)
    embeddings = np.array(embeddings)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index, embeddings



def hybrid_search_with_flagembedding(query, chunks, dense_index, model, candidate_multiplier=4, top_k=3):
   
    query_embedding = model.embed_query(query)
    query_embedding = np.array(query_embedding)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    _, candidate_indices = dense_index.search(np.array([query_embedding]), top_k * candidate_multiplier)
    candidate_chunks = [chunks[idx] for idx in candidate_indices[0]]
    candidate_texts = [chunk["text"] for chunk in candidate_chunks]

    reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
    sentence_pairs = [[query, text] for text in candidate_texts]
    scores = reranker.compute_score(sentence_pairs)

    min_score, max_score = min(scores), max(scores)
    normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]
    reranked = sorted(zip(candidate_chunks, normalized_scores), key=lambda x: x[1], reverse=True)[:top_k]
    return reranked

def generate_answer(query, context, model="gpt-3.5-turbo", max_tokens=200):
    context_text = "\n\n".join([chunk["text"] for chunk in context])
    prompt = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

def evaluate_ragas(query, answer, context):
    
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    row = {"user_input": query, "response": answer, "retrieved_contexts": [c["text"] for c in context]}
    results = {}
    results["Faithfulness"] = rm.Faithfulness(llm=evaluator_llm).score(row)
    results["Relevance"] = rm.AnswerRelevancy(llm=evaluator_llm, embeddings=embeddings).score(row)
    return results


st.title("Chat with Your Document: RAG System with FlagEmbedding & RAGAS")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    filename = uploaded_file.name
    pdf_path = "temp.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())
    
    # Load and split PDF with caching
    raw_text = load_pdf(pdf_path)
    chunks = split_text(raw_text, chunk_size=150, filename=filename)
    st.session_state.chunks = chunks

 
    st.session_state.model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    

    st.session_state.dense_index, st.session_state.dense_embeddings = build_dense_index(chunks, st.session_state.model)
    st.success(f"PDF loaded! {len(chunks)} chunks created.")


if "chunks" in st.session_state:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    #Display previous messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Ask a question about the document...")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        with st.spinner("Retrieving relevant context..."):
            results = hybrid_search_with_flagembedding(
                query,
                st.session_state.chunks,
                st.session_state.dense_index,
                st.session_state.dense_embeddings,
                st.session_state.model
            )
            retrieved_context = [chunk for chunk, _ in results]
        
        with st.chat_message("assistant"):
            st.markdown("### Retrieved Context")
            for i, (chunk, score) in enumerate(results):
                meta = chunk["metadata"]
                st.markdown(f"**Passage {i+1} (Score: {score:.3f}, Page: {meta['page_number']}, Chunk: {meta['chunk_index']}, File: {meta['filename']}):**")
                st.write(chunk["text"])
            
            with st.spinner("Generating answer..."):
                answer = generate_answer(query, retrieved_context)
                st.markdown("### Answer")
                st.write(answer)
            
            with st.spinner("Evaluating answer..."):
                evaluation_metrics = evaluate_ragas(query, answer, retrieved_context)
                st.markdown("### Evaluation Metrics")
                st.write(evaluation_metrics)
            
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
