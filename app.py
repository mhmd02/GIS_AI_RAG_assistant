import streamlit as st
import os
import tempfile
import json
from datetime import datetime

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


st.set_page_config(page_title="GIS RAG Assistant", page_icon="🗺️", layout="wide")
st.title("🗺️ Chat with Your GIS Documents")

# Sidebar: API key + file upload
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Gemini API Key", type="password")
    uploaded_files = st.file_uploader("📄 Upload GIS PDFs", type="pdf", accept_multiple_files=True)
    
    st.divider()
    
    # Export Chat Button
    if "messages" in st.session_state and len(st.session_state.messages) > 0:
        chat_history_str = json.dumps(st.session_state.messages, ensure_ascii=False, indent=2)
        st.download_button(
            label="💾 Export Chat as JSON",
            data=chat_history_str,
            file_name=f"gis_chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

if not api_key:
    st.warning("⚠️ Enter your Gemini API key in the sidebar to begin.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# Process PDF
if uploaded_files and "vectorstore" not in st.session_state:
    with st.spinner("📚 Processing PDFs..."):
        all_chunks = []
        
        for uploaded_file in uploaded_files:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            # Load and split
            try:
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                chunks = splitter.split_documents(docs)
                
                # Add source metadata explicitly for better referencing
                for chunk in chunks:
                    chunk.metadata["source"] = uploaded_file.name
                    
                all_chunks.extend(chunks)
            except Exception as e:
                st.error(f"Error reading {uploaded_file.name}: {e}")
                
        if all_chunks:
            # Create vector store using the correct supported embedding model
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            
            # Use an ephemeral, in-memory Chroma instance for Streamlit sessions
            st.session_state.vectorstore = Chroma.from_documents(all_chunks, embeddings)
            st.session_state.chunks_count = len(all_chunks)
            st.success(f"✅ Processed {st.session_state.chunks_count} chunks from {len(uploaded_files)} files!")
        else:
            st.warning("No readable text found in the provided PDFs. Please upload text-based PDFs.")
            st.stop()

# Chat interface
if "vectorstore" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # New question
    if question := st.chat_input("Ask a question about your GIS documents (English or Arabic)..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # Search
                    docs = st.session_state.vectorstore.similarity_search(question, k=4)
                    context = "\n\n".join([d.page_content for d in docs])
                    
                    # Build prompt with Arabic support directive
                    prompt = f"""
                    You are a highly knowledgeable GIS (Geographic Information Systems) Assistant.
                    Use the following pieces of retrieved context to answer the user's question.
                    If the context does not contain the answer, say that you don't know based on the provided documents.
                    If the user asks in Arabic, reply in clear, professional Arabic. If they ask in English, reply in English.
                    
                    Context:
                    {context}
                    
                    Question: {question}
                    
                    Helpful Answer:"""
                    
                    # Ask LLM using the supported model
                    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
                    response = llm.invoke(prompt)
                    answer = response.content
                    
                    st.markdown(answer)
                    
                    with st.expander("📚 Sources"):
                        for i, doc in enumerate(docs, 1):
                            source_name = doc.metadata.get("source", "Unknown Document")
                            page = doc.metadata.get("page", "?")
                            st.write(f"**Source {i}** ({source_name} - Page {page}):")
                            st.write(doc.page_content[:300] + "...")
            
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error querying model: {e}")
