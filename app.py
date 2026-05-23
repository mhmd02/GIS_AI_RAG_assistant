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

# --- Initialize Session State Variables ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "stats" not in st.session_state:
    st.session_state.stats = {
        "total_questions": 0,
        "source_counts": {}
    }

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Gemini API Key", type="password")
    
    st.subheader("🌐 Language / اللغة")
    language = st.radio("Response Language:", ["English", "Arabic (العربية)"])
    
    st.subheader("🎛️ Advanced Controls")
    temperature = st.slider("Creativity (Temperature)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    k_retrieval = st.slider("Document Chunks to Retrieve", min_value=1, max_value=10, value=4, step=1)
    chunk_size = st.slider("Chunk Size (Characters)", min_value=500, max_value=2000, value=1000, step=100)
    
    st.divider()
    uploaded_files = st.file_uploader("📄 Upload GIS PDFs", type="pdf", accept_multiple_files=True)
    
    st.divider()
    
    # Action Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 Reset Memory & DB"):
            st.session_state.messages = []
            st.session_state.stats = {"total_questions": 0, "source_counts": {}}
            if "vectorstore" in st.session_state:
                del st.session_state["vectorstore"]
            st.rerun()
            
    # Export Chat Button
    if len(st.session_state.messages) > 0:
        chat_history_str = json.dumps(st.session_state.messages, ensure_ascii=False, indent=2)
        st.download_button(
            label="💾 Export Chat as JSON",
            data=chat_history_str,
            file_name=f"gis_chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
    st.divider()
    st.header("📊 Usage Statistics")
    st.write(f"**Total Questions Asked:** {st.session_state.stats['total_questions']}")
    if st.session_state.stats["source_counts"]:
        st.write("**Top Sources:**")
        sorted_sources = sorted(st.session_state.stats["source_counts"].items(), key=lambda x: x[1], reverse=True)
        for source, count in sorted_sources[:5]:
            st.write(f"- {source}: {count} times")


if not api_key:
    st.warning("⚠️ Enter your Gemini API key in the sidebar to begin.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# --- Process PDF ---
if uploaded_files and "vectorstore" not in st.session_state:
    with st.spinner("📚 Processing PDFs..."):
        all_chunks = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            try:
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=200
                )
                chunks = splitter.split_documents(docs)
                
                for chunk in chunks:
                    chunk.metadata["source"] = uploaded_file.name
                    
                all_chunks.extend(chunks)
            except Exception as e:
                st.error(f"Error reading {uploaded_file.name}: {e}")
                
        if all_chunks:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            st.session_state.vectorstore = Chroma.from_documents(all_chunks, embeddings)
            st.session_state.chunks_count = len(all_chunks)
            st.success(f"✅ Processed {st.session_state.chunks_count} chunks from {len(uploaded_files)} files!")
        else:
            st.warning("No readable text found in the provided PDFs. Please upload text-based PDFs.")
            st.stop()

# --- Chat Interface ---
if "vectorstore" in st.session_state:
    
    # Example Questions
    example_q = None
    with st.expander("💡 Quick Document Questions", expanded=not st.session_state.messages):
        if st.button("What are the main topics covered in this document?", use_container_width=True): example_q = "What are the main topics covered in this document?"
        if st.button("Summarize the key points of the text.", use_container_width=True): example_q = "Summarize the key points of the text."
        if st.button("ما هو الملخص العام لهذا المستند؟", use_container_width=True): example_q = "ما هو الملخص العام لهذا المستند؟"

    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # New question
    user_input = st.chat_input("Ask a question about your documents...")
    question = example_q or user_input
    
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
            
        st.session_state.stats["total_questions"] += 1
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # Search
                    docs = st.session_state.vectorstore.similarity_search(question, k=k_retrieval)
                    context = "\n\n".join([d.page_content for d in docs])
                    
                    # Update source stats
                    for doc in docs:
                        source_name = doc.metadata.get("source", "Unknown Document")
                        page = doc.metadata.get("page", "?")
                        source_key = f"{source_name} (Page {page})"
                        st.session_state.stats["source_counts"][source_key] = st.session_state.stats["source_counts"].get(source_key, 0) + 1
                    
                    # Format Chat History
                    chat_history_text = ""
                    # Grab last 4 messages to prevent prompt overflow
                    recent_history = st.session_state.messages[-5:-1]
                    if recent_history:
                        chat_history_text = "Previous Conversation:\n"
                        for msg in recent_history:
                            role = "User" if msg["role"] == "user" else "Assistant"
                            chat_history_text += f"{role}: {msg['content']}\n"
                    
                    # Language Directive
                    lang_directive = "You must reply strictly in English." if language == "English" else "You must reply strictly in clear, professional Arabic."

                    # Build prompt
                    prompt = f"""
                    You are a highly knowledgeable GIS (Geographic Information Systems) Assistant.
                    Use the following pieces of retrieved context to answer the user's question.
                    If the context does not contain the answer, say that you don't know based on the provided documents.
                    
                    {lang_directive}
                    
                    {chat_history_text}
                    
                    Context:
                    {context}
                    
                    Current Question: {question}
                    
                    Helpful Answer:"""
                    
                    # Ask LLM
                    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temperature)
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
                    
                    # Force sidebar stats to update immediately
                    st.rerun()
                except Exception as e:
                    st.error(f"Error querying model: {e}")
