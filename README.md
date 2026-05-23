# GIS Document Assistant

A RAG (Retrieval-Augmented Generation) application built with Streamlit, LangChain, and Google Gemini. This app allows you to upload GIS-related PDF documents and chat with an AI assistant about their contents. 

## Features
- 📄 Upload and process multiple PDFs simultaneously.
- 🧠 Powered by Google Gemini (Embeddings & LLM).
- 💬 Interactive chat UI via Streamlit.
- 📚 Source citations indicating which document and page the answer was drawn from.
- 💾 Export your chat history to JSON.
- 🌍 Full Arabic language support.

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd task_4
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

Run the Streamlit application using the following command:

```bash
streamlit run app.py
```

1. Open the local URL provided by Streamlit (usually `http://localhost:8501`).
2. Enter your **Google Gemini API Key** in the sidebar.
3. Upload your GIS PDFs.
4. Start chatting!
