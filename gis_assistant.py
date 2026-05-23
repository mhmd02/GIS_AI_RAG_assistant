import os
import json
from typing import List
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

class GISDocumentAssistant:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it before running.")
        
        print("Initializing Embeddings and LLM...")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=self.api_key
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=self.api_key
        )
        
        self.vector_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        self.chat_history = []
        self.stats = {
            "total_questions": 0,
            "documents_loaded": 0,
            "most_referenced_sources": {}
        }

    def load_and_process_pdfs(self, pdf_paths: List[str]):
        """Load multiple PDFs, split them into chunks, and store them."""
        all_docs = []
        for path in pdf_paths:
            print(f"Loading {path}...")
            try:
                loader = PyPDFLoader(path)
                docs = loader.load()
                all_docs.extend(docs)
                self.stats["documents_loaded"] += 1
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        if not all_docs:
            print("No documents loaded.")
            return

        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(all_docs)
        
        if not splits:
            print("Warning: The provided PDFs did not contain any readable text. They might be scanned images or completely empty.")
            return

        print(f"Storing {len(splits)} chunks in Vector DB...")
        self.vector_store.add_documents(splits)
        print("Done processing PDFs.")

    def ask(self, query: str):
        """Ask a question, retrieve context, and generate an answer supporting Arabic."""
        self.stats["total_questions"] += 1
        print(f"\nSearching for context to answer: '{query}'")
        
        # Retrieve relevant documents
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(query)
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Track source statistics
        sources = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            source_key = f"{source} (Page {page})"
            sources.append(source_key)
            self.stats["most_referenced_sources"][source_key] = self.stats["most_referenced_sources"].get(source_key, 0) + 1
        
        unique_sources = list(set(sources))

        # Prompt supporting both English and Arabic
        prompt_template = """
        You are a highly knowledgeable GIS (Geographic Information Systems) Assistant.
        Use the following pieces of retrieved context to answer the user's question.
        If the context does not contain the answer, say that you don't know based on the provided documents.
        If the user asks in Arabic, reply in clear, professional Arabic. If they ask in English, reply in English.
        
        Context:
        {context}
        
        Question: {question}
        
        Helpful Answer:"""
        
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        
        print("Generating answer...")
        response = chain.invoke({"context": context, "question": query})
        answer = response.content
        
        # Save to chat history
        self.chat_history.append({
            "timestamp": datetime.now().isoformat(),
            "question": query,
            "answer": answer,
            "sources": unique_sources
        })
        
        return answer, unique_sources

    def export_chat_history(self, filename: str = "chat_history.json"):
        """Export chat history to a JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.chat_history, f, ensure_ascii=False, indent=4)
        print(f"\nChat history exported to {filename}")

    def display_stats(self):
        """Display usage statistics."""
        print("\n--- 📊 Usage Statistics ---")
        print(f"Total Questions Asked: {self.stats['total_questions']}")
        print(f"Documents Loaded: {self.stats['documents_loaded']}")
        print("Top 3 Most Referenced Sources/Pages:")
        
        sorted_sources = sorted(self.stats["most_referenced_sources"].items(), key=lambda item: item[1], reverse=True)
        for i, (source, count) in enumerate(sorted_sources[:3]):
            print(f"  {i+1}. {source} ({count} times)")
        print("---------------------------\n")

def main():
    print("🌍 Welcome to the GIS Document Assistant!")
    print("Make sure you have set your GOOGLE_API_KEY environment variable.\n")
    
    try:
        assistant = GISDocumentAssistant(persist_directory="./task_4_chroma_db")
    except ValueError as e:
        print(f"Error: {e}")
        return

    # User input loop for PDFs
    pdf_paths = []
    while True:
        path = input("Enter path to a PDF file to process (or press Enter to skip/stop): ").strip()
        if not path:
            break
        if os.path.exists(path):
            pdf_paths.append(path)
        else:
            print("File not found. Please try again.")
            
    if pdf_paths:
        assistant.load_and_process_pdfs(pdf_paths)
    else:
        print("No new PDFs provided. Using existing Vector DB if available.")

    print("\n💬 You can now ask questions about the GIS documents.")
    print("Type 'exit' to quit, 'stats' for usage statistics, or 'export' to save chat history.")
    
    while True:
        query = input("\n🤔 Question: ").strip()
        
        if query.lower() == 'exit':
            break
        elif query.lower() == 'stats':
            assistant.display_stats()
            continue
        elif query.lower() == 'export':
            assistant.export_chat_history()
            continue
        elif not query:
            continue
            
        try:
            answer, sources = assistant.ask(query)
            print("\n🤖 Answer:")
            print(answer)
            print("\n📚 Sources:")
            for source in sources:
                print(f" - {source}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
