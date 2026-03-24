# RAG-CHAT-OLLAMA
Mini Project for PDF/DOCX chat summarizer

Gemini said
This README.md is designed to be a professional portfolio piece that showcases your engineering judgment—specifically the use of Multi-Query Retrieval and local LLM orchestration.

RAG-CHAT-OLLAMA 🚀
A privacy-focused, local Retrieval-Augmented Generation (RAG) system designed to summarize and interact with PDF and DOCX documents. This project leverages Ollama for local inference, ensuring that sensitive data remains on-premise while providing high-performance document intelligence.

📌 Project Overview
This mini-project provides a conversational interface for large documents. By combining vector embeddings with a multi-query retrieval strategy, the system can provide both high-level summaries and granular answers to specific technical questions.

Key Features
Local-First Architecture: Powered by Ollama (Llama 3/Mistral) for data privacy and zero API costs.

Multi-Format Ingestion: Supports .pdf and .docx file formats.

Advanced Retrieval: Utilizes MultiQueryRetriever to generate multiple perspectives of a user's query, improving document hit rates.

Intelligent Chunking: Employs RecursiveCharacterTextSplitter to maintain context and structural integrity of technical text.

Persistent Vector Store: Uses ChromaDB for efficient, indexed document retrieval.

🛠️ Tech Stack
Orchestration: LangChain

LLM Engine: Ollama

Vector Database: ChromaDB

Embeddings: sentence-transformers (Local)

Language: Python 3.10+

🚀 Getting Started
1. Prerequisites
Ensure you have Ollama installed and running. Pull your preferred model (e.g., Llama 3):

Bash
ollama pull llama3
2. Installation
Clone the repository and install dependencies:

Bash
git clone https://github.com/your-username/RAG-CHAT-OLLAMA.git
cd RAG-CHAT-OLLAMA
pip install langchain langchain-community langchain-ollama langchain-text-splitters chromadb pypdf sentence-transformers
3. Usage
Place your documents in the project directory and run the main script:

Python
from langchain_ollama import OllamaLLM
# Initialize the local model
llm = OllamaLLM(model="llama3")

# [The RAG logic follows here...]
🏗️ Architecture
The system follows a standard RAG pipeline enhanced with Multi-Query logic:

Ingestion: Loaders extract text from PDF/DOCX.

Transformation: Text is split using recursive logic to preserve paragraph context.

Embedding: Text chunks are converted into vectors using local sentence-transformers.

Multi-Query: The LLM generates 3-5 variations of the user's prompt to ensure comprehensive retrieval from ChromaDB.

Synthesis: The LLM generates a final response grounded strictly in the retrieved document context.

📝 Engineering Insights
This project focuses on overcoming the "semantic gap" in standard RAG by using a Multi-Query Retriever. By generating variations of a user's input, the system can bridge the gap between user terminology (e.g., "how do I fix it?") and document terminology (e.g., "remediation steps"), which is critical for technical and legal document analysis.

⚖️ License
Distributed under the MIT License. See LICENSE for more information.
