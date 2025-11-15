# AmbedkarGPT – AI Intern Assignment (Kalpit Pvt Ltd)

This project is a simple, command-line Q&A system built as part of the AI Intern Assignment for Kalpit Pvt Ltd.

The system ingests a short speech by Dr. B.R. Ambedkar ("Annihilation of Caste") and allows a user to ask questions based solely on the content of that speech. It uses a RAG (Retrieval Augmented Generation) pipeline orchestrated with LangChain, running entirely on local hardware with no API keys or external services.

<img width="1777" height="930" alt="image" src="https://github.com/user-attachments/assets/78cf26a1-6cb3-4a50-a497-cc11c67d20b0" />

## Overview

This prototype demonstrates the core skills required to build a minimal RAG workflow:

Loading and processing raw text

Chunking text for efficient retrieval

Generating embeddings using a local HuggingFace model

Storing and retrieving embeddings using ChromaDB

Producing answers using a local LLM (Mistral 7B via Ollama)

Running a simple interactive command-line Q&A interface

The system is fully local, requires no API keys, and uses only free and open-source tools.

## Project Structure
    AmbedkarGPT-Intern-Task/
    │
    ├── app.py                # Main RAG pipeline
    ├── speech.txt            # Provided Ambedkar excerpt
    ├── requirements.txt      # Python dependencies
    ├── README.md             # Project documentation
    └── chroma_db/            # Auto-generated vector store

## Requirements (As Specified in the Assignment PDF)

### Backend 
    Flask

### Framework
    LangChain

### Vector Database
     ChromaDB (local)

### Embeddings Model
    HuggingFaceEmbeddings
    Model: sentence-transformers/all-MiniLM-L6-v2

### LLM
    Ollama
    Model: Mistral 7B

### Other Requirements
Must run locally

No API keys

No cloud services


## Setup Instructions
### 1. Clone the Repository
    git clone https://github.com/<your-username>/AmbedkarGPT-Intern-Task.git
    cd AmbedkarGPT-Intern-Task

#### 2. Create and Activate a Virtual Environment
    python -m venv venv

### Activate:
    Windows: venv\Scripts\activate
    Linux/Mac: source venv/bin/activate

### 3. Install Dependencies
    pip install -r requirements.txt

### 4. Install and Configure Ollama
Download Ollama from:     https://ollama.ai

Pull the required model:  ollama pull mistral

### 5. Run the Application
     python app.py

You will see:

    --- System Ready! Ask a question based on the speech. ---
    (Type 'exit' or 'quit' to stop)

You may now enter questions based solely on the provided text.

## How the System Works
1. The program loads speech.txt.
2. The text is split into overlapping chunks for better retrieval.
3. Each chunk is converted into vector embeddings using the MiniLM model.
4. These embeddings are stored locally in ChromaDB.
5. When the user asks a question, relevant chunks are retrieved from Chroma.
6. The retrieved text, along with the question, is passed to Mistral 7B via Ollama.
7. The model generates an answer based strictly on the retrieved content.
 


## System Architecture (Mermaid Diagram)

```mermaid
flowchart TD

A[Start Application] --> B[Load speech.txt]
B --> C[Split Text into Chunks<br>RecursiveCharacterTextSplitter]

C --> D[Generate Embeddings<br>MiniLM-L6-v2]
D --> E[Store Embeddings in ChromaDB]

%% Retrieval Phase
E --> F[User Inputs Question]
F --> G[Retrieve Relevant Chunks<br>Top-k Retrieval]

G --> H[Pass Context + Question to LLM<br>Mistral 7B via Ollama]

H --> I[Generate Answer]

I --> J[Display Answer in Terminal]

J --> F



