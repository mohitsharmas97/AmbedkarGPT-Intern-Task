import os
import sys

# Core Framework: Using LangChain to orchestrate the RAG pipeline 
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

def main():
    print("--- AmbedkarGPT Initialization ---")

    # 1. Load the provided text file [cite: 8]
    # Loading the specific speech excerpt provided in the data section [cite: 25-37]
    try:
        loader = TextLoader("./speech.txt", encoding='utf-8') # Added encoding for safety
        documents = loader.load()
        print(f"[+] Successfully loaded speech.txt")
    except Exception as e:
        print(f"[-] Error loading file: {e}")
        return

    # 2. Split the text into manageable chunks [cite: 9]
    # Splitting text to ensure it fits within context windows
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(documents)
    print(f"[+] Text split into {len(texts)} chunks")

    # 3. Create Embeddings and store in local vector store [cite: 10]
    print("[*] Creating embeddings and vector store (this may take a moment)...")
    
    # Requirement: Use HuggingFaceEmbeddings with sentence-transformers/all-MiniLM-L6-v2 
    # This ensures no API keys or costs are incurred.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Requirement: Use ChromaDB as the vector store 
    # Persisting data locally as requested.
    db = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings,
        persist_directory="./chroma_db" 
    )
    print("[+] Vector store created successfully")

    # 4 & 5. Retrieve relevant chunks and generate answer [cite: 11, 12]
    
    # Requirement: Use Ollama with Mistral 7B 
    # Operating 100% free with no accounts or API keys.
    llm = Ollama(model="mistral")

    # Setting up the RetrievalQA chain to feed retrieved context to the LLM
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}), # Retrieving top relevant context
        return_source_documents=False
    )

    print("\n--- System Ready! Ask a question based on the speech. ---")
    print("(Type 'exit' or 'quit' to stop)")

    # Interactive Loop for the command-line Q&A system [cite: 4]
    while True:
        user_query = input("\nYour Question: ")
        
        if user_query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        if not user_query.strip():
            continue

        print("Thinking...")
        try:
            # Generate answer based solely on the content [cite: 6, 12]
            response = qa_chain.invoke({"query": user_query})
            print(f"\nAnswer: {response['result']}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()