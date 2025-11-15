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

    # 1. Load the provided text file 
    #
    try:
        loader = TextLoader("./speech.txt", encoding='utf-8') # Added encoding for safety
        documents = loader.load()
        print(f"[+] Successfully loaded speech.txt")
    except Exception as e:
        print(f"[-] Error loading file: {e}")
        return

    # 2. Split the text into manageable chunks 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(documents)
    print(f"[+] Text split into {len(texts)} chunks")

    print("[*] Creating embeddings and vector store (this may take a moment)...")
    
    # Use HuggingFaceEmbeddings with sentence-transformers/all-MiniLM-L6-v2 
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Use ChromaDB as the vector store 
    db = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings,
        persist_directory="./chroma_db" 
    )
    print("[+] Vector store created successfully")

  
    
    # Use Ollama with Mistral 7B 
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

    # Interactive Loop for the command-line Q&A system
        user_query = input("\nYour Question: ")
        
        if user_query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        if not user_query.strip():
            continue

        print("Thinking...")
        try:
            # Generate answer based solely on the content 
            response = qa_chain.invoke({"query": user_query})
            print(f"\nAnswer: {response['result']}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
