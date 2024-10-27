from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
import pickle
from dotenv import load_dotenv
import os
# Load environment variables
load_dotenv()
GROQ = os.getenv('GROQ_API_KEY')
def response(query, file_path):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # print(GROQ)

    # Step 6: Initialize the ChatGroq LLM
    llm = ChatGroq(
        temperature=0.9,
        groq_api_key=GROQ,
        model_name='llama-3.1-70b-versatile'
    )

    # Load the FAISS index and document chunks from the pickle file
    with open(file_path, 'rb') as f:
        index, chunks = pickle.load(f)

    # Step 7: Encode the query using the same SentenceTransformer model
    query_embedding = model.encode([query])

    # Step 8: Search the FAISS index for the nearest neighbors
    D, I = index.search(query_embedding, k=5)  # Retrieve top-5 results

    # Retrieve the top matching chunks
    matching_chunks = [chunks[i] for i in I[0]]

    # Convert the matching chunks into Document objects
    documents = [Document(page_content=chunk) for chunk in matching_chunks]

    # Step 9: Initialize FAISS-based retriever using Langchain's FAISS wrapper
    hf_embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    # Use FAISS from_documents method to build the vector store
    faiss_store = FAISS.from_documents(documents, hf_embeddings)

    # Step 10: Create a retrieval chain using the LLM and FAISS retriever
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=faiss_store.as_retriever(),
        return_source_documents=True
    )

    # Query the LLM with the user's input
    result = chain({"query": query})

    return result
    
