import streamlit as st
import nltk
import os
import time

from text_extraction import textExtraction
from chunks import chunks
from embed import embeddings
from vector_store import faiss_vs
from response import response

nltk.download('punkt')
# Streamlit App Interface
st.title("PDF Research Tool 📄")
st.sidebar.title("Upload PDF Files")
st.sidebar.header("PLEASE CLICK ON PROCESS PDF BUTTON!!")

main_placeholder = st.empty()

pdf_files = st.sidebar.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

process_pdf_clicked = st.sidebar.button("Process PDFs")
file_path = "vector_db.pkl"

if process_pdf_clicked and pdf_files:
    # Extract text from the uploaded PDFs
    main_placeholder.text("Extracting text from PDFs...Started...✅✅✅")
    txt = textExtraction(pdf_files)
    st.write(txt)
    # Split the text into smaller chunks
    main_placeholder.text("Text Splitting...Started...✅✅✅")
    chunks = chunks(txt)

    # Initialize the SentenceTransformer model
    embeddings = embeddings(chunks)
    # Create FAISS index
    faiss_vs(embeddings=embeddings,chunks=chunks,file_path=file_path)
    main_placeholder.text("FAISS index created and saved...✅✅✅")
    time.sleep(1)

# Querying the processed data
query = main_placeholder.text_input("Ask a question about the PDF content:")

if query:
    result = response(query=query, file_path=file_path)    
    # Display the result in Streamlit
    st.write(result['result'])