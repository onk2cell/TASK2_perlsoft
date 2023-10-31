


import os
from decouple import config

# Load the API key from the environment variable
api_key = config('API_KEY')
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS 
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Set the page title
st.set_page_config(page_title="PDF Text Extraction")

# Create a Streamlit file uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Check if a file is uploaded
if uploaded_file is not None:
    # Read the uploaded PDF file
    doc_reader = PdfReader(uploaded_file)
    
    raw_text = ''
    for i, page in enumerate(doc_reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    st.write(f"Total characters in the PDF: {len(raw_text)}")
    
    # Splitting up the text into smaller chunks for indexing
    text_splitter = CharacterTextSplitter(        
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    docsearch = FAISS.from_texts(texts, embeddings)

    chain = load_qa_chain(OpenAI(openai_api_key=api_key), 
                      chain_type="map_rerank",
                      return_intermediate_steps=True
                      ) 
    # A dictnarey to maintain a memory of priviosely asked questions and responce given by GPT
    results_dict = {} 
    # Add questions and their corresponding answers to the dictionary
    

    while True:
    # Prompt the user for a question
        user_query = st.text_input("Enter a question:", key="unique_key")
        if user_query.lower() == 'exit':
            break  # Exit the loop if the user types 'exit'

    # Perform a similarity search to find relevant documents
        docs = docsearch.similarity_search(user_query, k=5)

    # Chain the documents and question, and return the outputs
        results = chain({"input_documents": docs, "question": user_query}, return_only_outputs=True)
        st.write(results['output_text'])
         # Print the results for the user's query
        results_dict[user_query] = results['output_text']
    print(results_dict)


    # You can now work with the 'texts' variable containing the split text chunks
