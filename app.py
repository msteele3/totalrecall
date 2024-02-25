import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import chromadb
st.title('Total Recall')


def createCollection():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splitted = text_splitter.split_text(documents)
    client = chromadb.Client()
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    collection = client.create_collection("test", embedding_function=embedding_function)
    collection.add(ids=[str(i) for i in range(len(splitted))], documents=splitted, metadatas=[{"visited": False} for i in range(len(splitted))])
    print("collection created")
    gotted = collection.get(ids=["1"], where={"visited": True})
    print(gotted)
    return collection
    
    
def search(collection: chromadb.Collection, question: str):
    agent = 'bruh'
  

question = st.text_input('Enter your question', '')

documents = st.text_area('Enter the documents', '', height=300)

if st.button('Submit'):
    collection = createCollection()

        

