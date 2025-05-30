#!/usr/bin/python3
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core.llms import ChatMessage
from llama_index.core import VectorStoreIndex

#load environment variable
load_dotenv()

#set environment variable
api_key = os.getenv("GROQ_API_KEY")

llm=Groq(model="llama3-70b-8192", api_key=api_key)

#load documents
print("Loading documents...")

loaded_rag=SimpleDirectoryReader(input_files=["docs/learning_rag.pdf"]).load_data()

#create embeddings-Numerical representation of text
embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

#now lets build a vector search and use our embed model
index = VectorStoreIndex.from_documents(loaded_rag, embed_model=embed_model)

#look for 3 similar searches from the loaded documents
query_engine = index.as_query_engine(similarity_top_k=3, llm=llm)

#pass our query to the retriever
response=query_engine.query("Define Gen AI")

#get response from the llm, the llm receives our chunks plus user query and this provides additional context
print(f"\nAI: {str(response)}")
