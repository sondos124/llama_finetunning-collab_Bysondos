# helpers.py
from sentence_transformers import SentenceTransformer
import faiss
import torch
from llama_cpp import Llama
import json

# Initialize SentenceTransformer for embedding
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the Llama model (adjust the path to where your model is located)
llama_model = Llama(model_path="models/llama-2-7b-chat.gguf")

# Function to load documents
def load_documents(file_path):
    with open(file_path, 'r') as f:
        documents = f.readlines()
    return [doc.strip() for doc in documents]

# Function to embed documents
def embed_documents(documents):
    return embedding_model.encode(documents)

# Function to perform nearest neighbor search using FAISS
def retrieve_documents(query_embedding, document_embeddings, top_k=3):
    index = faiss.IndexFlatL2(document_embeddings.shape[1])  # Using L2 distance
    index.add(document_embeddings)
    _, indices = index.search(query_embedding, top_k)
    return indices[0]

# Function to generate response using TinyLlama
def generate_response(query, retrieved_docs):
    context = " ".join(retrieved_docs)
    prompt = f"Question: {query}\nContext: {context}\nAnswer:"
    response = llama_model(prompt)
    return response['choices'][0]['text'].strip()

# Function to save scored responses
def save_scored_responses(responses, file_path="outputs/scored_responses.json"):
    with open(file_path, 'w') as f:
        json.dump(responses, f, indent=4)

