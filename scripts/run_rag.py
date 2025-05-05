# run_rag.py
import json
from helpers import load_documents, embed_documents, retrieve_documents, generate_response, save_scored_responses

# Load documents (ensure docs.txt is present in the data directory)
documents = load_documents("data/docs.txt")
documents = load_documents("data/stm32f401re.txt")

# Embed the documents
document_embeddings = embed_documents(documents)

# Test the system with sample questions (loaded from prompts/sample_questions.txt)
def run_rag_pipeline():
    # Load the questions
    with open('prompts/sample_questions.txt', 'r') as f:
        questions = f.readlines()

    # Prepare a list for responses
    responses = []

    # Iterate through each question, retrieve relevant documents, and generate a response
    for question in questions:
        question = question.strip()
        
        # Embed the query
        query_embedding = embed_documents([question])
        
        # Retrieve the top 3 most relevant documents
        retrieved_indices = retrieve_documents(query_embedding, document_embeddings, top_k=3)
        retrieved_docs = [documents[i] for i in retrieved_indices]
        
        # Generate a response using the TinyLlama model
        response = generate_response(question, retrieved_docs)
        
        # Save the response with the corresponding question
        responses.append({
            "question": question,
            "response": response,
            "retrieved_docs": retrieved_docs
        })
    
    # Save responses to a JSON file
    save_scored_responses(responses)

if __name__ == "__main__":
    run_rag_pipeline()
