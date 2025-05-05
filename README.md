# RAG with TinyLlama

## Project Overview
This project integrates **Retrieval-Augmented Generation (RAG)** with **TinyLlama** for answering questions based on a document corpus. It retrieves the most relevant documents from a set of text documents and generates responses using the TinyLlama model.

## Setup

1. Clone this repository.
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Add your text documents to the `data/docs.txt` file.
4. Add sample questions to the `prompts/sample_questions.txt` file.

## Run the Project
To run the RAG pipeline, execute the following:
```bash
python scripts/run_rag.py
