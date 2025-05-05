import sys
import os
import json

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

from sentence_transformers import SentenceTransformer, util

# Add root path to sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llama_config import MODEL_ID


def load_questions(path):
    with open(path, "r") as f:
        return [q.strip() for q in f.readlines() if q.strip()]


def load_golden_answers(path):
    with open(path, "r") as f:
        return json.load(f)


def compute_similarity(predicted, reference, scorer):
    embeddings = scorer.encode([predicted, reference], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return float(similarity.item())


def main():
    # Get HF token
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("Please set your Hugging Face token via the HF_TOKEN environment variable.")

    # Load tokenizer and model (CPU only)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        device_map=None,
        torch_dtype=None
    )

    # Use CPU
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

    # Load input questions
    questions = load_questions("prompts/sample_questions.txt")

    # Load golden answers
    golden_answers = load_golden_answers("prompts/sample_golden.json")

    # Load sentence similarity model
    scorer_model = SentenceTransformer("all-MiniLM-L6-v2")

    responses = {}

    for i, question in enumerate(questions, 1):
        prompt = f"[INST] {question} [/INST]"
        print(f"\nQuestion {i}: {question}")

        response = pipe(prompt, max_new_tokens=150)[0]["generated_text"]

        # Extract answer part after [/INST]
        answer_start = response.find("[/INST]") + len("[/INST]")
        model_answer = response[answer_start:].strip()

        golden = golden_answers.get(question, "")
        score = compute_similarity(model_answer, golden, scorer_model)*100

        print("Answer:", model_answer)
        print("Score:", score)

        responses[question] = {
            "question": question,
            "model_answer": model_answer,
            "golden_answer": golden,
            "score": score
        }

    # Save results
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/scored_responses.json", "w") as f:
        json.dump(responses, f, indent=2)


if __name__ == "__main__":
    main()
