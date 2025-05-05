from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
import json
import os

HF_TOKEN = os.getenv("HF_TOKEN")

# Load your dataset
with open("dataset/pdf_qa_dataset.json") as f:
    qa_data = json.load(f)

dataset = Dataset.from_list(qa_data)

# Load tokenizer and model
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=HF_TOKEN)

# Tokenize
def format_and_tokenize(example):
    prompt = f"[INST] {example['instruction']} [/INST] {example['output']}"
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(format_and_tokenize)

# Training setup
training_args = TrainingArguments(
    output_dir="models/llama-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_dir="outputs/logs",
    save_steps=100,
    logging_steps=10,
    evaluation_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
trainer.save_model("models/llama-finetuned")
