import fitz  # PyMuPDF
import os
import json

def extract_text_from_pdfs(pdf_folder):
    qa_data = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            doc = fitz.open(os.path.join(pdf_folder, filename))
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()

            # Basic heuristic: Look for "?" and assume the rest is the answer
            paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 30]
            for para in paragraphs:
                if "?" in para:
                    parts = para.split("?")
                    question = parts[0].strip() + "?"
                    answer = "?".join(parts[1:]).strip()
                    if len(answer) > 10:
                        qa_data.append({
                            "instruction": question,
                            "output": answer
                        })
    return qa_data

if __name__ == "__main__":
    input_folder = os.path.join("dataset", "pdfs")
    output_file = os.path.join("dataset", "pdf_qa_dataset.json")
    
    data = extract_text_from_pdfs(input_folder)
    os.makedirs("dataset", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[+] Extracted {len(data)} Q&A pairs to {output_file}")
