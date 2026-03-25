import os
from pathlib import Path
from striprtf.striprtf import rtf_to_text
from docx import Document


def load_rtf(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return rtf_to_text(content)


def load_docx(filepath: str) -> str:
    doc = Document(filepath)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])


def load_document(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    if ext == ".rtf":
        return load_rtf(filepath)
    elif ext == ".docx":
        return load_docx(filepath)
    return ""


def load_all_documents(data_dir: str) -> list:
    documents = []
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"Data directory not found: {data_dir}")
        return documents

    for filepath in data_path.rglob("*"):
        if filepath.suffix.lower() not in [".rtf", ".docx"]:
            continue

        try:
            text = load_document(str(filepath))
            text = text.strip()
            if len(text) < 50:
                continue

            category = filepath.parent.name
            filename = filepath.name

            documents.append({
                "text": text,
                "metadata": {
                    "filename": filename,
                    "category": category,
                    "filepath": str(filepath),
                }
            })
        except Exception as e:
            print(f"Skipping {filepath.name}: {e}")

    print(f"Loaded {len(documents)} documents from {data_dir}")
    return documents