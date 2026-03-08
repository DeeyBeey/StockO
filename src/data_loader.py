from pathlib import Path
from llama_index.core import Document


def load_text_document(file_path, topic="stocks_news"):
    file_path = Path(file_path)
    text = file_path.read_text(encoding="utf-8")

    doc = Document(
        text=text,
        metadata={
            "source": str(file_path),
            "topic": topic,
        }
    )

    return doc