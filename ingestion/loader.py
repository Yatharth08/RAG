import fitz  # PyMuPDF
class DocumentLoader:
    def load_data(self, path: str) -> str:
        full_text = ""
        with fitz.open(path) as doc:
            for page in doc:
                full_text += page.get_text("text") + "\n"
        return full_text