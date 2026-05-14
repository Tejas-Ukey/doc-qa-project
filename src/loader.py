from PyPDF2 import PdfReader
from langchain_core.documents import Document

def load_pdfs(pdf_files):
    documents = []
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": pdf.name,
                            "page": page_num + 1
                        }
                    )
                )
    return documents