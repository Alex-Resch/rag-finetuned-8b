from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_folder(folder_path: str) -> list[Document]:
    all_chunks = []

    for file in Path(folder_path).iterdir():
        if file.suffix not in (".pdf", ".txt", ".md"):
            continue
        chunks = seperate_in_chunks(str(file))

    return all_chunks

def get_document_loader(path: str):
    if path is None:
        raise ValueError("File path is None.")
    if path.endswith(".pdf"):
        return PyPDFLoader(path)
    elif path.endswith(".txt"):
        return TextLoader(path)
    elif path.endswith(".md"):
        return UnstructuredMarkdownLoader(path)
    raise ValueError(f"Unsupported file type: '{path}'")


def seperate_in_chunks(path: str) -> list[Document]:
    loader = get_document_loader(path)
    pages = loader.load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(pages)
    return chunks