from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import instructor
from litellm import completion
from openai.types.chat import ChatCompletionUserMessageParam
from schemas.qa import QADataset, QAPair


def process_folder(folder_path: str) -> list[Document]:
    all_chunks = []

    for file in Path(folder_path).iterdir():
        if file.suffix not in (".pdf", ".txt", ".md"):
            continue
        chunks = seperate_in_chunks(str(file))
        train_data = generate_train_data(chunks)

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


def generate_train_data(chunks: list[Document]):
    client = instructor.from_litellm(completion)

    for chunk in chunks:
        result = client.chat.completions.create(
            model="groq/llama-3.1-8b-instant", # later: groq/llama-3.3-70b-versatile
            response_model=QADataset,
            messages=[
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"""
                        You are a dataset generator for fine-tuning a RAG model.
                        Generate 3 high-quality question-answer pairs per chunk.
                        The answer must come ONLY from the chunk content.
                        Questions should be diverse: factual, conceptual, and applied.
    
                        Chunk:
                        {chunk.page_content}
                    """
                )
            ],
            temperature=0.7,
        )

        good_datasets = validate_result(result)

def validate_result(result: QADataset):
    good_datasets: list[QAPair] = []
    for pair in result.pairs:
        keep_dataset = input(f"""
        context: {pair.context} \n
        question: {pair.question} \n
        answer: {pair.answer} \n
        keep dataset? (j/n)
        """)

        if keep_dataset == "j":
            good_datasets.append(pair)
    return good_datasets

if __name__ == "__main__":
    process_folder("./data/raw")