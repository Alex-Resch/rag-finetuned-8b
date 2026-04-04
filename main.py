from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import instructor
from litellm import completion
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam, \
    ChatCompletionMessageParam
from schemas.qa import QADataset, QAPair


def process_folder(folder_path: str, is_eval: bool = False):
    for file in Path(folder_path).iterdir():
        if file.suffix not in (".pdf", ".txt", ".md"):
            continue
        chunks = seperate_in_chunks(str(file))
        generate_train_data(chunks, is_eval)

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


def generate_train_data(chunks: list[Document], is_eval: bool):
    client = instructor.from_litellm(completion)
    good_datasets = []

    for chunk in chunks:
        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(
                role="user",
                content=f"""
                    Generate 3 high-quality question-answer pairs per chunk.
                    The answer must come ONLY from the chunk content.
                    Questions should be diverse: factual, conceptual, and applied.

                    Chunk:
                    {chunk.page_content}
                """
            )
        ]

        if is_eval:
            messages.append(ChatCompletionSystemMessageParam(
                role="system",
                content="You are an eval dataset generator. Generate diverse questions with detailed reference answers."
            ))
        else:
            messages.append(ChatCompletionSystemMessageParam(
                role="system",
                content="You are a dataset generator for fine-tuning a RAG model."
            ))

        result = client.chat.completions.create(
            model="groq/llama-3.1-8b-instant", # later: groq/llama-3.3-70b-versatile
            response_model=QADataset,
            messages=messages,
            temperature=0.7,
        )

        if not is_eval: good_datasets = validate_result(result)
        dataset = good_datasets if good_datasets else result
        generate_json_files(dataset, is_eval)

def validate_result(result: QADataset) -> list[QAPair]:
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

def generate_json_files(pairs: list[QAPair], is_eval: bool):
    output_path = Path("./data/processed/dataset.jsonl")
    if is_eval:
        output_path = Path("./data/processed_evals/eval.jsonl")

    with open(output_path, "a") as f:
        for pair in pairs:
            f.write(pair.model_dump_json() + "\n")

if __name__ == "__main__":
    process_folder("./data/raw")