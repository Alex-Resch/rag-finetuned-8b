from langchain_litellm import ChatLiteLLM
from pathlib import Path

from eval.judge import judge_answers
from main import process_folder
from schemas.qa import QAPair

base_model = "llama-3.1-8b-instant"
finetuned_model = "openai/local-model"

process_folder("./data/raw_evals", is_eval=True)
files = list(Path("./data/processed_evals").glob("*"))

qa_pairs: list[QAPair] = []
for file in files:
    with open(file) as f:
        for line in f:
            qa_pairs.append(QAPair.model_validate_json(line))


def qa_model(model, chunk_question: str):
    llm = ChatLiteLLM(model=model, temperature=0)
    return llm.invoke(chunk_question)


for qa_pair in qa_pairs:
    base_answer = qa_model(base_model, qa_pair.question)
    finetuned_answer = qa_model(finetuned_model, qa_pair.question)

    result_1 = judge_answers(base_answer.content, finetuned_answer.content, qa_pair.answer)
    result_2 = judge_answers(finetuned_answer.content, base_answer.content, qa_pair.answer)

    base_score = (result_1.a_score + result_2.a_score) / 2
    finetuned_score = (result_1.b_score + result_2.b_score) / 2

    print(f"Base model score: {base_score}")
    print(f"Finetuned model score: {finetuned_score}")