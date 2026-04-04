from litellm import completion
import instructor
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam

from schemas.answers import Answers


SYSTEM_PROMPT = """You are an impartial judge evaluating two AI responses.
You do not know which model generated which answer.
Evaluate based only on: accuracy, completeness, and clarity.
Give a score from 0.0 to 1.0 for each response."""


def judge_answers(base_answer: str, finetuned_answer: str, reference_answer: str) -> Answers:
    client = instructor.from_litellm(completion)

    user_prompt = f"""
    Reference Answer: {reference_answer}
    
    Response A: {base_answer}
    Response B: {finetuned_answer}
    Evaluate both responses against the reference answer."""

    result = client.chat.completions.create(
        model="groq/llama-3.3-70b-versatile",
        response_model=Answers,
        messages=[
            ChatCompletionSystemMessageParam(
                role="system",
                content=SYSTEM_PROMPT,
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=user_prompt,
            ),
        ],
        temperature=0,
    )
    return result