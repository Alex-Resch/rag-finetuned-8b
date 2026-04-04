from pydantic import BaseModel


class Answers(BaseModel):
    a_score: float  # 0.0 - 1.0
    b_score: float  # 0.0 - 1.0
    reasoning: str