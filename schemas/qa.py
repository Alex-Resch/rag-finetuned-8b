from pydantic import BaseModel


class QAPair(BaseModel):
    context: str
    question: str
    answer: str

class QADataset(BaseModel):
    pairs: list[QAPair]