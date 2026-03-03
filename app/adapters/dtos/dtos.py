from pydantic import BaseModel, Field
from typing import List, Optional

class ChatRequestDTO(BaseModel):
    message: str = Field(..., min_length=1, examples=["I goes to school"])

class ChatResponseDTO(BaseModel):
    is_correct: bool
    correction: Optional[str]
    explanation: str
    suggestions: List[str]
    reply: str
    inferred_context: str

    class Config:
        from_attributes = True