from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ChatAnalysis:
    message: str
    is_correct: bool
    explanation: str
    reply: str
    inferred_context: str
    correction: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)