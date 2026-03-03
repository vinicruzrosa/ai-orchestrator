from abc import ABC, abstractmethod
from app.domain.entities import ChatAnalysis


class AIServicePort(ABC):
    @abstractmethod
    async def process_text(self, text: str) -> ChatAnalysis:
        pass
