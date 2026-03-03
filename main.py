from fastapi import FastAPI
from app.adapters.dtos.dtos import ChatRequestDTO, ChatResponseDTO
from app.adapters.ai_adapter import GroqAdapter
from app.adapters.exceptions.error_handler import add_exception_handlers

app = FastAPI(title="FluentBase: O Seu tutor de inglês na palma da sua mão.")
add_exception_handlers(app)
ai_service = GroqAdapter()


@app.post("/chat", response_model=ChatResponseDTO)
async def chat_endpoint(request: ChatRequestDTO):
    analysis_entity = await ai_service.process_text(request.message)
    return analysis_entity