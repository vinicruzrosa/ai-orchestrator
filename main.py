import os
from fastapi import FastAPI
from app.adapters.dtos.dtos import ChatRequestDTO, ChatResponseDTO
from app.adapters.ai_adapter import GroqAdapter
from app.adapters.exceptions.error_handler import add_exception_handlers

app = FastAPI(title="FluentBase: AI Orchestrator")
add_exception_handlers(app)

ai_service = GroqAdapter()

@app.get("/")
async def root():
    return {"status": "online"}

@app.post("/chat", response_model=ChatResponseDTO)
async def chat_endpoint(request: ChatRequestDTO):
    return await ai_service.process_text(request.message)