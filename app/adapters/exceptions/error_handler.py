from fastapi import Request, status
from fastapi.responses import JSONResponse
from app.domain.exceptions import AIProviderError, InvalidMessageError

def add_exception_handlers(app):
    @app.exception_handler(AIProviderError)
    async def ai_provider_exception_handler(request: Request, exc: AIProviderError):
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"error": "AI_SERVICE_UNAVAILABLE", "message": exc.message},
        )

    @app.exception_handler(InvalidMessageError)
    async def invalid_message_exception_handler(request: Request, exc: InvalidMessageError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "INVALID_INPUT", "message": exc.message},
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "INTERNAL_SERVER_ERROR", "message": "Ocorreu um erro inesperado."},
        )