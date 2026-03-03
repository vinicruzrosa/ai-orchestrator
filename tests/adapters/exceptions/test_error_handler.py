import pytest
from unittest.mock import AsyncMock


from fastapi.testclient import TestClient
from main import app, ai_service
from app.domain.exceptions import AIProviderError, InvalidMessageError

client = TestClient(app)


@pytest.mark.asyncio
async def test_handle_ai_provider_error():
    ai_service.process_text = AsyncMock(side_effect=AIProviderError("Groq is down"))
    response = client.post("/chat", json={"message": "I has a car"})

    assert response.status_code == 502
    assert response.json()["error"] == "AI_SERVICE_UNAVAILABLE"


@pytest.mark.asyncio
async def test_handle_invalid_message_error():
    ai_service.process_text = AsyncMock(side_effect=InvalidMessageError("Empty message"))
    response = client.post("/chat", json={"message": ""})

    assert response.status_code == 400
    assert response.json()["error"] == "INVALID_INPUT"


@pytest.mark.asyncio
async def test_handle_general_exception():
    ai_service.process_text = AsyncMock(side_effect=Exception("Unexpected crash"))
    response = client.post("/chat", json={"message": "Hello"})

    assert response.status_code == 500
    assert response.json()["error"] == "INTERNAL_SERVER_ERROR"