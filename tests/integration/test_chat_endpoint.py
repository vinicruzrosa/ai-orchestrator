"""
Integration tests for the FastAPI chat endpoint.

These tests verify the integration between the HTTP layer, DTO validation,
and exception handling — mocking the AI service at the port boundary so the
full request/response pipeline is exercised without real Groq calls.
"""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.domain.entities import ChatAnalysis
from app.domain.exceptions import AIProviderError, InvalidMessageError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_analysis(
    message="Hello",
    is_correct=True,
    explanation="Correct sentence.",
    reply="Great!",
    inferred_context="Casual",
    correction=None,
    suggestions=None,
) -> ChatAnalysis:
    return ChatAnalysis(
        message=message,
        is_correct=is_correct,
        explanation=explanation,
        reply=reply,
        inferred_context=inferred_context,
        correction=correction,
        suggestions=suggestions or [],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from main import app
    return TestClient(app)


@pytest.fixture(scope="module")
def lenient_client():
    """Client that does not re-raise server-side exceptions.

    Needed when testing the generic Exception → 500 handler, because
    TestClient's default behaviour re-raises unhandled exceptions before
    the ASGI exception handler can produce a response.
    """
    from main import app
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------

class TestRootEndpoint:
    def test_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_returns_online_status(self, client):
        assert client.get("/").json() == {"status": "online"}

    def test_content_type_is_json(self, client):
        assert "application/json" in client.get("/").headers["content-type"]


# ---------------------------------------------------------------------------
# /chat — request validation
# ---------------------------------------------------------------------------

class TestChatRequestValidation:
    def test_empty_message_returns_422(self, client):
        response = client.post("/chat", json={"message": ""})
        assert response.status_code == 422

    def test_missing_message_field_returns_422(self, client):
        response = client.post("/chat", json={})
        assert response.status_code == 422

    def test_null_message_returns_422(self, client):
        response = client.post("/chat", json={"message": None})
        assert response.status_code == 422

    def test_non_json_body_returns_422(self, client):
        response = client.post(
            "/chat",
            content="plain text",
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code in (415, 422)

    def test_extra_fields_are_ignored(self, client):
        analysis = make_analysis()
        with patch("main.ai_service") as mock_service:
            mock_service.process_text = AsyncMock(return_value=analysis)
            response = client.post(
                "/chat", json={"message": "Hello", "unexpected": "field"}
            )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# /chat — successful responses
# ---------------------------------------------------------------------------

class TestChatEndpointSuccess:
    def test_valid_request_returns_200(self, client):
        with patch("main.ai_service") as mock_service:
            mock_service.process_text = AsyncMock(return_value=make_analysis())
            response = client.post("/chat", json={"message": "Hello"})
        assert response.status_code == 200

    def test_response_contains_all_required_fields(self, client):
        with patch("main.ai_service") as mock_service:
            mock_service.process_text = AsyncMock(return_value=make_analysis())
            data = client.post("/chat", json={"message": "Hello"}).json()

        required = {
            "is_correct", "correction", "explanation",
            "suggestions", "reply", "inferred_context",
        }
        assert required.issubset(data.keys())

    def test_correct_sentence_has_is_correct_true(self, client):
        analysis = make_analysis(is_correct=True, correction=None)
        with patch("main.ai_service") as mock_service:
            mock_service.process_text = AsyncMock(return_value=analysis)
            data = client.post("/chat", json={"message": "I go to school"}).json()

        assert data["is_correct"] is True

    def test_correct_sentence_has_null_correction(self, client):
        analysis = make_analysis(is_correct=True, correction=None)
        with patch("main.ai_service") as mock_service:
            mock_service.process_text = AsyncMock(return_value=analysis)
            data = client.post("/chat", json={"message": "I go to school"}).json()

        assert data["correction"] is None

    def test_incorrect_sentence_has_is_correct_false(self, client):
        analysis = make_analysis(
            message="I goes to school",
            is_correct=False,
            correction="I go to school",
            explanation="Use 'go' with 'I'.",
        )
        with patch("main.ai_service") as mock_service:
            mock_service.process_text = AsyncMock(return_value=analysis)
            data = client.post("/chat", json={"message": "I goes to school"}).json()

        assert data["is_correct"] is False

    def test_incorrect_sentence_returns_correction(self, client):
        analysis = make_analysis(
            is_correct=False,
            correction="I go to school",
        )
        with patch("main.ai_service") as mock_service:
            mock_service.process_text = AsyncMock(return_value=analysis)
            data = client.post("/chat", json={"message": "I goes to school"}).json()

        assert data["correction"] == "I go to school"

    def test_suggestions_field_is_a_list(self, client):
        analysis = make_analysis(suggestions=["Option A", "Option B"])
        with patch("main.ai_service") as mock_service:
            mock_service.process_text = AsyncMock(return_value=analysis)
            data = client.post("/chat", json={"message": "Hello"}).json()

        assert isinstance(data["suggestions"], list)

    def test_suggestions_contain_expected_items(self, client):
        analysis = make_analysis(suggestions=["Option A", "Option B"])
        with patch("main.ai_service") as mock_service:
            mock_service.process_text = AsyncMock(return_value=analysis)
            data = client.post("/chat", json={"message": "Hello"}).json()

        assert "Option A" in data["suggestions"]
        assert "Option B" in data["suggestions"]

    def test_inferred_context_is_returned(self, client):
        analysis = make_analysis(inferred_context="Business")
        with patch("main.ai_service") as mock_service:
            mock_service.process_text = AsyncMock(return_value=analysis)
            data = client.post("/chat", json={"message": "Hello"}).json()

        assert data["inferred_context"] == "Business"

    def test_service_is_called_with_exact_message(self, client):
        analysis = make_analysis()
        with patch("main.ai_service") as mock_service:
            mock_service.process_text = AsyncMock(return_value=analysis)
            client.post("/chat", json={"message": "My unique test message"})

        mock_service.process_text.assert_called_once_with("My unique test message")

    def test_service_is_called_exactly_once_per_request(self, client):
        analysis = make_analysis()
        with patch("main.ai_service") as mock_service:
            mock_service.process_text = AsyncMock(return_value=analysis)
            client.post("/chat", json={"message": "Hello"})

        assert mock_service.process_text.call_count == 1


# ---------------------------------------------------------------------------
# /chat — error propagation
# ---------------------------------------------------------------------------

class TestChatEndpointErrors:
    def test_ai_provider_error_returns_502(self, client):
        with patch("main.ai_service") as mock_service:
            mock_service.process_text = AsyncMock(
                side_effect=AIProviderError("Groq is down")
            )
            response = client.post("/chat", json={"message": "Hello"})

        assert response.status_code == 502

    def test_ai_provider_error_body_has_error_code(self, client):
        with patch("main.ai_service") as mock_service:
            mock_service.process_text = AsyncMock(
                side_effect=AIProviderError("Groq is down")
            )
            data = client.post("/chat", json={"message": "Hello"}).json()

        assert data["error"] == "AI_SERVICE_UNAVAILABLE"

    def test_ai_provider_error_body_has_message_field(self, client):
        with patch("main.ai_service") as mock_service:
            mock_service.process_text = AsyncMock(
                side_effect=AIProviderError("Groq is down")
            )
            data = client.post("/chat", json={"message": "Hello"}).json()

        assert "message" in data

    def test_invalid_message_error_returns_400(self, client):
        with patch("main.ai_service") as mock_service:
            mock_service.process_text = AsyncMock(
                side_effect=InvalidMessageError("Message too short")
            )
            response = client.post("/chat", json={"message": "Hi"})

        assert response.status_code == 400

    def test_invalid_message_error_body_has_error_code(self, client):
        with patch("main.ai_service") as mock_service:
            mock_service.process_text = AsyncMock(
                side_effect=InvalidMessageError("Message too short")
            )
            data = client.post("/chat", json={"message": "Hi"}).json()

        assert data["error"] == "INVALID_INPUT"

    def test_generic_exception_returns_500(self, lenient_client):
        with patch("main.ai_service") as mock_service:
            mock_service.process_text = AsyncMock(
                side_effect=RuntimeError("Unexpected crash")
            )
            response = lenient_client.post("/chat", json={"message": "Hello"})

        assert response.status_code == 500

    def test_generic_exception_body_has_error_code(self, lenient_client):
        with patch("main.ai_service") as mock_service:
            mock_service.process_text = AsyncMock(
                side_effect=RuntimeError("Unexpected crash")
            )
            data = lenient_client.post("/chat", json={"message": "Hello"}).json()

        assert data["error"] == "INTERNAL_SERVER_ERROR"
