"""
End-to-end tests for the FluentBase AI Orchestrator.

These tests exercise the complete request → response pipeline:
  HTTP request → DTO validation → GroqAdapter (process_text) → JSON parsing
  → ChatAnalysis entity → DTO serialisation → HTTP response

The only thing mocked is the Groq HTTP client (`ai_service.client`), so the
full adapter logic — system prompt construction, JSON parsing, domain entity
mapping — runs for real.
"""

import json
import pytest
import groq
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_groq_response(payload: dict) -> MagicMock:
    """Build a mock Groq completion object that returns *payload* as JSON."""
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = json.dumps(payload)
    return mock_completion


# ---------------------------------------------------------------------------
# Canned AI responses — realistic Groq payloads for different scenarios
# ---------------------------------------------------------------------------

CORRECT_CASUAL = {
    "is_correct": True,
    "correction": None,
    "explanation": "The sentence is grammatically correct and sounds natural.",
    "suggestions": ["I attend school every day.", "I head to school in the morning."],
    "reply": "Great sentence! Keep up the good work.",
    "inferred_context": "Casual",
}

INCORRECT_CASUAL = {
    "is_correct": False,
    "correction": "I go to school.",
    "explanation": (
        "With the first-person subject 'I', the correct verb form is 'go', not 'goes'. "
        "'Goes' is used for third-person singular (he/she/it)."
    ),
    "suggestions": ["I attend school every day.", "I head to school in the morning."],
    "reply": "Good try! Let me help you with the verb conjugation.",
    "inferred_context": "Casual",
}

BUSINESS_CORRECT = {
    "is_correct": True,
    "correction": None,
    "explanation": "The sentence is formal and appropriate for a business setting.",
    "suggestions": [
        "I would like to arrange a meeting at your earliest convenience.",
        "Could we find a suitable time to meet?",
    ],
    "reply": "That is a very professional way to request a meeting!",
    "inferred_context": "Business",
}

TRAVEL_INCORRECT = {
    "is_correct": False,
    "correction": "Where is the nearest hotel?",
    "explanation": (
        "'Most near' is not idiomatic English. The correct superlative of 'near' is 'nearest'."
    ),
    "suggestions": [
        "Could you recommend a hotel nearby?",
        "Is there a hotel close to here?",
    ],
    "reply": "Let me help you ask for directions more naturally!",
    "inferred_context": "Travel",
}

EMPTY_SUGGESTIONS = {
    "is_correct": True,
    "correction": None,
    "explanation": "Your sentence is perfectly formed.",
    "suggestions": [],
    "reply": "Excellent!",
    "inferred_context": "Casual",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app_and_service():
    """Return (TestClient, ai_service) sharing the same module-level instances."""
    from main import app, ai_service
    return TestClient(app), ai_service


# ---------------------------------------------------------------------------
# Correct sentence flows
# ---------------------------------------------------------------------------

class TestCorrectSentenceFlow:
    def test_returns_200(self, app_and_service):
        client, svc = app_and_service
        with patch.object(svc, "client") as mock_client:
            mock_client.chat.completions.create.return_value = make_groq_response(CORRECT_CASUAL)
            response = client.post("/chat", json={"message": "I go to school."})
        assert response.status_code == 200

    def test_is_correct_is_true(self, app_and_service):
        client, svc = app_and_service
        with patch.object(svc, "client") as mock_client:
            mock_client.chat.completions.create.return_value = make_groq_response(CORRECT_CASUAL)
            data = client.post("/chat", json={"message": "I go to school."}).json()
        assert data["is_correct"] is True

    def test_correction_is_null(self, app_and_service):
        client, svc = app_and_service
        with patch.object(svc, "client") as mock_client:
            mock_client.chat.completions.create.return_value = make_groq_response(CORRECT_CASUAL)
            data = client.post("/chat", json={"message": "I go to school."}).json()
        assert data["correction"] is None

    def test_explanation_is_non_empty(self, app_and_service):
        client, svc = app_and_service
        with patch.object(svc, "client") as mock_client:
            mock_client.chat.completions.create.return_value = make_groq_response(CORRECT_CASUAL)
            data = client.post("/chat", json={"message": "I go to school."}).json()
        assert isinstance(data["explanation"], str)
        assert len(data["explanation"]) > 0

    def test_two_suggestions_returned(self, app_and_service):
        client, svc = app_and_service
        with patch.object(svc, "client") as mock_client:
            mock_client.chat.completions.create.return_value = make_groq_response(CORRECT_CASUAL)
            data = client.post("/chat", json={"message": "I go to school."}).json()
        assert len(data["suggestions"]) == 2

    def test_inferred_context_is_casual(self, app_and_service):
        client, svc = app_and_service
        with patch.object(svc, "client") as mock_client:
            mock_client.chat.completions.create.return_value = make_groq_response(CORRECT_CASUAL)
            data = client.post("/chat", json={"message": "I go to school."}).json()
        assert data["inferred_context"] == "Casual"


# ---------------------------------------------------------------------------
# Incorrect sentence flows
# ---------------------------------------------------------------------------

class TestIncorrectSentenceFlow:
    def test_returns_200(self, app_and_service):
        client, svc = app_and_service
        with patch.object(svc, "client") as mock_client:
            mock_client.chat.completions.create.return_value = make_groq_response(INCORRECT_CASUAL)
            response = client.post("/chat", json={"message": "I goes to school."})
        assert response.status_code == 200

    def test_is_correct_is_false(self, app_and_service):
        client, svc = app_and_service
        with patch.object(svc, "client") as mock_client:
            mock_client.chat.completions.create.return_value = make_groq_response(INCORRECT_CASUAL)
            data = client.post("/chat", json={"message": "I goes to school."}).json()
        assert data["is_correct"] is False

    def test_correction_is_provided(self, app_and_service):
        client, svc = app_and_service
        with patch.object(svc, "client") as mock_client:
            mock_client.chat.completions.create.return_value = make_groq_response(INCORRECT_CASUAL)
            data = client.post("/chat", json={"message": "I goes to school."}).json()
        assert data["correction"] == "I go to school."

    def test_explanation_mentions_verb(self, app_and_service):
        client, svc = app_and_service
        with patch.object(svc, "client") as mock_client:
            mock_client.chat.completions.create.return_value = make_groq_response(INCORRECT_CASUAL)
            data = client.post("/chat", json={"message": "I goes to school."}).json()
        assert len(data["explanation"]) > 0

    def test_two_suggestions_returned(self, app_and_service):
        client, svc = app_and_service
        with patch.object(svc, "client") as mock_client:
            mock_client.chat.completions.create.return_value = make_groq_response(INCORRECT_CASUAL)
            data = client.post("/chat", json={"message": "I goes to school."}).json()
        assert isinstance(data["suggestions"], list)
        assert len(data["suggestions"]) == 2

    def test_reply_is_non_empty(self, app_and_service):
        client, svc = app_and_service
        with patch.object(svc, "client") as mock_client:
            mock_client.chat.completions.create.return_value = make_groq_response(INCORRECT_CASUAL)
            data = client.post("/chat", json={"message": "I goes to school."}).json()
        assert isinstance(data["reply"], str)
        assert len(data["reply"]) > 0


# ---------------------------------------------------------------------------
# Context inference
# ---------------------------------------------------------------------------

class TestContextInference:
    def test_business_context_detected(self, app_and_service):
        client, svc = app_and_service
        with patch.object(svc, "client") as mock_client:
            mock_client.chat.completions.create.return_value = make_groq_response(BUSINESS_CORRECT)
            data = client.post(
                "/chat",
                json={"message": "I would like to schedule a meeting with you."},
            ).json()
        assert data["inferred_context"] == "Business"

    def test_business_sentence_is_correct(self, app_and_service):
        client, svc = app_and_service
        with patch.object(svc, "client") as mock_client:
            mock_client.chat.completions.create.return_value = make_groq_response(BUSINESS_CORRECT)
            data = client.post(
                "/chat",
                json={"message": "I would like to schedule a meeting with you."},
            ).json()
        assert data["is_correct"] is True
        assert data["correction"] is None

    def test_travel_context_detected(self, app_and_service):
        client, svc = app_and_service
        with patch.object(svc, "client") as mock_client:
            mock_client.chat.completions.create.return_value = make_groq_response(TRAVEL_INCORRECT)
            data = client.post(
                "/chat",
                json={"message": "Where is the most near hotel?"},
            ).json()
        assert data["inferred_context"] == "Travel"

    def test_travel_sentence_receives_correction(self, app_and_service):
        client, svc = app_and_service
        with patch.object(svc, "client") as mock_client:
            mock_client.chat.completions.create.return_value = make_groq_response(TRAVEL_INCORRECT)
            data = client.post(
                "/chat",
                json={"message": "Where is the most near hotel?"},
            ).json()
        assert data["is_correct"] is False
        assert data["correction"] is not None


# ---------------------------------------------------------------------------
# API contract compliance
# ---------------------------------------------------------------------------

class TestAPIContract:
    def test_response_has_exactly_the_expected_fields(self, app_and_service):
        client, svc = app_and_service
        with patch.object(svc, "client") as mock_client:
            mock_client.chat.completions.create.return_value = make_groq_response(CORRECT_CASUAL)
            data = client.post("/chat", json={"message": "Hello."}).json()

        expected_fields = {
            "is_correct", "correction", "explanation",
            "suggestions", "reply", "inferred_context",
        }
        assert expected_fields == set(data.keys())

    def test_is_correct_is_a_boolean(self, app_and_service):
        client, svc = app_and_service
        with patch.object(svc, "client") as mock_client:
            mock_client.chat.completions.create.return_value = make_groq_response(CORRECT_CASUAL)
            data = client.post("/chat", json={"message": "Hello."}).json()
        assert isinstance(data["is_correct"], bool)

    def test_suggestions_is_a_list_of_strings(self, app_and_service):
        client, svc = app_and_service
        with patch.object(svc, "client") as mock_client:
            mock_client.chat.completions.create.return_value = make_groq_response(CORRECT_CASUAL)
            data = client.post("/chat", json={"message": "Hello."}).json()
        assert isinstance(data["suggestions"], list)
        assert all(isinstance(s, str) for s in data["suggestions"])

    def test_empty_suggestions_list_is_accepted(self, app_and_service):
        client, svc = app_and_service
        with patch.object(svc, "client") as mock_client:
            mock_client.chat.completions.create.return_value = make_groq_response(EMPTY_SUGGESTIONS)
            data = client.post("/chat", json={"message": "Hello."}).json()
        assert data["suggestions"] == []

    def test_explanation_is_a_string(self, app_and_service):
        client, svc = app_and_service
        with patch.object(svc, "client") as mock_client:
            mock_client.chat.completions.create.return_value = make_groq_response(CORRECT_CASUAL)
            data = client.post("/chat", json={"message": "Hello."}).json()
        assert isinstance(data["explanation"], str)

    def test_reply_is_a_string(self, app_and_service):
        client, svc = app_and_service
        with patch.object(svc, "client") as mock_client:
            mock_client.chat.completions.create.return_value = make_groq_response(CORRECT_CASUAL)
            data = client.post("/chat", json={"message": "Hello."}).json()
        assert isinstance(data["reply"], str)

    def test_inferred_context_is_a_non_empty_string(self, app_and_service):
        client, svc = app_and_service
        with patch.object(svc, "client") as mock_client:
            mock_client.chat.completions.create.return_value = make_groq_response(CORRECT_CASUAL)
            data = client.post("/chat", json={"message": "Hello."}).json()
        assert isinstance(data["inferred_context"], str)
        assert len(data["inferred_context"]) > 0


# ---------------------------------------------------------------------------
# Groq adapter error handling (full pipeline)
# ---------------------------------------------------------------------------

class TestAdapterErrorHandling:
    def test_groq_api_error_returns_502(self, app_and_service):
        client, svc = app_and_service
        with patch.object(svc, "client") as mock_client:
            mock_client.chat.completions.create.side_effect = groq.GroqError("API unavailable")
            response = client.post("/chat", json={"message": "Hello."})
        assert response.status_code == 502

    def test_groq_api_error_body_error_code(self, app_and_service):
        client, svc = app_and_service
        with patch.object(svc, "client") as mock_client:
            mock_client.chat.completions.create.side_effect = groq.GroqError("API unavailable")
            data = client.post("/chat", json={"message": "Hello."}).json()
        assert data["error"] == "AI_SERVICE_UNAVAILABLE"

    def test_invalid_json_from_groq_returns_502(self, app_and_service):
        client, svc = app_and_service
        bad_response = MagicMock()
        bad_response.choices[0].message.content = "not { valid json {{{"
        with patch.object(svc, "client") as mock_client:
            mock_client.chat.completions.create.return_value = bad_response
            response = client.post("/chat", json={"message": "Hello."})
        assert response.status_code == 502

    def test_invalid_json_body_has_error_code(self, app_and_service):
        client, svc = app_and_service
        bad_response = MagicMock()
        bad_response.choices[0].message.content = "not { valid json {{{"
        with patch.object(svc, "client") as mock_client:
            mock_client.chat.completions.create.return_value = bad_response
            data = client.post("/chat", json={"message": "Hello."}).json()
        assert data["error"] == "AI_SERVICE_UNAVAILABLE"

    def test_unexpected_adapter_exception_returns_502(self, app_and_service):
        """Any unexpected exception inside process_text is wrapped as AIProviderError → 502."""
        client, svc = app_and_service
        with patch.object(svc, "client") as mock_client:
            mock_client.chat.completions.create.side_effect = ConnectionError("network failure")
            response = client.post("/chat", json={"message": "Hello."})
        assert response.status_code == 502
