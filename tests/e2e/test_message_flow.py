"""
End-to-end tests for the FluentBase AI Orchestrator message flow.

These tests exercise the complete pipeline:
  Protobuf request → AnalysisWorker → GroqAdapter (mocked Groq client)
  → ChatAnalysis → Protobuf response → Publisher

The only thing mocked is the Groq HTTP client, so the full pipeline —
proto deserialization, adapter logic, domain entity mapping, proto
serialization, and publishing — runs for real.
"""

import json
import pytest
import groq
from unittest.mock import AsyncMock, MagicMock

from app.adapters.ai_adapter import GroqAdapter
from app.adapters.proto_generated import analysis_pb2
from app.adapters.messaging.proto_serializer import ProtoSerializer
from app.application.worker import AnalysisWorker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_groq_response(payload: dict) -> MagicMock:
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = json.dumps(payload)
    return mock_completion


class _FakeProcessContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


def make_incoming_message(correlation_id: str, text: str) -> MagicMock:
    request = analysis_pb2.AnalysisRequest(
        correlation_id=correlation_id,
        message=text,
        requested_at="2025-01-01T00:00:00Z",
    )
    msg = MagicMock()
    msg.body = request.SerializeToString()
    msg.process.return_value = _FakeProcessContext()
    return msg


def parse_published_response(publisher: AsyncMock) -> analysis_pb2.AnalysisResponse:
    args, _ = publisher.publish.call_args
    response = analysis_pb2.AnalysisResponse()
    response.ParseFromString(args[0])
    return response


# ---------------------------------------------------------------------------
# Canned AI responses
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
    "explanation": "'Most near' is not idiomatic English. The correct superlative of 'near' is 'nearest'.",
    "suggestions": [
        "Could you recommend a hotel nearby?",
        "Is there a hotel close to here?",
    ],
    "reply": "Let me help you ask for directions more naturally!",
    "inferred_context": "Travel",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def groq_adapter():
    from unittest.mock import patch
    with patch("app.adapters.ai_adapter.os.getenv") as mock_env, \
         patch("app.adapters.ai_adapter.groq.Groq") as MockGroq:
        mock_env.return_value = "fake-api-key"
        adapter = GroqAdapter()
        yield adapter, MockGroq.return_value


@pytest.fixture
def publisher():
    return AsyncMock()


# ---------------------------------------------------------------------------
# Correct sentence flow
# ---------------------------------------------------------------------------

class TestCorrectSentenceFlow:
    @pytest.mark.asyncio
    async def test_returns_success(self, groq_adapter, publisher):
        adapter, mock_client = groq_adapter
        mock_client.chat.completions.create.return_value = make_groq_response(CORRECT_CASUAL)

        worker = AnalysisWorker(adapter, publisher)
        msg = make_incoming_message("cid-correct", "I go to school.")
        await worker.handle_message(msg)

        response = parse_published_response(publisher)
        assert response.success is True

    @pytest.mark.asyncio
    async def test_is_correct_is_true(self, groq_adapter, publisher):
        adapter, mock_client = groq_adapter
        mock_client.chat.completions.create.return_value = make_groq_response(CORRECT_CASUAL)

        worker = AnalysisWorker(adapter, publisher)
        msg = make_incoming_message("cid-1", "I go to school.")
        await worker.handle_message(msg)

        response = parse_published_response(publisher)
        assert response.is_correct is True

    @pytest.mark.asyncio
    async def test_correction_not_set(self, groq_adapter, publisher):
        adapter, mock_client = groq_adapter
        mock_client.chat.completions.create.return_value = make_groq_response(CORRECT_CASUAL)

        worker = AnalysisWorker(adapter, publisher)
        msg = make_incoming_message("cid-1", "I go to school.")
        await worker.handle_message(msg)

        response = parse_published_response(publisher)
        assert not response.HasField("correction")

    @pytest.mark.asyncio
    async def test_two_suggestions_returned(self, groq_adapter, publisher):
        adapter, mock_client = groq_adapter
        mock_client.chat.completions.create.return_value = make_groq_response(CORRECT_CASUAL)

        worker = AnalysisWorker(adapter, publisher)
        msg = make_incoming_message("cid-1", "I go to school.")
        await worker.handle_message(msg)

        response = parse_published_response(publisher)
        assert len(response.suggestions) == 2

    @pytest.mark.asyncio
    async def test_inferred_context_is_casual(self, groq_adapter, publisher):
        adapter, mock_client = groq_adapter
        mock_client.chat.completions.create.return_value = make_groq_response(CORRECT_CASUAL)

        worker = AnalysisWorker(adapter, publisher)
        msg = make_incoming_message("cid-1", "I go to school.")
        await worker.handle_message(msg)

        response = parse_published_response(publisher)
        assert response.inferred_context == "Casual"


# ---------------------------------------------------------------------------
# Incorrect sentence flow
# ---------------------------------------------------------------------------

class TestIncorrectSentenceFlow:
    @pytest.mark.asyncio
    async def test_is_correct_is_false(self, groq_adapter, publisher):
        adapter, mock_client = groq_adapter
        mock_client.chat.completions.create.return_value = make_groq_response(INCORRECT_CASUAL)

        worker = AnalysisWorker(adapter, publisher)
        msg = make_incoming_message("cid-1", "I goes to school.")
        await worker.handle_message(msg)

        response = parse_published_response(publisher)
        assert response.is_correct is False

    @pytest.mark.asyncio
    async def test_correction_is_provided(self, groq_adapter, publisher):
        adapter, mock_client = groq_adapter
        mock_client.chat.completions.create.return_value = make_groq_response(INCORRECT_CASUAL)

        worker = AnalysisWorker(adapter, publisher)
        msg = make_incoming_message("cid-1", "I goes to school.")
        await worker.handle_message(msg)

        response = parse_published_response(publisher)
        assert response.correction == "I go to school."

    @pytest.mark.asyncio
    async def test_explanation_is_non_empty(self, groq_adapter, publisher):
        adapter, mock_client = groq_adapter
        mock_client.chat.completions.create.return_value = make_groq_response(INCORRECT_CASUAL)

        worker = AnalysisWorker(adapter, publisher)
        msg = make_incoming_message("cid-1", "I goes to school.")
        await worker.handle_message(msg)

        response = parse_published_response(publisher)
        assert len(response.explanation) > 0


# ---------------------------------------------------------------------------
# Context inference
# ---------------------------------------------------------------------------

class TestContextInference:
    @pytest.mark.asyncio
    async def test_business_context_detected(self, groq_adapter, publisher):
        adapter, mock_client = groq_adapter
        mock_client.chat.completions.create.return_value = make_groq_response(BUSINESS_CORRECT)

        worker = AnalysisWorker(adapter, publisher)
        msg = make_incoming_message("cid-biz", "I would like to schedule a meeting.")
        await worker.handle_message(msg)

        response = parse_published_response(publisher)
        assert response.inferred_context == "Business"

    @pytest.mark.asyncio
    async def test_travel_context_detected(self, groq_adapter, publisher):
        adapter, mock_client = groq_adapter
        mock_client.chat.completions.create.return_value = make_groq_response(TRAVEL_INCORRECT)

        worker = AnalysisWorker(adapter, publisher)
        msg = make_incoming_message("cid-travel", "Where is the most near hotel?")
        await worker.handle_message(msg)

        response = parse_published_response(publisher)
        assert response.inferred_context == "Travel"


# ---------------------------------------------------------------------------
# Response contract
# ---------------------------------------------------------------------------

class TestResponseContract:
    @pytest.mark.asyncio
    async def test_correlation_id_matches_request(self, groq_adapter, publisher):
        adapter, mock_client = groq_adapter
        mock_client.chat.completions.create.return_value = make_groq_response(CORRECT_CASUAL)

        worker = AnalysisWorker(adapter, publisher)
        msg = make_incoming_message("unique-cid-789", "Hello.")
        await worker.handle_message(msg)

        response = parse_published_response(publisher)
        assert response.correlation_id == "unique-cid-789"

    @pytest.mark.asyncio
    async def test_processed_at_is_set(self, groq_adapter, publisher):
        adapter, mock_client = groq_adapter
        mock_client.chat.completions.create.return_value = make_groq_response(CORRECT_CASUAL)

        worker = AnalysisWorker(adapter, publisher)
        msg = make_incoming_message("cid-1", "Hello.")
        await worker.handle_message(msg)

        response = parse_published_response(publisher)
        assert len(response.processed_at) > 0

    @pytest.mark.asyncio
    async def test_message_field_echoes_input(self, groq_adapter, publisher):
        adapter, mock_client = groq_adapter
        mock_client.chat.completions.create.return_value = make_groq_response(CORRECT_CASUAL)

        worker = AnalysisWorker(adapter, publisher)
        msg = make_incoming_message("cid-1", "Hello.")
        await worker.handle_message(msg)

        response = parse_published_response(publisher)
        assert response.message == "Hello."


# ---------------------------------------------------------------------------
# Error handling (full pipeline)
# ---------------------------------------------------------------------------

class TestAdapterErrorHandling:
    @pytest.mark.asyncio
    async def test_groq_api_error_publishes_error_response(self, groq_adapter, publisher):
        adapter, mock_client = groq_adapter
        mock_client.chat.completions.create.side_effect = groq.GroqError("API unavailable")

        worker = AnalysisWorker(adapter, publisher)
        msg = make_incoming_message("cid-groq-err", "Hello.")
        await worker.handle_message(msg)

        response = parse_published_response(publisher)
        assert response.success is False
        assert response.error_code == "AIProviderError"

    @pytest.mark.asyncio
    async def test_invalid_json_publishes_error_response(self, groq_adapter, publisher):
        adapter, mock_client = groq_adapter
        bad = MagicMock()
        bad.choices[0].message.content = "not valid json {{{"
        mock_client.chat.completions.create.return_value = bad

        worker = AnalysisWorker(adapter, publisher)
        msg = make_incoming_message("cid-json-err", "Hello.")
        await worker.handle_message(msg)

        response = parse_published_response(publisher)
        assert response.success is False

    @pytest.mark.asyncio
    async def test_connection_error_publishes_error_response(self, groq_adapter, publisher):
        adapter, mock_client = groq_adapter
        mock_client.chat.completions.create.side_effect = ConnectionError("network failure")

        worker = AnalysisWorker(adapter, publisher)
        msg = make_incoming_message("cid-conn-err", "Hello.")
        await worker.handle_message(msg)

        response = parse_published_response(publisher)
        assert response.success is False
