import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.adapters.proto_generated import analysis_pb2
from app.adapters.messaging.proto_serializer import ProtoSerializer
from app.application.worker import AnalysisWorker
from app.domain.entities import ChatAnalysis
from app.domain.exceptions import AIProviderError, InvalidMessageError


class _FakeProcessContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


def _make_incoming_message(correlation_id: str, text: str) -> MagicMock:
    request = analysis_pb2.AnalysisRequest(
        correlation_id=correlation_id,
        message=text,
        requested_at="2025-01-01T00:00:00Z",
    )
    msg = MagicMock()
    msg.body = request.SerializeToString()
    msg.process.return_value = _FakeProcessContext()
    return msg


def _make_analysis(**overrides) -> ChatAnalysis:
    defaults = dict(
        message="Hello",
        is_correct=True,
        explanation="Correct.",
        reply="Great!",
        inferred_context="Casual",
        correction=None,
        suggestions=["Hi there", "Hey"],
    )
    defaults.update(overrides)
    return ChatAnalysis(**defaults)


def _parse_published_response(publisher_mock: AsyncMock) -> analysis_pb2.AnalysisResponse:
    args, _ = publisher_mock.publish.call_args
    response = analysis_pb2.AnalysisResponse()
    response.ParseFromString(args[0])
    return response


class TestSuccessFlow:
    @pytest.mark.asyncio
    async def test_publishes_success_response(self):
        ai_service = AsyncMock()
        ai_service.process_text.return_value = _make_analysis()
        publisher = AsyncMock()

        worker = AnalysisWorker(ai_service, publisher)
        message = _make_incoming_message("cid-1", "Hello")

        await worker.handle_message(message)

        publisher.publish.assert_awaited_once()
        response = _parse_published_response(publisher)
        assert response.success is True

    @pytest.mark.asyncio
    async def test_forwards_correlation_id(self):
        ai_service = AsyncMock()
        ai_service.process_text.return_value = _make_analysis()
        publisher = AsyncMock()

        worker = AnalysisWorker(ai_service, publisher)
        message = _make_incoming_message("my-unique-cid", "Hello")

        await worker.handle_message(message)

        response = _parse_published_response(publisher)
        assert response.correlation_id == "my-unique-cid"

    @pytest.mark.asyncio
    async def test_calls_ai_service_with_message_text(self):
        ai_service = AsyncMock()
        ai_service.process_text.return_value = _make_analysis()
        publisher = AsyncMock()

        worker = AnalysisWorker(ai_service, publisher)
        message = _make_incoming_message("cid-1", "I goes to school")

        await worker.handle_message(message)

        ai_service.process_text.assert_awaited_once_with("I goes to school")

    @pytest.mark.asyncio
    async def test_response_contains_analysis_fields(self):
        analysis = _make_analysis(
            message="Test",
            is_correct=False,
            correction="I go to school",
            explanation="Verb fix",
            reply="Nice try!",
            inferred_context="Casual",
            suggestions=["Option A"],
        )
        ai_service = AsyncMock()
        ai_service.process_text.return_value = analysis
        publisher = AsyncMock()

        worker = AnalysisWorker(ai_service, publisher)
        message = _make_incoming_message("cid-1", "Test")

        await worker.handle_message(message)

        response = _parse_published_response(publisher)
        assert response.is_correct is False
        assert response.correction == "I go to school"
        assert response.explanation == "Verb fix"
        assert response.reply == "Nice try!"
        assert response.inferred_context == "Casual"
        assert list(response.suggestions) == ["Option A"]

    @pytest.mark.asyncio
    async def test_publisher_receives_correlation_id_arg(self):
        ai_service = AsyncMock()
        ai_service.process_text.return_value = _make_analysis()
        publisher = AsyncMock()

        worker = AnalysisWorker(ai_service, publisher)
        message = _make_incoming_message("cid-42", "Hello")

        await worker.handle_message(message)

        args, _ = publisher.publish.call_args
        assert args[1] == "cid-42"


class TestDomainErrorFlow:
    @pytest.mark.asyncio
    async def test_publishes_error_on_ai_provider_error(self):
        ai_service = AsyncMock()
        ai_service.process_text.side_effect = AIProviderError("Groq is down")
        publisher = AsyncMock()

        worker = AnalysisWorker(ai_service, publisher)
        message = _make_incoming_message("cid-err", "Hello")

        await worker.handle_message(message)

        publisher.publish.assert_awaited_once()
        response = _parse_published_response(publisher)
        assert response.success is False
        assert response.error_code == "AIProviderError"
        assert "Groq is down" in response.error_message

    @pytest.mark.asyncio
    async def test_publishes_error_on_invalid_message_error(self):
        ai_service = AsyncMock()
        ai_service.process_text.side_effect = InvalidMessageError("Empty input")
        publisher = AsyncMock()

        worker = AnalysisWorker(ai_service, publisher)
        message = _make_incoming_message("cid-inv", "Hello")

        await worker.handle_message(message)

        response = _parse_published_response(publisher)
        assert response.success is False
        assert response.error_code == "InvalidMessageError"

    @pytest.mark.asyncio
    async def test_error_response_has_correlation_id(self):
        ai_service = AsyncMock()
        ai_service.process_text.side_effect = AIProviderError("fail")
        publisher = AsyncMock()

        worker = AnalysisWorker(ai_service, publisher)
        message = _make_incoming_message("err-cid-99", "Hello")

        await worker.handle_message(message)

        response = _parse_published_response(publisher)
        assert response.correlation_id == "err-cid-99"


class TestUnexpectedErrorFlow:
    @pytest.mark.asyncio
    async def test_publishes_internal_error_on_unexpected_exception(self):
        ai_service = AsyncMock()
        ai_service.process_text.side_effect = RuntimeError("unexpected crash")
        publisher = AsyncMock()

        worker = AnalysisWorker(ai_service, publisher)
        message = _make_incoming_message("cid-crash", "Hello")

        await worker.handle_message(message)

        publisher.publish.assert_awaited_once()
        response = _parse_published_response(publisher)
        assert response.success is False
        assert response.error_code == "INTERNAL_ERROR"
        assert "unexpected crash" in response.error_message

    @pytest.mark.asyncio
    async def test_message_is_always_processed(self):
        ai_service = AsyncMock()
        ai_service.process_text.side_effect = RuntimeError("crash")
        publisher = AsyncMock()

        worker = AnalysisWorker(ai_service, publisher)
        message = _make_incoming_message("cid-ack", "Hello")

        await worker.handle_message(message)

        # message.process() context manager was used
        message.process.assert_called_once()
