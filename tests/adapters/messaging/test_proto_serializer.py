import pytest

from app.adapters.messaging.proto_serializer import ProtoSerializer
from app.adapters.proto_generated import analysis_pb2
from app.domain.entities import ChatAnalysis


class TestDeserializeRequest:
    def test_parses_correlation_id(self):
        request = analysis_pb2.AnalysisRequest(
            correlation_id="abc-123",
            message="Hello world",
            requested_at="2025-01-01T00:00:00Z",
        )
        cid, _ = ProtoSerializer.deserialize_request(request.SerializeToString())
        assert cid == "abc-123"

    def test_parses_message(self):
        request = analysis_pb2.AnalysisRequest(
            correlation_id="abc-123",
            message="I goes to school",
            requested_at="2025-01-01T00:00:00Z",
        )
        _, message = ProtoSerializer.deserialize_request(request.SerializeToString())
        assert message == "I goes to school"

    def test_empty_fields_return_empty_strings(self):
        request = analysis_pb2.AnalysisRequest()
        cid, message = ProtoSerializer.deserialize_request(request.SerializeToString())
        assert cid == ""
        assert message == ""


class TestSerializeSuccess:
    def _make_analysis(self, **overrides):
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

    def test_round_trips_is_correct(self):
        analysis = self._make_analysis(is_correct=True)
        data = ProtoSerializer.serialize_success("cid-1", analysis)
        response = analysis_pb2.AnalysisResponse()
        response.ParseFromString(data)
        assert response.is_correct is True

    def test_round_trips_message(self):
        analysis = self._make_analysis(message="Test message")
        data = ProtoSerializer.serialize_success("cid-1", analysis)
        response = analysis_pb2.AnalysisResponse()
        response.ParseFromString(data)
        assert response.message == "Test message"

    def test_round_trips_suggestions(self):
        analysis = self._make_analysis(suggestions=["Option A", "Option B"])
        data = ProtoSerializer.serialize_success("cid-1", analysis)
        response = analysis_pb2.AnalysisResponse()
        response.ParseFromString(data)
        assert list(response.suggestions) == ["Option A", "Option B"]

    def test_correction_none_does_not_set_field(self):
        analysis = self._make_analysis(correction=None)
        data = ProtoSerializer.serialize_success("cid-1", analysis)
        response = analysis_pb2.AnalysisResponse()
        response.ParseFromString(data)
        assert not response.HasField("correction")

    def test_correction_present_sets_field(self):
        analysis = self._make_analysis(correction="I go to school")
        data = ProtoSerializer.serialize_success("cid-1", analysis)
        response = analysis_pb2.AnalysisResponse()
        response.ParseFromString(data)
        assert response.correction == "I go to school"

    def test_success_is_true(self):
        analysis = self._make_analysis()
        data = ProtoSerializer.serialize_success("cid-1", analysis)
        response = analysis_pb2.AnalysisResponse()
        response.ParseFromString(data)
        assert response.success is True

    def test_correlation_id_is_forwarded(self):
        analysis = self._make_analysis()
        data = ProtoSerializer.serialize_success("my-cid", analysis)
        response = analysis_pb2.AnalysisResponse()
        response.ParseFromString(data)
        assert response.correlation_id == "my-cid"

    def test_processed_at_is_set(self):
        analysis = self._make_analysis()
        data = ProtoSerializer.serialize_success("cid-1", analysis)
        response = analysis_pb2.AnalysisResponse()
        response.ParseFromString(data)
        assert len(response.processed_at) > 0

    def test_empty_suggestions_list(self):
        analysis = self._make_analysis(suggestions=[])
        data = ProtoSerializer.serialize_success("cid-1", analysis)
        response = analysis_pb2.AnalysisResponse()
        response.ParseFromString(data)
        assert list(response.suggestions) == []


class TestSerializeError:
    def test_success_is_false(self):
        data = ProtoSerializer.serialize_error("cid-err", "AI_ERROR", "Groq down")
        response = analysis_pb2.AnalysisResponse()
        response.ParseFromString(data)
        assert response.success is False

    def test_error_code_is_set(self):
        data = ProtoSerializer.serialize_error("cid-err", "AIProviderError", "Groq down")
        response = analysis_pb2.AnalysisResponse()
        response.ParseFromString(data)
        assert response.error_code == "AIProviderError"

    def test_error_message_is_set(self):
        data = ProtoSerializer.serialize_error("cid-err", "AIProviderError", "Groq is unavailable")
        response = analysis_pb2.AnalysisResponse()
        response.ParseFromString(data)
        assert response.error_message == "Groq is unavailable"

    def test_correlation_id_is_forwarded(self):
        data = ProtoSerializer.serialize_error("cid-err-42", "ERR", "msg")
        response = analysis_pb2.AnalysisResponse()
        response.ParseFromString(data)
        assert response.correlation_id == "cid-err-42"

    def test_processed_at_is_set(self):
        data = ProtoSerializer.serialize_error("cid", "ERR", "msg")
        response = analysis_pb2.AnalysisResponse()
        response.ParseFromString(data)
        assert len(response.processed_at) > 0
