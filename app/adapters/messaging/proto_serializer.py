from datetime import datetime, timezone

from app.adapters.proto_generated import analysis_pb2
from app.domain.entities import ChatAnalysis


class ProtoSerializer:

    @staticmethod
    def deserialize_request(data: bytes) -> tuple[str, str]:
        request = analysis_pb2.AnalysisRequest()
        request.ParseFromString(data)
        return request.correlation_id, request.message

    @staticmethod
    def serialize_success(correlation_id: str, analysis: ChatAnalysis) -> bytes:
        response = analysis_pb2.AnalysisResponse(
            correlation_id=correlation_id,
            message=analysis.message,
            is_correct=analysis.is_correct,
            explanation=analysis.explanation,
            reply=analysis.reply,
            inferred_context=analysis.inferred_context,
            suggestions=analysis.suggestions,
            processed_at=datetime.now(timezone.utc).isoformat(),
            success=True,
        )
        if analysis.correction is not None:
            response.correction = analysis.correction
        return response.SerializeToString()

    @staticmethod
    def serialize_error(correlation_id: str, error_code: str, error_message: str) -> bytes:
        response = analysis_pb2.AnalysisResponse(
            correlation_id=correlation_id,
            processed_at=datetime.now(timezone.utc).isoformat(),
            success=False,
            error_code=error_code,
            error_message=error_message,
        )
        return response.SerializeToString()
