import logging

import aio_pika

from app.application.ports import AIServicePort
from app.adapters.messaging.proto_serializer import ProtoSerializer
from app.adapters.messaging.rabbitmq_publisher import RabbitMQPublisher
from app.domain.exceptions import DomainException

logger = logging.getLogger(__name__)


class AnalysisWorker:

    def __init__(self, ai_service: AIServicePort, publisher: RabbitMQPublisher):
        self._ai_service = ai_service
        self._publisher = publisher

    async def handle_message(self, message: aio_pika.abc.AbstractIncomingMessage) -> None:
        async with message.process():
            correlation_id = "unknown"
            try:
                correlation_id, text = ProtoSerializer.deserialize_request(message.body)
                logger.info(
                    "Processing correlation_id=%s, message='%s'",
                    correlation_id,
                    text[:50],
                )

                analysis = await self._ai_service.process_text(text)

                response_bytes = ProtoSerializer.serialize_success(correlation_id, analysis)
                await self._publisher.publish(response_bytes, correlation_id)

            except DomainException as e:
                logger.error(
                    "Domain error for correlation_id=%s: %s",
                    correlation_id,
                    e.message,
                )
                error_bytes = ProtoSerializer.serialize_error(
                    correlation_id,
                    error_code=type(e).__name__,
                    error_message=e.message,
                )
                await self._publisher.publish(error_bytes, correlation_id)

            except Exception as e:
                logger.exception("Unexpected error for correlation_id=%s", correlation_id)
                error_bytes = ProtoSerializer.serialize_error(
                    correlation_id,
                    error_code="INTERNAL_ERROR",
                    error_message=str(e),
                )
                await self._publisher.publish(error_bytes, correlation_id)
