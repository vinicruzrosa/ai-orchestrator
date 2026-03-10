import logging

import aio_pika

logger = logging.getLogger(__name__)


class RabbitMQPublisher:

    def __init__(
        self,
        connection: aio_pika.abc.AbstractRobustConnection,
        queue_name: str,
    ):
        self._connection = connection
        self._queue_name = queue_name
        self._channel: aio_pika.abc.AbstractChannel | None = None

    async def setup(self) -> None:
        self._channel = await self._connection.channel()
        await self._channel.declare_queue(self._queue_name, durable=True)

    async def publish(self, body: bytes, correlation_id: str) -> None:
        if not self._channel:
            raise RuntimeError("Publisher not set up. Call setup() first.")
        await self._channel.default_exchange.publish(
            aio_pika.Message(
                body=body,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                correlation_id=correlation_id,
            ),
            routing_key=self._queue_name,
        )
        logger.info("Published response for correlation_id=%s", correlation_id)

    async def close(self) -> None:
        if self._channel and not self._channel.is_closed:
            await self._channel.close()
