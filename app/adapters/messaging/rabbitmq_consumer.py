import asyncio
import logging
from typing import Callable, Awaitable

import aio_pika

logger = logging.getLogger(__name__)


class RabbitMQConsumer:

    def __init__(
        self,
        connection: aio_pika.abc.AbstractRobustConnection,
        queue_name: str,
        prefetch_count: int = 1,
    ):
        self._connection = connection
        self._queue_name = queue_name
        self._prefetch_count = prefetch_count
        self._channel: aio_pika.abc.AbstractChannel | None = None
        self._queue: aio_pika.abc.AbstractQueue | None = None

    async def setup(self) -> None:
        self._channel = await self._connection.channel()
        await self._channel.set_qos(prefetch_count=self._prefetch_count)
        self._queue = await self._channel.declare_queue(self._queue_name, durable=True)

    async def consume(
        self,
        callback: Callable[[aio_pika.abc.AbstractIncomingMessage], Awaitable[None]],
    ) -> None:
        if not self._queue:
            raise RuntimeError("Consumer not set up. Call setup() first.")
        await self._queue.consume(callback)
        logger.info("Consumer started on queue '%s'", self._queue_name)
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            logger.info("Consumer cancelled, shutting down.")

    async def close(self) -> None:
        if self._channel and not self._channel.is_closed:
            await self._channel.close()
