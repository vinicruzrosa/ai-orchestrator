import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.adapters.messaging.rabbitmq_consumer import RabbitMQConsumer


@pytest.fixture
def mock_connection():
    conn = AsyncMock()
    channel = AsyncMock()
    queue = AsyncMock()
    conn.channel.return_value = channel
    channel.declare_queue.return_value = queue
    return conn, channel, queue


class TestConsumerSetup:
    @pytest.mark.asyncio
    async def test_creates_channel(self, mock_connection):
        conn, channel, queue = mock_connection
        consumer = RabbitMQConsumer(conn, "test.queue")
        await consumer.setup()
        conn.channel.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sets_qos(self, mock_connection):
        conn, channel, queue = mock_connection
        consumer = RabbitMQConsumer(conn, "test.queue", prefetch_count=5)
        await consumer.setup()
        channel.set_qos.assert_awaited_once_with(prefetch_count=5)

    @pytest.mark.asyncio
    async def test_declares_durable_queue(self, mock_connection):
        conn, channel, queue = mock_connection
        consumer = RabbitMQConsumer(conn, "analysis.requested")
        await consumer.setup()
        channel.declare_queue.assert_awaited_once_with("analysis.requested", durable=True)

    @pytest.mark.asyncio
    async def test_default_prefetch_is_one(self, mock_connection):
        conn, channel, queue = mock_connection
        consumer = RabbitMQConsumer(conn, "test.queue")
        await consumer.setup()
        channel.set_qos.assert_awaited_once_with(prefetch_count=1)


class TestConsumerConsume:
    @pytest.mark.asyncio
    async def test_raises_if_not_setup(self):
        conn = AsyncMock()
        consumer = RabbitMQConsumer(conn, "test.queue")
        callback = AsyncMock()
        with pytest.raises(RuntimeError, match="not set up"):
            await consumer.consume(callback)

    @pytest.mark.asyncio
    async def test_registers_callback(self, mock_connection):
        conn, channel, queue = mock_connection
        consumer = RabbitMQConsumer(conn, "test.queue")
        await consumer.setup()
        callback = AsyncMock()

        # consume() blocks on asyncio.Future(), so we cancel it immediately
        import asyncio
        task = asyncio.create_task(consumer.consume(callback))
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        queue.consume.assert_awaited_once_with(callback)


class TestConsumerClose:
    @pytest.mark.asyncio
    async def test_closes_channel(self, mock_connection):
        conn, channel, queue = mock_connection
        channel.is_closed = False
        consumer = RabbitMQConsumer(conn, "test.queue")
        await consumer.setup()
        await consumer.close()
        channel.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_close_if_already_closed(self, mock_connection):
        conn, channel, queue = mock_connection
        channel.is_closed = True
        consumer = RabbitMQConsumer(conn, "test.queue")
        await consumer.setup()
        await consumer.close()
        channel.close.assert_not_awaited()
