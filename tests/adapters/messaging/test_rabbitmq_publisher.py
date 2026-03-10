import pytest
from unittest.mock import AsyncMock, MagicMock, call

import aio_pika

from app.adapters.messaging.rabbitmq_publisher import RabbitMQPublisher


@pytest.fixture
def mock_connection():
    conn = AsyncMock()
    channel = AsyncMock()
    conn.channel.return_value = channel
    return conn, channel


class TestPublisherSetup:
    @pytest.mark.asyncio
    async def test_creates_channel(self, mock_connection):
        conn, channel = mock_connection
        publisher = RabbitMQPublisher(conn, "test.output")
        await publisher.setup()
        conn.channel.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_declares_durable_queue(self, mock_connection):
        conn, channel = mock_connection
        publisher = RabbitMQPublisher(conn, "analysis.completed")
        await publisher.setup()
        channel.declare_queue.assert_awaited_once_with("analysis.completed", durable=True)


class TestPublisherPublish:
    @pytest.mark.asyncio
    async def test_raises_if_not_setup(self):
        conn = AsyncMock()
        publisher = RabbitMQPublisher(conn, "test.output")
        with pytest.raises(RuntimeError, match="not set up"):
            await publisher.publish(b"data", "cid-1")

    @pytest.mark.asyncio
    async def test_publishes_to_correct_routing_key(self, mock_connection):
        conn, channel = mock_connection
        publisher = RabbitMQPublisher(conn, "analysis.completed")
        await publisher.setup()
        await publisher.publish(b"payload", "cid-1")

        channel.default_exchange.publish.assert_awaited_once()
        _, kwargs = channel.default_exchange.publish.call_args
        assert kwargs["routing_key"] == "analysis.completed"

    @pytest.mark.asyncio
    async def test_message_has_persistent_delivery(self, mock_connection):
        conn, channel = mock_connection
        publisher = RabbitMQPublisher(conn, "test.output")
        await publisher.setup()
        await publisher.publish(b"payload", "cid-1")

        args, _ = channel.default_exchange.publish.call_args
        message = args[0]
        assert message.delivery_mode == aio_pika.DeliveryMode.PERSISTENT

    @pytest.mark.asyncio
    async def test_message_has_correlation_id(self, mock_connection):
        conn, channel = mock_connection
        publisher = RabbitMQPublisher(conn, "test.output")
        await publisher.setup()
        await publisher.publish(b"payload", "my-correlation-id")

        args, _ = channel.default_exchange.publish.call_args
        message = args[0]
        assert message.correlation_id == "my-correlation-id"

    @pytest.mark.asyncio
    async def test_message_body_matches_input(self, mock_connection):
        conn, channel = mock_connection
        publisher = RabbitMQPublisher(conn, "test.output")
        await publisher.setup()
        await publisher.publish(b"binary-data", "cid-1")

        args, _ = channel.default_exchange.publish.call_args
        message = args[0]
        assert message.body == b"binary-data"


class TestPublisherClose:
    @pytest.mark.asyncio
    async def test_closes_channel(self, mock_connection):
        conn, channel = mock_connection
        channel.is_closed = False
        publisher = RabbitMQPublisher(conn, "test.output")
        await publisher.setup()
        await publisher.close()
        channel.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_close_if_already_closed(self, mock_connection):
        conn, channel = mock_connection
        channel.is_closed = True
        publisher = RabbitMQPublisher(conn, "test.output")
        await publisher.setup()
        await publisher.close()
        channel.close.assert_not_awaited()
