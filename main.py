import os
import sys
import asyncio
import signal
import logging

import aio_pika

from app.adapters.ai_adapter import GroqAdapter
from app.adapters.messaging.rabbitmq_consumer import RabbitMQConsumer
from app.adapters.messaging.rabbitmq_publisher import RabbitMQPublisher
from app.application.worker import AnalysisWorker

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
INPUT_QUEUE = os.getenv("INPUT_QUEUE", "analysis.requested")
OUTPUT_QUEUE = os.getenv("OUTPUT_QUEUE", "analysis.completed")


async def main() -> None:
    logger.info("Connecting to RabbitMQ at %s", RABBITMQ_URL)
    connection = await aio_pika.connect_robust(RABBITMQ_URL)

    ai_service = GroqAdapter()
    publisher = RabbitMQPublisher(connection, OUTPUT_QUEUE)
    consumer = RabbitMQConsumer(connection, INPUT_QUEUE)
    worker = AnalysisWorker(ai_service, publisher)

    await publisher.setup()
    await consumer.setup()

    shutdown_event = asyncio.Event()

    if sys.platform != "win32":
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, shutdown_event.set)

    consumer_task = asyncio.create_task(consumer.consume(worker.handle_message))

    logger.info(
        "Worker started. Consuming from '%s', publishing to '%s'.",
        INPUT_QUEUE,
        OUTPUT_QUEUE,
    )

    try:
        await shutdown_event.wait()
    except KeyboardInterrupt:
        pass

    logger.info("Shutting down gracefully...")
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass

    await consumer.close()
    await publisher.close()
    await connection.close()
    logger.info("Worker stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
