import pytest
from unittest.mock import MagicMock, patch
from app.adapters.ai_adapter import GroqAdapter
from app.domain.entities import ChatAnalysis


@pytest.mark.asyncio
async def test_groq_adapter_mapping():
    with patch('app.adapters.ai_adapter.os.getenv') as mock_env, \
            patch('app.adapters.ai_adapter.groq.Groq') as MockGroq:
        mock_env.return_value = "fake-api-key"

        mock_client = MockGroq.return_value
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = (
            '{"is_correct": true, "explanation": "Perfect", "reply": "Hi", '
            '"inferred_context": "Greeting", "suggestions": []}'
        )
        mock_client.chat.completions.create.return_value = mock_completion

        adapter = GroqAdapter()
        result = await adapter.process_text("Hello")

        assert isinstance(result, ChatAnalysis)
        assert result.inferred_context == "Greeting"
        assert result.is_correct is True