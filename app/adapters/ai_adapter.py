import os
import json
import groq
from app.application.ports import AIServicePort
from app.domain.entities import ChatAnalysis
from app.domain.exceptions import AIProviderError


class GroqAdapter(AIServicePort):
    def __init__(self):
        self.api_key = os.getenv("FLUENT_BASE_API_KEY")
        if not self.api_key:
            raise RuntimeError("FLUENT_BASE_API_KEY not found.")

        self.client = groq.Groq(api_key=self.api_key)
        self.model = "llama-3.1-8b-instant"

    async def process_text(self, text: str) -> ChatAnalysis:
        system_prompt = (
            "You are an expert English Language Tutor. Analyze the user's sentence and return ONLY a JSON object.\n\n"
            "TASKS:\n"
            "1. Infer the social context (e.g., 'Business', 'Travel', 'Casual').\n"
            "2. Check grammar and naturalness. Provide 'correction' if needed.\n"
            "3. Explain the correction pedagogically in 'explanation'.\n"
            "4. Provide 2 natural 'suggestions' to say the same thing.\n"
            "5. Give a friendly 'reply' to the user's message.\n\n"
            "OUTPUT FORMAT (JSON ONLY):\n"
            "{\n"
            "  \"is_correct\": boolean,\n"
            "  \"correction\": \"string or null\",\n"
            "  \"explanation\": \"string\",\n"
            "  \"suggestions\": [\"string\", \"string\"],\n"
            "  \"reply\": \"string\",\n"
            "  \"inferred_context\": \"string\"\n"
            "}"
        )

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"}
            )

            raw_content = completion.choices[0].message.content
            data = json.loads(raw_content)

            return ChatAnalysis(
                message=text,
                is_correct=data.get("is_correct", False),
                correction=data.get("correction"),
                explanation=data.get("explanation", ""),
                suggestions=data.get("suggestions", []),
                reply=data.get("reply", ""),
                inferred_context=data.get("inferred_context", "Unknown")
            )

        except groq.GroqError as e:
            raise AIProviderError(f"Groq API error: {str(e)}")

        except json.JSONDecodeError:
            raise AIProviderError("Invalid JSON format returned by AI.")

        except Exception as e:
            raise AIProviderError(f"Unexpected adapter error: {str(e)}")