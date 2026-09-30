import json
from typing import Protocol, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class StructuredModel(Protocol):
    """Provider-neutral interface for schema-constrained model responses."""

    def generate_structured(
        self,
        *,
        prompt: str,
        response_model: type[ResponseT],
    ) -> ResponseT: ...


class ModelProviderError(RuntimeError):
    """Raised when a model provider cannot return the requested structured output."""


class GeminiStructuredModel:
    """Google GenAI adapter; callers can inject a client or swap the model identifier."""

    def __init__(self, model_name: str = "gemini-2.5-flash", client: genai.Client | None = None):
        self.model_name = model_name
        self._client = client if client is not None else genai.Client()

    def generate_structured(
        self,
        *,
        prompt: str,
        response_model: type[ResponseT],
    ) -> ResponseT:
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_model,
            ),
        )
        if response.parsed is None:
            raise ModelProviderError("Gemini returned no parsed structured response")
        return response_model.model_validate(response.parsed)


def json_document(document: str) -> str:
    """Encode source text as data so document content is clearly delimited in prompts."""
    return json.dumps(document, ensure_ascii=True)