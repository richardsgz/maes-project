from typing import Any, TypeVar

from pydantic import BaseModel

from agents.model_provider import ModelProviderError

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class GeminiStructuredModel:
    """Structured-generation adapter for Google's optional GenAI SDK."""

    def __init__(self, model_name: str, client: Any | None = None):
        self.model_name = model_name
        if client is None:
            try:
                from google import genai
            except ImportError as error:
                raise ModelProviderError(
                    "The Google provider requires the optional dependency: "
                    "install maes with the 'google' extra"
                ) from error
            client = genai.Client()
        self._client = client

    def generate_structured(
        self,
        *,
        prompt: str,
        response_model: type[ResponseT],
    ) -> ResponseT:
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": response_model,
            },
        )
        parsed_response = getattr(response, "parsed", None)
        if parsed_response is None:
            raise ModelProviderError("Google GenAI returned no parsed structured response")
        return response_model.model_validate(parsed_response)