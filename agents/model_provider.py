from typing import Protocol, TypeVar

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