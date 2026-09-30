import os
from collections.abc import Callable

from agents.model_provider import ModelProviderError, StructuredModel

ProviderFactory = Callable[[str | None], StructuredModel]


def _create_google_model(model_name: str | None) -> StructuredModel:
    from agents.providers.google_genai import GeminiStructuredModel

    return GeminiStructuredModel(model_name=model_name or "gemini-2.5-flash")


_PROVIDER_FACTORIES: dict[str, ProviderFactory] = {"google": _create_google_model}


def register_model_provider(name: str, factory: ProviderFactory) -> None:
    """Register an adapter constructor for use by the application composition root."""
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError("Provider name must not be empty")
    _PROVIDER_FACTORIES[normalized_name] = factory


def create_structured_model(
    provider: str | None = None,
    model_name: str | None = None,
) -> StructuredModel:
    """Construct the configured provider without coupling workers to a vendor SDK.

    Configuration can be passed explicitly or supplied through MAES_MODEL_PROVIDER
    and MAES_MODEL_NAME. Provider adapters are loaded only when selected.
    """
    selected_provider = (provider or os.getenv("MAES_MODEL_PROVIDER", "")).strip().lower()
    if not selected_provider:
        raise ModelProviderError(
            "Select a model provider with MAES_MODEL_PROVIDER or the provider argument"
        )

    selected_model = model_name or os.getenv("MAES_MODEL_NAME")
    factory = _PROVIDER_FACTORIES.get(selected_provider)
    if factory is None:
        raise ModelProviderError(f"No model provider is registered as {selected_provider!r}")
    return factory(selected_model)