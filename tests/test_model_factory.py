import subprocess
import sys
from typing import TypeVar

import pytest
from pydantic import BaseModel

from agents.model_provider import ModelProviderError
from core.model_factory import create_structured_model, register_model_provider

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class StubModel:
    def __init__(self, model_name: str | None):
        self.model_name = model_name

    def generate_structured(
        self,
        *,
        prompt: str,
        response_model: type[ResponseT],
    ) -> ResponseT:
        raise NotImplementedError


def test_factory_reads_provider_and_model_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    register_model_provider("factory-test", StubModel)
    monkeypatch.setenv("MAES_MODEL_PROVIDER", "factory-test")
    monkeypatch.setenv("MAES_MODEL_NAME", "stub-v2")

    model = create_structured_model()

    assert isinstance(model, StubModel)
    assert model.model_name == "stub-v2"


def test_factory_requires_a_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAES_MODEL_PROVIDER", raising=False)

    with pytest.raises(ModelProviderError, match="Select a model provider"):
        create_structured_model()


def test_factory_rejects_unregistered_provider() -> None:
    with pytest.raises(ModelProviderError, match="No model provider is registered"):
        create_structured_model(provider="not-configured")


def test_worker_imports_do_not_load_google_sdk() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import agents.extractor; "
                "assert 'google.genai' not in sys.modules"
            ),
        ],
        check=True,
    )