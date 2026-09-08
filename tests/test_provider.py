import pytest

from esg_extractor.extraction.provider import get_llm_client


def test_get_llm_client_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_llm_client("not-a-real-provider")


def test_get_llm_client_anthropic_constructs(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    client = get_llm_client("anthropic")
    assert client.model
    assert hasattr(client, "extract_metrics")
    assert hasattr(client, "answer_question")


def test_get_llm_client_gemini_constructs(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    client = get_llm_client("gemini")
    assert client.model
    assert hasattr(client, "extract_metrics")
    assert hasattr(client, "answer_question")


def test_get_llm_client_reads_env_var_when_provider_omitted(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    client = get_llm_client()
    assert client.__class__.__name__ == "GeminiClient"
