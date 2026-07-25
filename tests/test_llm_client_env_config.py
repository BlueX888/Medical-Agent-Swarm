import pytest

import core.llm_client as llm_client


LLM_ENV_NAMES = (
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_TEMPERATURE",
    "OPENAI_MAX_TOKENS",
)


def clear_llm_environment(monkeypatch):
    for name in LLM_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_llm_client_loads_project_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=dotenv-api-key",
                "OPENAI_MODEL=dotenv-model",
                "OPENAI_BASE_URL=https://dotenv.example.test/v1",
                "OPENAI_TEMPERATURE=0.3",
                "OPENAI_MAX_TOKENS=3072",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(llm_client, "PROJECT_ENV_FILE", env_file)
    monkeypatch.setattr(llm_client, "LLM_CONFIG", {})
    clear_llm_environment(monkeypatch)

    client = llm_client.LLMClient()

    assert client.config["api_key"] == "dotenv-api-key"
    assert client.model_name == "dotenv-model"
    assert str(client.client.base_url) == "https://dotenv.example.test/v1/"
    assert client.temperature == 0.3
    assert client.max_tokens == 3072


def test_llm_client_prefers_environment_configuration(monkeypatch):
    monkeypatch.setattr(
        llm_client,
        "LLM_CONFIG",
        {
            "api_key": "file-api-key",
            "model_name": "file-model",
            "base_url": "https://file.example.test/v1",
        },
    )
    monkeypatch.setenv("OPENAI_API_KEY", "env-api-key")
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_TEMPERATURE", "0.25")
    monkeypatch.setenv("OPENAI_MAX_TOKENS", "2048")

    client = llm_client.LLMClient()

    assert client.config["api_key"] == "env-api-key"
    assert client.model_name == "env-model"
    assert str(client.client.base_url) == "https://example.test/v1/"
    assert client.temperature == 0.25
    assert client.max_tokens == 2048


def test_llm_client_keeps_config_file_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_client, "PROJECT_ENV_FILE", tmp_path / "missing.env")
    monkeypatch.setattr(
        llm_client,
        "LLM_CONFIG",
        {
            "api_key": "file-api-key",
            "model_name": "file-model",
            "base_url": "https://file.example.test/v1",
            "temperature": 0.4,
            "max_tokens": 1024,
        },
    )
    clear_llm_environment(monkeypatch)

    client = llm_client.LLMClient()

    assert client.config["api_key"] == "file-api-key"
    assert client.model_name == "file-model"
    assert str(client.client.base_url) == "https://file.example.test/v1/"
    assert client.temperature == 0.4
    assert client.max_tokens == 1024


@pytest.mark.parametrize(
    ("variable", "value", "message"),
    [
        ("OPENAI_TEMPERATURE", "warm", "OPENAI_TEMPERATURE must be a number"),
        ("OPENAI_MAX_TOKENS", "many", "OPENAI_MAX_TOKENS must be an integer"),
    ],
)
def test_llm_client_rejects_invalid_numeric_environment_values(
    monkeypatch,
    tmp_path,
    variable,
    value,
    message,
):
    monkeypatch.setattr(llm_client, "PROJECT_ENV_FILE", tmp_path / "missing.env")
    monkeypatch.setattr(
        llm_client,
        "LLM_CONFIG",
        {"api_key": "file-api-key", "model_name": "file-model"},
    )
    clear_llm_environment(monkeypatch)
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValueError, match=message):
        llm_client.LLMClient()


def test_llm_client_reports_missing_required_configuration(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_client, "PROJECT_ENV_FILE", tmp_path / "missing.env")
    monkeypatch.setattr(llm_client, "LLM_CONFIG", {})
    clear_llm_environment(monkeypatch)

    with pytest.raises(
        ValueError,
        match="set OPENAI_API_KEY, OPENAI_MODEL in .env",
    ):
        llm_client.LLMClient()


def test_llm_client_uses_defaults_for_optional_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_client, "PROJECT_ENV_FILE", tmp_path / "missing.env")
    monkeypatch.setattr(
        llm_client,
        "LLM_CONFIG",
        {"api_key": "file-api-key", "model_name": "file-model"},
    )
    clear_llm_environment(monkeypatch)

    client = llm_client.LLMClient()

    assert str(client.client.base_url) == "https://api.openai.com/v1/"
    assert client.temperature == 0.7
    assert client.max_tokens == 8192
