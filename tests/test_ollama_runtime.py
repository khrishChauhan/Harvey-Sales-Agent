"""Tests for local Ollama runtime and Brain integration."""

import json
import os
import tempfile
import httpx
import pytest
import pytest_asyncio

from sellbird.brain import Brain
from sellbird.config import BrainConfig
from sellbird.ollama_runtime import (
    OllamaTransport,
    model_matches_exact,
    validate_loopback_url,
)
from sellbird.state import StateManager


@pytest_asyncio.fixture
async def temp_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        sm = StateManager(db_path)
        await sm.init_db()
        yield sm


# ── 1. Loopback Validation ──

def test_loopback_url_allowed():
    assert validate_loopback_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434"
    assert validate_loopback_url("http://localhost:11434/") == "http://localhost:11434"
    assert validate_loopback_url("http://[::1]:11434") == "http://[::1]:11434"


def test_non_loopback_url_rejected():
    with pytest.raises(ValueError, match="Non-loopback address"):
        validate_loopback_url("http://192.168.1.100:11434")

    with pytest.raises(ValueError, match="Non-loopback address"):
        validate_loopback_url("https://api.openai.com/v1")

    with pytest.raises(ValueError, match="Non-loopback address"):
        validate_loopback_url("http://remote-server.lan:11434")


# ── 2. Exact Model Matching ──

def test_exact_model_matching():
    # Exact tag
    assert model_matches_exact("qwen3:4b", "qwen3:4b") is True
    # Default latest suffix
    assert model_matches_exact("qwen3:4b", "qwen3:4b:latest") is True
    assert model_matches_exact("qwen3", "qwen3:latest") is True

    # Rejection of loose prefix or wrong size
    assert model_matches_exact("qwen3:4b", "qwen3:40b") is False
    assert model_matches_exact("qwen3:4b", "qwen3:4b-instruct-q4") is False
    assert model_matches_exact("qwen3:4b", "qwen3:8b") is False
    assert model_matches_exact("qwen3:4b", "my-qwen3:4b") is False


# ── 3. Ollama Available and Preflight ──

@pytest.mark.asyncio
async def test_ollama_preflight_available(temp_state):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200,
            json={"models": [{"name": "qwen3:4b:latest"}, {"name": "llama3:8b"}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = OllamaTransport(
        base_url="http://127.0.0.1:11434",
        model="qwen3:4b",
        state=temp_state,
        client=client,
    )

    err = await transport.preflight()
    assert err == ""
    assert transport.paused_reason == ""


# ── 4. Ollama Unavailable ──

@pytest.mark.asyncio
async def test_ollama_preflight_unavailable(temp_state):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = OllamaTransport(
        base_url="http://127.0.0.1:11434",
        model="qwen3:4b",
        state=temp_state,
        client=client,
    )

    err = await transport.preflight()
    assert "offline" in err
    assert transport.paused_reason != ""


# ── 5. Missing Model in Ollama ──

@pytest.mark.asyncio
async def test_ollama_preflight_missing_model(temp_state):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "mistral:7b"}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = OllamaTransport(
        base_url="http://127.0.0.1:11434",
        model="qwen3:4b",
        state=temp_state,
        client=client,
    )

    err = await transport.preflight()
    assert "not installed in Ollama" in err
    assert "ollama pull qwen3:4b" in err


# ── 6. Successful Inference (Text) ──

@pytest.mark.asyncio
async def test_ollama_successful_text_inference(temp_state):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        assert body["model"] == "qwen3:4b"
        assert body["messages"][0]["content"] == "Hello local AI"
        assert body["options"]["num_ctx"] == 2048
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "Hello from Qwen3 4B!"},
                "prompt_eval_count": 15,
                "eval_count": 8,
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = OllamaTransport(
        model="qwen3:4b", state=temp_state, client=client
    )

    result = await transport(
        "Hello local AI", agent="test_agent", task="greeting"
    )
    assert result == "Hello from Qwen3 4B!"

    # Verify telemetry in SQLite
    logs = await temp_state.get_local_inference_logs(limit=5)
    assert len(logs) == 1
    assert logs[0]["model"] == "qwen3:4b"
    assert logs[0]["agent"] == "test_agent"
    assert logs[0]["task"] == "greeting"
    assert logs[0]["prompt_tokens"] == 15
    assert logs[0]["completion_tokens"] == 8
    assert logs[0]["status"] == "success"


# ── 7. Successful Inference (JSON format) and Brain think_json ──

@pytest.mark.asyncio
async def test_ollama_successful_json_inference(temp_state):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        assert body.get("format") == "json"
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": '{"score": 85, "reason": "Strong ICP fit"}',
                },
                "prompt_eval_count": 40,
                "eval_count": 12,
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = OllamaTransport(
        model="qwen3:4b", state=temp_state, client=client
    )
    brain = Brain(temp_state, transport=transport)

    parsed = await brain.think_json("Score this prospect", agent="scout", task="score")
    assert parsed == {"score": 85, "reason": "Strong ICP fit"}


# ── 8. Invalid / Malformed JSON Envelope Handling ──

@pytest.mark.asyncio
async def test_ollama_invalid_json_envelope(temp_state):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"Non-JSON error from server")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = OllamaTransport(
        model="qwen3:4b", state=temp_state, client=client
    )

    result = await transport("Test prompt", agent="scout", task="test")
    assert result == ""

    logs = await temp_state.get_local_inference_logs(limit=1)
    assert len(logs) == 1
    assert logs[0]["status"] == "error"
    assert "Malformed" in logs[0]["error_message"]


# ── 9. Request Timeout Handling ──

@pytest.mark.asyncio
async def test_ollama_timeout_handling(temp_state):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("Read timed out")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = OllamaTransport(
        model="qwen3:4b", state=temp_state, client=client
    )

    result = await transport("Test timeout", timeout=1.0)
    assert result == ""

    logs = await temp_state.get_local_inference_logs(limit=1)
    assert len(logs) == 1
    assert logs[0]["status"] == "error"
    assert "timed out" in logs[0]["error_message"]


# ── 10. Strict No Cloud Fallback ──

@pytest.mark.asyncio
async def test_strict_no_cloud_fallback(temp_state, monkeypatch):
    # If Claude or Codex CLI were attempted, raise immediately
    def forbid_exec(*args, **kwargs):
        raise AssertionError("Prohibited external CLI executed!")

    import asyncio
    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbid_exec)

    def failing_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Ollama is not running")

    client = httpx.AsyncClient(transport=httpx.MockTransport(failing_handler))
    transport = OllamaTransport(model="qwen3:4b", state=temp_state, client=client)
    brain = Brain(temp_state, transport=transport)

    # When Ollama fails, it returns empty string and NEVER falls back to cloud
    result = await brain.think("test prompt")
    assert result == ""
