"""Ollama runtime — fully local, zero-cloud inference engine for Sellbird."""

from __future__ import annotations

import ipaddress
import json
import logging
import time
import urllib.parse
from typing import Any, Optional

import httpx

logger = logging.getLogger("sellbird.ollama")

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:4b"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_CONTEXT_LENGTH = 2048
DEFAULT_TEMPERATURE = 0.2


class OllamaRuntimeError(Exception):
    """Raised when Ollama inference fails or is misconfigured."""


def validate_loopback_url(url: str) -> str:
    """Ensure the target endpoint is strictly on a loopback interface.

    Raises ValueError if the hostname is not localhost or a loopback IP.
    """
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"Invalid Ollama URL '{url}': missing hostname.")

    if hostname in ("localhost", "127.0.0.1", "::1"):
        return url.rstrip("/")

    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_loopback:
            return url.rstrip("/")
    except ValueError:
        pass

    raise ValueError(
        f"Non-loopback address '{hostname}' is prohibited. "
        "All local inference must target a validated loopback endpoint (127.0.0.1 or localhost)."
    )


def model_matches_exact(target_model: str, candidate_name: str) -> bool:
    """Check if candidate_name from Ollama tags matches target_model exactly.

    Handles explicit tags (e.g. 'qwen3:4b' matches 'qwen3:4b' or 'qwen3:4b:latest')
    while rejecting loose prefix matches (e.g. 'qwen3:4b' will never match 'qwen3:4b-extra'
    or 'qwen3:40b').
    """
    target = target_model.strip().lower()
    candidate = candidate_name.strip().lower()

    if target == candidate:
        return True

    # Ollama often adds :latest if untagged
    if ":" not in target and candidate == f"{target}:latest":
        return True
    if candidate.endswith(":latest") and candidate[:-7] == target:
        return True

    return False


class OllamaTransport:
    """Asynchronous HTTP transport connecting Sellbird Brain to local Ollama."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_OLLAMA_MODEL,
        num_ctx: int = DEFAULT_CONTEXT_LENGTH,
        temperature: float = DEFAULT_TEMPERATURE,
        state: Any = None,
        client: Optional[httpx.AsyncClient] = None,
        endpoint: Optional[str] = None,
        context_length: Optional[int] = None,
    ):
        target_url = endpoint if endpoint is not None else base_url
        self.base_url = validate_loopback_url(target_url)
        self.model = model or DEFAULT_OLLAMA_MODEL
        self.num_ctx = context_length if context_length is not None else num_ctx
        self.temperature = temperature
        self.state = state
        self._client = client
        self._paused_reason: str = ""
        self._available_models: list[str] = []

    @property
    def paused_reason(self) -> str:
        """Why Ollama calls are paused (empty string if healthy)."""
        return self._paused_reason

    async def preflight(self) -> str:
        """Check whether local Ollama is running and has the exact requested model.

        Returns an empty string on success, or an actionable error message.
        """
        try:
            if self._client is not None:
                resp = await self._client.get(f"{self.base_url}/api/tags")
            else:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"{self.base_url}/api/tags")

            if resp.status_code != 200:
                self._paused_reason = f"Ollama returned HTTP {resp.status_code} at {self.base_url}"
                return self._paused_reason

            data = resp.json()
            models_list = data.get("models", [])
            installed = [m.get("name", "") for m in models_list if isinstance(m, dict)]
            self._available_models = installed

            # Strict exact model validation
            has_model = any(model_matches_exact(self.model, name) for name in installed)
            if not has_model:
                if installed:
                    self._paused_reason = (
                        f"Exact model '{self.model}' is not installed in Ollama. "
                        f"Installed: {', '.join(installed[:5])}. "
                        f"Run: ollama pull {self.model}"
                    )
                else:
                    self._paused_reason = (
                        f"Ollama has no models installed. Run: ollama pull {self.model}"
                    )
                return self._paused_reason

            self._paused_reason = ""
            return ""

        except (httpx.ConnectError, httpx.NetworkError):
            self._paused_reason = (
                f"Ollama is offline (could not connect to {self.base_url}). "
                "Ensure Ollama is started locally ('ollama serve')."
            )
            return self._paused_reason
        except httpx.TimeoutException:
            self._paused_reason = f"Ollama health check timed out at {self.base_url}."
            return self._paused_reason
        except Exception as e:
            self._paused_reason = f"Ollama preflight check failed: {type(e).__name__}: {e}"
            return self._paused_reason

    async def __call__(
        self,
        prompt: str,
        *,
        agent: str = "",
        task: str = "",
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        expect_json: bool = False,
    ) -> str:
        """Execute inference against local Ollama. Matches Brain transport interface.

        Never falls back to cloud providers. Logs telemetry to local SQLite.
        """
        if self._paused_reason:
            logger.debug(f"Ollama call skipped: {self._paused_reason}")
            return ""

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
            },
        }
        if expect_json:
            payload["format"] = "json"

        start_time = time.monotonic()
        duration_ms = 0
        prompt_tokens = 0
        completion_tokens = 0
        status = "error"
        error_msg = ""
        response_text = ""

        try:
            if self._client is not None:
                resp = await self._client.post(f"{self.base_url}/api/chat", json=payload)
            else:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(f"{self.base_url}/api/chat", json=payload)

            duration_ms = int((time.monotonic() - start_time) * 1000)

            if resp.status_code != 200:
                error_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
                logger.error(f"Ollama call failed ({agent}/{task}): {error_msg}")
                return ""

            try:
                data = resp.json()
            except Exception as parse_err:
                error_msg = f"Malformed Ollama response envelope: {parse_err}"
                logger.error(f"Ollama JSON decode error: {error_msg}")
                return ""

            prompt_tokens = int(data.get("prompt_eval_count") or 0)
            completion_tokens = int(data.get("eval_count") or 0)

            message = data.get("message", {})
            if isinstance(message, dict):
                response_text = str(message.get("content") or "").strip()

            if response_text:
                status = "success"
            else:
                error_msg = "Empty completion content from Ollama."
                logger.warning(f"Ollama returned empty response for {agent}/{task}")

            return response_text

        except httpx.TimeoutException:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            error_msg = f"Request timed out after {timeout:.0f}s"
            logger.error(f"Ollama call timed out ({agent}/{task}): {error_msg}")
            return ""
        except (httpx.ConnectError, httpx.NetworkError) as net_err:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            error_msg = f"Connection failed: {net_err}"
            self._paused_reason = f"Ollama disconnected ({net_err})"
            logger.error(f"Ollama disconnected during call ({agent}/{task}): {net_err}")
            return ""
        except Exception as e:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            error_msg = f"Unexpected error: {type(e).__name__}: {e}"
            logger.error(f"Ollama error ({agent}/{task}): {error_msg}")
            return ""
        finally:
            # Telemetry logging to SQLite: never stores prompt text
            if self.state and hasattr(self.state, "record_local_inference"):
                try:
                    await self.state.record_local_inference(
                        model=self.model,
                        agent=agent,
                        task=task,
                        duration_ms=duration_ms,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        status=status,
                        error_message=error_msg[:300] if error_msg else "",
                    )
                except Exception as log_err:
                    logger.warning(f"Failed to record local inference telemetry: {log_err}")
