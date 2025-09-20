"""Client wrapper for the Qwen chat completion API."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Mapping, MutableMapping, Sequence


class QwenClientError(RuntimeError):
    """Raised when the Qwen API returns an error."""


class QwenClient:
    """Minimal client for interacting with the Qwen chat completion endpoint."""

    _DEFAULT_ENDPOINT = (
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    )

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout: float | None = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required to use the Qwen client")
        if not model:
            raise ValueError("model is required to use the Qwen client")
        self._api_key = api_key
        self._model = model
        self._endpoint = base_url or self._DEFAULT_ENDPOINT
        self._timeout = timeout

    @classmethod
    def from_environment(
        cls,
        *,
        api_key_env: str = "QWEN_API_KEY",
        model_env: str = "QWEN_MODEL",
        default_model: str | None = "qwen-turbo",
        base_url_env: str = "QWEN_BASE_URL",
    ) -> "QwenClient | None":
        """Instantiate the client from standard environment variables."""

        api_key = os.environ.get(api_key_env)
        if not api_key:
            return None
        model = os.environ.get(model_env) or default_model
        base_url = os.environ.get(base_url_env)
        if not model:
            return None
        return cls(api_key=api_key, model=model, base_url=base_url)

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        extra_parameters: Mapping[str, object] | None = None,
    ) -> str:
        """Send a chat completion request and return the message content."""

        payload: MutableMapping[str, object] = {
            "model": self._model,
            "messages": [dict(message) for message in messages],
        }
        if extra_parameters:
            payload.update(dict(extra_parameters))

        request = urllib.request.Request(
            self._endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                raw_body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:  # pragma: no cover - network failure
            raise QwenClientError("Failed to reach Qwen API") from exc

        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise QwenClientError("Invalid response from Qwen API") from exc

        if isinstance(body, Mapping) and "error" in body:
            error = body["error"]
            if isinstance(error, Mapping):
                message = error.get("message") or "Qwen API error"
            else:
                message = "Qwen API error"
            raise QwenClientError(str(message))

        choices = body.get("choices") if isinstance(body, Mapping) else None
        if not choices:
            raise QwenClientError("Qwen API response did not contain choices")

        first_choice = choices[0]
        if not isinstance(first_choice, Mapping):
            raise QwenClientError("Unexpected response structure from Qwen API")

        message = first_choice.get("message")
        if not isinstance(message, Mapping):
            raise QwenClientError("Qwen API response missing message content")

        content = message.get("content")
        if not isinstance(content, str):
            raise QwenClientError("Qwen API response content must be a string")
        return content


__all__ = ["QwenClient", "QwenClientError"]
