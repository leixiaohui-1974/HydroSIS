"""LLM-backed narrative helpers for HydroSIS reports."""
from __future__ import annotations

import json
import os
import socket
from typing import Any, Dict
from urllib import error, request

__all__ = ["qwen_narrative"]


_DASHSCOPE_COMPATIBLE_ENDPOINT = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
)


def _load_api_key(explicit: str | None) -> str:
    api_key = (
        explicit
        or os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("QWEN_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "Qwen API key missing. Provide it via the api_key argument or set "
            "the DASHSCOPE_API_KEY/QWEN_API_KEY environment variable."
        )
    return api_key


def _build_payload(prompt: str, model: str) -> Dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }


def _decode_response(data: bytes) -> str:
    try:
        payload = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive branch
        raise RuntimeError("Failed to decode Qwen API response as JSON") from exc

    choices = payload.get("choices")
    if not choices:
        raise RuntimeError("Qwen API response did not include any choices")

    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    if not message or "content" not in message:
        raise RuntimeError("Qwen API response missing message content")

    return str(message["content"]).strip()


def qwen_narrative(
    prompt: str,
    *,
    api_key: str | None = None,
    model: str = "qwen-plus",
) -> str:
    """Generate a markdown narrative by calling the Qwen chat-completions API.

    The function posts the supplied ``prompt`` to DashScope's OpenAI-compatible
    endpoint and returns the textual content from the first choice. Network
    failures, server errors and malformed responses are surfaced as
    :class:`RuntimeError` exceptions with descriptive messages.
    """

    payload = json.dumps(_build_payload(prompt, model)).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_load_api_key(api_key)}",
    }
    req = request.Request(
        _DASHSCOPE_COMPATIBLE_ENDPOINT,
        data=payload,
        headers=headers,
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=30) as response:
            response_body = response.read()
    except error.HTTPError as exc:
        detail: str
        try:
            detail = exc.read().decode("utf-8", "ignore")
        except Exception:  # pragma: no cover - highly defensive
            detail = str(exc)
        raise RuntimeError(
            f"Qwen API request failed with status {exc.code}: {detail}"
        ) from None
    except (error.URLError, socket.timeout, OSError) as exc:
        raise RuntimeError(f"Qwen API request failed: {exc}") from None

    return _decode_response(response_body)
