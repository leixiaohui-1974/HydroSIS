"""Lightweight natural language interpreter used by the portal."""
from __future__ import annotations

import json
import re
from typing import Dict, List, Mapping, MutableMapping, Protocol, Sequence


class ChatCompletionClient(Protocol):
    """Protocol describing the subset of LLM client functionality we use."""

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Return the generated message content for the provided conversation."""


class IntentParser:
    """Interpreter translating chat messages into structured intents."""

    def __init__(
        self,
        *,
        llm_client: ChatCompletionClient | None = None,
        context_window: int = 5,
    ) -> None:
        self._llm_client = llm_client
        self._context_window = max(0, context_window)
        self._scenario_pattern = re.compile(r"scenario[s]?\s+([\w,\- ]+)", re.IGNORECASE)

    def parse(
        self,
        message: str,
        *,
        context: Sequence[Mapping[str, str]] | None = None,
    ) -> Dict[str, object]:
        """Return a structured representation of the requested action."""

        if self._llm_client:
            llm_intent = self._attempt_llm_parse(message, context=context)
            if llm_intent:
                return llm_intent

        return self._rule_based_parse(message)

    # ------------------------------------------------------------------
    # Rule-based parsing fallback
    def _rule_based_parse(self, message: str) -> Dict[str, object]:
        lowered = message.lower()
        if any(keyword in lowered for keyword in ["run", "simulate", "execute", "运行", "模拟"]):
            scenario_ids = self._extract_scenarios(message)
            return {
                "action": "run_scenarios",
                "parameters": {"scenario_ids": scenario_ids} if scenario_ids else {},
                "confidence": 0.6 if scenario_ids else 0.4,
            }
        if "create" in lowered and "scenario" in lowered:
            name = self._extract_name(message)
            return {
                "action": "create_scenario",
                "parameters": {"name": name} if name else {},
                "confidence": 0.5,
            }
        if "list" in lowered and "scenario" in lowered:
            return {"action": "list_scenarios", "parameters": {}, "confidence": 0.7}
        if "report" in lowered or "summary" in lowered:
            return {"action": "summarise_results", "parameters": {}, "confidence": 0.5}
        return {"action": "general_chat", "parameters": {}, "confidence": 0.2}

    # ------------------------------------------------------------------
    # LLM helpers
    def _attempt_llm_parse(
        self,
        message: str,
        *,
        context: Sequence[Mapping[str, str]] | None,
    ) -> Dict[str, object] | None:
        if not self._llm_client:
            return None

        llm_messages = self._build_llm_messages(message, context=context)
        try:
            response = self._llm_client.complete(llm_messages)
        except Exception:  # pragma: no cover - defensive fallback
            return None

        candidate = self._parse_llm_response(response)
        if candidate is None:
            return None
        return candidate

    def _build_llm_messages(
        self,
        message: str,
        *,
        context: Sequence[Mapping[str, str]] | None,
    ) -> List[Mapping[str, str]]:
        system_prompt = (
            "You are a helpful assistant for a hydrological modelling portal. "
            "Analyse the conversation and classify the user's latest message into an action. "
            "Only respond with valid JSON using the schema: {\"action\": string, \"parameters\": object, \"confidence\": number}."
        )

        llm_messages: List[MutableMapping[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        if context:
            trimmed_context = list(context)[-self._context_window :]
            for item in trimmed_context:
                role = item.get("role", "user")
                content = item.get("content")
                if isinstance(content, str):
                    llm_messages.append({"role": role, "content": content})

        llm_messages.append({"role": "user", "content": message})
        return llm_messages

    def _parse_llm_response(self, response: str) -> Dict[str, object] | None:
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, Mapping):
            return None

        action = data.get("action")
        if not isinstance(action, str) or not action:
            return None

        parameters = data.get("parameters")
        if parameters is None:
            parameters = {}
        elif not isinstance(parameters, Mapping):
            return None

        intent: Dict[str, object] = {
            "action": action,
            "parameters": dict(parameters),
        }

        confidence = data.get("confidence")
        if isinstance(confidence, (int, float)):
            intent["confidence"] = float(confidence)

        return intent

    # ------------------------------------------------------------------
    # Regex helpers
    def _extract_scenarios(self, message: str) -> List[str]:
        match = self._scenario_pattern.search(message)
        if not match:
            return []
        candidates = re.split(r"[,\s]+", match.group(1).strip())
        return [item for item in candidates if item]

    def _extract_name(self, message: str) -> str | None:
        name_match = re.search(
            r"scenario\s+(?:called|named)?\s*([\w\-]+)", message, re.IGNORECASE
        )
        if name_match:
            return name_match.group(1)
        return None


__all__ = ["ChatCompletionClient", "IntentParser"]
