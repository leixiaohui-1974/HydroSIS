"""Lightweight natural language interpreter used by the portal."""
from __future__ import annotations

import re
from typing import Dict, List


class IntentParser:
    """Rule-based interpreter translating chat messages into structured intents."""

    def __init__(self) -> None:
        self._scenario_pattern = re.compile(r"scenario[s]?\s+([\w,\- ]+)", re.IGNORECASE)

    def parse(self, message: str) -> Dict[str, object]:
        """Return a structured representation of the requested action."""

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

    def _extract_scenarios(self, message: str) -> List[str]:
        match = self._scenario_pattern.search(message)
        if not match:
            return []
        candidates = re.split(r"[,\s]+", match.group(1).strip())
        return [item for item in candidates if item]

    def _extract_name(self, message: str) -> str | None:
        name_match = re.search(r"scenario\s+(?:called|named)?\s*([\w\-]+)", message, re.IGNORECASE)
        if name_match:
            return name_match.group(1)
        return None


__all__ = ["IntentParser"]
