"""Unit tests for the portal intent parser and LLM integration."""
from __future__ import annotations

from typing import List, Mapping, Sequence

from hydrosis.portal.llm import IntentParser


class DummyLLMClient:
    """Simple stand-in for an LLM chat completion client."""

    def __init__(self, responses: Sequence[str], *, raise_on_call: bool = False) -> None:
        self._responses = list(responses)
        self.raise_on_call = raise_on_call
        self.calls: List[Sequence[Mapping[str, str]]] = []

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        self.calls.append(messages)
        if self.raise_on_call:
            raise RuntimeError("simulated failure")
        if not self._responses:
            raise AssertionError("No more responses configured for dummy client")
        return self._responses.pop(0)


def test_intent_parser_llm_success_path() -> None:
    client = DummyLLMClient(
        ['{"action": "run_scenarios", "parameters": {"scenario_ids": ["baseline"]}, "confidence": 0.85}']
    )
    parser = IntentParser(llm_client=client)

    context = [{"role": "assistant", "content": "How can I help with your hydrology project?"}]
    intent = parser.parse("Please run the baseline scenario", context=context)

    assert intent["action"] == "run_scenarios"
    assert intent["parameters"] == {"scenario_ids": ["baseline"]}
    assert intent["confidence"] == 0.85

    assert client.calls, "Expected the LLM client to be invoked"
    llm_messages = client.calls[0]
    assert llm_messages[0]["role"] == "system"
    assert llm_messages[-1]["content"].endswith("baseline scenario")


def test_intent_parser_llm_failure_falls_back_to_rules() -> None:
    client = DummyLLMClient(['{"unexpected": "value"}'], raise_on_call=True)
    parser = IntentParser(llm_client=client)

    intent = parser.parse("Run scenario S1")

    assert intent["action"] == "run_scenarios"
    assert intent["parameters"]["scenario_ids"] == ["S1"]
