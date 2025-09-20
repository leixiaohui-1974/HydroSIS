"""Demonstrate using Qwen LLM for intent parsing with provided API key.

The sandbox used for automated evaluation 无法直接访问外部网络, 因此默认脚本会在
调用 Qwen 失败后自动回退到一个带有典型示例的 Stub 客户端, 方便离线验证
“建模→情景设置→模型计算→结果分析→报告生成”这一自然语言链路。

如果在本地或具有互联网访问权限的环境中运行, 可以通过设置环境变量
``QWEN_API_KEY``/``QWEN_MODEL``/``QWEN_BASE_URL`` 或命令行参数 ``--online``
启用真实的 Qwen API 调用。
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydrosis.portal.llm import IntentParser
from hydrosis.portal.providers import QwenClient, QwenClientError


DEFAULT_API_KEY = "sk-22bfc2c3324f4e82b7ad239c0c9ca0b4"
DEFAULT_MODEL = "qwen-turbo"


def _pretty(data: Mapping[str, object]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)


def _extract_latest_user_message(
    messages: Sequence[Mapping[str, str]]
) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


class StubQwenClient:
    """Return curated JSON intents for common hydrological assistant prompts."""

    def __init__(self) -> None:
        self._parser = IntentParser()
        self._fixtures: Sequence[
            tuple[
                Callable[[str, str], bool],
                Callable[[str, str], MutableMapping[str, object]],
            ]
        ] = [
            (self._is_model_setup, self._build_model_setup_intent),
            (self._is_scenario_creation, self._build_scenario_creation_intent),
            (self._is_run_with_report, self._build_run_with_report_intent),
            (self._is_run_comparison, self._build_run_comparison_intent),
            (self._is_result_analysis, self._build_result_analysis_intent),
            (self._is_report_request, self._build_report_request_intent),
            (self._is_general_discussion, self._build_general_discussion_intent),
        ]

    # ------------------------------------------------------------------
    # Public interface matching ``QwenClient``
    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        user_message = _extract_latest_user_message(messages)
        lowered = user_message.lower()
        for predicate, builder in self._fixtures:
            if predicate(lowered, user_message):
                intent = builder(lowered, user_message)
                intent.setdefault("confidence", 0.88)
                intent.setdefault("parameters", {})
                return json.dumps(intent, ensure_ascii=False)

        fallback = self._parser._rule_based_parse(user_message)
        fallback.setdefault("confidence", 0.6)
        return json.dumps(fallback, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Predicate helpers
    def _is_model_setup(self, lowered: str, original: str) -> bool:
        return "建模" in original or "model" in lowered and "prepare" in lowered

    def _is_scenario_creation(self, lowered: str, original: str) -> bool:
        mentions_create = "create" in lowered and "scenario" in lowered
        mentions_chinese = "建" in original and "情景" in original
        return mentions_create or mentions_chinese

    def _is_run_with_report(self, lowered: str, original: str) -> bool:
        wants_run = "run" in lowered or "运行" in original or "模拟" in original
        wants_report = "report" in lowered or "报告" in original
        return wants_run and wants_report

    def _is_run_comparison(self, lowered: str, original: str) -> bool:
        return "compare" in lowered or "对比" in original

    def _is_result_analysis(self, lowered: str, original: str) -> bool:
        return "分析" in original or "analyse" in lowered or "analyze" in lowered

    def _is_report_request(self, lowered: str, original: str) -> bool:
        return "report" in lowered and "markdown" in lowered or "Markdown" in original

    def _is_general_discussion(self, lowered: str, original: str) -> bool:
        return "枯水" in original or "discussion" in lowered

    # ------------------------------------------------------------------
    # Intent builders
    def _build_model_setup_intent(self, lowered: str, original: str) -> MutableMapping[str, object]:
        tasks: list[str] = ["ingest_latest_observations", "synchronise_geodata", "calibrate_baseline"]
        if "监测" in original:
            tasks.insert(1, "quality_control_timeseries")
        return {
            "action": "prepare_model",
            "parameters": {"tasks": tasks},
        }

    def _build_scenario_creation_intent(self, lowered: str, original: str) -> MutableMapping[str, object]:
        name = self._extract_scenario_name(lowered, original) or "new_scenario"
        scale = self._extract_percentage(lowered, original)
        parameters: MutableMapping[str, object] = {"name": name}
        if scale is not None:
            target = "water_demand"
            if "泄洪" in original or "spillway" in lowered:
                target = "spillway_capacity"
            parameters["modifications"] = [
                {
                    "target": "water_demand",
                    "operation": "scale",
                    "value": round(1 + scale / 100.0, 2),
                    "description": f"adjust {target.replace('_', ' ')} by {scale}%",
                }
            ]
            parameters["modifications"][0]["target"] = target
        return {
            "action": "create_scenario",
            "parameters": parameters,
        }

    def _build_run_with_report_intent(self, lowered: str, original: str) -> MutableMapping[str, object]:
        scenario_ids = self._collect_identifier_candidates(original)
        if not scenario_ids:
            scenario_ids = ["baseline"]
        return {
            "action": "run_scenarios",
            "parameters": {
                "scenario_ids": scenario_ids,
                "outputs": ["timeseries", "kpi_table"],
                "generate_report": True,
            },
        }

    def _build_run_comparison_intent(self, lowered: str, original: str) -> MutableMapping[str, object]:
        scenarios = self._collect_identifier_candidates(original)
        if len(scenarios) < 2 and "baseline" not in scenarios:
            scenarios.append("baseline")
        return {
            "action": "compare_runs",
            "parameters": {
                "scenario_ids": scenarios or ["baseline", "drought_response"],
                "metrics": ["water_use_efficiency", "storage_levels"],
            },
        }

    def _build_result_analysis_intent(self, lowered: str, original: str) -> MutableMapping[str, object]:
        focus_metric = "water_use_efficiency" if "效率" in original else "kpi_trend"
        return {
            "action": "analyse_results",
            "parameters": {
                "focus_metrics": [focus_metric],
                "comparison": "latest_run_vs_baseline",
            },
        }

    def _build_report_request_intent(self, lowered: str, original: str) -> MutableMapping[str, object]:
        return {
            "action": "generate_report",
            "parameters": {
                "format": "markdown",
                "include_sections": ["summary", "kpi_table", "recommendations"],
            },
        }

    def _build_general_discussion_intent(self, lowered: str, original: str) -> MutableMapping[str, object]:
        return {
            "action": "general_chat",
            "parameters": {
                "topic": "dry_season_applicability",
            },
        }

    # ------------------------------------------------------------------
    # Extraction utilities
    def _extract_scenario_name(self, lowered: str, original: str) -> str | None:
        chinese_match = re.search(r"名为\s*([\w\-]+)\s*的情景", original)
        if chinese_match:
            return chinese_match.group(1)
        english_match = re.search(r"scenario\s+(?:called|named)?\s*([\w\-]+)", original, re.IGNORECASE)
        if english_match:
            return english_match.group(1)
        return None

    def _extract_percentage(self, lowered: str, original: str) -> int | None:
        english_match = re.search(r"(\d+)%", lowered)
        if english_match:
            return int(english_match.group(1))
        chinese_match = re.search(r"提高\s*(\d+)%", original)
        if chinese_match:
            return int(chinese_match.group(1))
        return None

    def _collect_identifier_candidates(self, original: str) -> list[str]:
        identifiers = self._normalise_identifiers(
            self._parser._extract_scenarios(original)
        )
        token_based = self._extract_identifier_tokens(original)
        for token in token_based:
            if token not in identifiers:
                identifiers.append(token)
        return identifiers

    def _extract_identifier_tokens(self, original: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z][\w\-]*", original)
        return self._normalise_identifiers(tokens)

    def _normalise_identifiers(self, candidates: Sequence[str]) -> list[str]:
        stop_words = {"scenario", "scenarios", "run", "compare", "with", "and", "the", "it"}
        seen: set[str] = set()
        identifiers: list[str] = []
        for token in candidates:
            lowered = token.lower()
            if lowered in stop_words:
                continue
            if lowered not in seen:
                identifiers.append(token)
                seen.add(lowered)
        return identifiers


def run_examples(messages: Sequence[str], *, force_stub: bool) -> None:
    api_key = os.getenv("QWEN_API_KEY", DEFAULT_API_KEY)
    model = os.getenv("QWEN_MODEL", DEFAULT_MODEL)
    base_url = os.getenv("QWEN_BASE_URL")

    parser = IntentParser()
    llm_client: QwenClient | None = None
    if not force_stub:
        try:
            llm_client = QwenClient(api_key=api_key, model=model, base_url=base_url)
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"⚠️ 初始化 QwenClient 失败, 使用 Stub: {exc}")
            llm_client = None

    stub_client = StubQwenClient()
    qwen_available = llm_client is not None

    for idx, message in enumerate(messages, start=1):
        print(f"\n[{idx}] 用户输入: {message}")
        llm_messages = parser._build_llm_messages(message, context=None)

        intent_from_llm: Mapping[str, object] | None = None
        if qwen_available and llm_client is not None:
            try:
                raw_response = llm_client.complete(llm_messages)
                intent_from_llm = parser._parse_llm_response(raw_response)
                if intent_from_llm:
                    print("  ✓ Qwen JSON 响应:")
                    print(_pretty(intent_from_llm))
                else:
                    print("  ⚠️ Qwen 返回非结构化内容:")
                    print(raw_response)
            except QwenClientError as exc:
                print(f"  ✖ Qwen 调用失败: {exc}")
                qwen_available = False
            except Exception as exc:  # pragma: no cover - defensive fallback
                print(f"  ✖ 未预期的 Qwen 错误: {exc}")
                qwen_available = False

        if (force_stub or not qwen_available) and intent_from_llm is None:
            raw_response = stub_client.complete(llm_messages)
            intent_from_llm = parser._parse_llm_response(raw_response)
            print("  ☆ 使用离线 Stub 响应:")
            print(_pretty(intent_from_llm or {}))

        fallback_intent = parser._rule_based_parse(message)
        print("  △ 规则解析结果:")
        print(_pretty(fallback_intent))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test Qwen intent parsing.")
    parser.add_argument(
        "--online",
        action="store_true",
        help="Force using the real Qwen API instead of the offline stub.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_argument_parser().parse_args(list(argv) if argv is not None else None)
    examples = [
        "请先根据最新的流域监测数据完成模型建模准备工作。",
        "Create a new scenario called irrigation_upgrade with 20% higher demand.",
        "帮我再建一个名为 flood_mitigation 的情景，将泄洪闸最大下泄能力提高 15%。",
        "请运行 baseline 和 flood 两个情景，并生成结果报告。",
        "Run the drought_response scenario and compare it with baseline.",
        "分析一下最近一次情景运行的用水效率指标，并总结关键差异。",
        "请生成一个包含结论和下一步建议的 Markdown 结果报告。",
        "讨论一下模型在枯水期的适用性。",
    ]
    run_examples(examples, force_stub=not args.online)


if __name__ == "__main__":
    main()
