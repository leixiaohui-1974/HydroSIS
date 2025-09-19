"""Reusable reporting templates and LLM integration hooks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping


NarrativeCallback = Callable[[str], str]


@dataclass
class ReportSection:
    """Defines a logical section in a markdown report."""

    heading: str
    prompt: str


@dataclass
class EvaluationReportTemplate:
    """Describe how an evaluation report should be structured."""

    overview: ReportSection
    highlights: ReportSection
    next_steps: ReportSection


def default_evaluation_template() -> EvaluationReportTemplate:
    return EvaluationReportTemplate(
        overview=ReportSection(
            heading="模型运行概述",
            prompt=(
                "请根据模型的总体评价指标，概述基准情景与对比情景的"
                "主要表现，并说明数据来源。"
            ),
        ),
        highlights=ReportSection(
            heading="关键发现",
            prompt=(
                "结合各子流域的径流对比和指标排序，归纳表现最佳"
                "与需要改进的模型，并说明可能的原因。"
            ),
        ),
        next_steps=ReportSection(
            heading="后续建议",
            prompt=(
                "基于当前评估结果，提出进一步校准、资料收集或情景"
                "分析的建议。"
            ),
        ),
    )


def render_template(
    builder,
    template: EvaluationReportTemplate,
    context: Mapping[str, str],
    narrator: NarrativeCallback | None = None,
) -> None:
    """Attach templated sections to a markdown builder.

    ``context`` can pre-populate key sentences for deterministic reports.
    When ``narrator`` is supplied it is expected to transform the
    template prompts into natural language descriptions (e.g. using a
    large language model).  The generated text will be appended as
    paragraphs in the markdown report.
    """

    sections = [
        template.overview,
        template.highlights,
        template.next_steps,
    ]
    for section in sections:
        builder.add_heading(section.heading, level=2)
        if section.heading in context:
            builder.add_paragraph(context[section.heading])
        elif narrator is not None:
            builder.add_paragraph(narrator(section.prompt))


__all__ = [
    "EvaluationReportTemplate",
    "ReportSection",
    "default_evaluation_template",
    "render_template",
]
