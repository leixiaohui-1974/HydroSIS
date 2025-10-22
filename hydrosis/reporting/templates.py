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
            heading="Model Run Overview",
            prompt=(
                "Based on the overall evaluation metrics, summarize the main performance "
                "of baseline and comparison scenarios, and explain the data sources."
            ),
        ),
        highlights=ReportSection(
            heading="Key Findings",
            prompt=(
                "Combining subbasin hydrograph comparisons and metric rankings, summarize "
                "the best-performing and areas needing improvement, and explain possible causes."
            ),
        ),
        next_steps=ReportSection(
            heading="Recommendations",
            prompt=(
                "Based on current evaluation results, propose recommendations for further "
                "calibration, data collection, or scenario analysis."
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
