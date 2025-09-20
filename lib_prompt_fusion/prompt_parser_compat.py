"""Compatibility helpers for prompt parser selection."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence

_FUNCTION_PATTERN = re.compile(
    r"\[[^\]]*:\s*(?::\s*)?(?:bezier|catmull|linear|mean)\b",
    re.IGNORECASE,
)
_COLON_COMMA_PATTERN = re.compile(r"\[[^\]]*:\s*,")
_COLON_CLOSE_PATTERN = re.compile(r"\[[^\]]*:\s*]")


def requires_legacy_prompt_parser(prompts: Iterable[str]) -> bool:
    """Return ``True`` if any prompt needs the legacy WebUI parser."""

    for prompt in prompts:
        if not isinstance(prompt, str) or "[" not in prompt:
            continue

        if "[[" in prompt:
            return True

        if _FUNCTION_PATTERN.search(prompt):
            return True

        if _COLON_COMMA_PATTERN.search(prompt) or _COLON_CLOSE_PATTERN.search(prompt):
            return True

    return False


def convert_legacy_schedules(legacy_schedules: Sequence[Sequence]) -> List[List]:
    """Convert legacy ``ScheduledPromptConditioning`` objects to the new type."""

    from modules import prompt_parser

    return [
        [
            prompt_parser.ScheduledPromptConditioning(
                cond=schedule.cond,
                end_at_step=schedule.end_at_step,
            )
            for schedule in schedules
        ]
        for schedules in legacy_schedules
    ]

