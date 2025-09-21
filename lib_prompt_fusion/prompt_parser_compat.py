"""Compatibility helpers for prompt parser selection."""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

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


def unpack_conditioning_options(
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> Tuple[Optional[Any], bool]:
    """Extract ``hires_steps`` and ``use_old_scheduling`` from positional/keyword args.

    The WebUI may pass these parameters either positionally or via keywords,
    and ``use_old_scheduling`` defaults to ``False`` in the upstream parser.
    This helper normalises the values so the extension can follow the same
    contract regardless of how the caller supplied them.
    """

    hires_steps = kwargs.get("hires_steps")
    use_old_scheduling = kwargs.get("use_old_scheduling")

    if len(args) >= 1:
        hires_steps = args[0]

    if len(args) >= 2:
        use_old_scheduling = args[1]

    if use_old_scheduling is None:
        return hires_steps, False

    if isinstance(use_old_scheduling, str):
        use_old_scheduling = use_old_scheduling.strip().lower()
        use_old_scheduling = use_old_scheduling not in {"", "0", "false", "no", "off"}
    else:
        use_old_scheduling = bool(use_old_scheduling)

    return hires_steps, use_old_scheduling


def normalize_conditioning_arguments(
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> Tuple[Optional[Any], bool, Tuple[Any, ...], Mapping[str, Any]]:
    """Return normalised hires/use_old flags along with updated args/kwargs."""

    hires_steps, use_old_scheduling = unpack_conditioning_options(args, kwargs)

    normalized_args = list(args)
    normalized_kwargs = dict(kwargs)

    if len(normalized_args) >= 1:
        normalized_args[0] = hires_steps
    elif "hires_steps" in normalized_kwargs:
        normalized_kwargs["hires_steps"] = hires_steps

    if len(normalized_args) >= 2:
        normalized_args[1] = use_old_scheduling
    elif "use_old_scheduling" in normalized_kwargs:
        normalized_kwargs["use_old_scheduling"] = use_old_scheduling

    return hires_steps, use_old_scheduling, tuple(normalized_args), normalized_kwargs


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

