from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib_prompt_fusion.prompt_parser_compat import (
    normalize_conditioning_arguments,
    unpack_conditioning_options,
)


def test_unpack_defaults_to_false_when_no_args():
    hires_steps, use_old = unpack_conditioning_options((), {})
    assert hires_steps is None
    assert use_old is False


def test_unpack_reads_values_from_kwargs():
    hires_steps, use_old = unpack_conditioning_options((), {"hires_steps": 12, "use_old_scheduling": True})
    assert hires_steps == 12
    assert use_old is True


def test_unpack_merges_args_and_kwargs():
    hires_steps, use_old = unpack_conditioning_options((20,), {"use_old_scheduling": True})
    assert hires_steps == 20
    assert use_old is True


def test_unpack_prefers_positional_over_keyword():
    hires_steps, use_old = unpack_conditioning_options((30, False), {"hires_steps": 99, "use_old_scheduling": True})
    assert hires_steps == 30
    assert use_old is False


def test_unpack_handles_string_flags():
    hires_steps, use_old = unpack_conditioning_options((), {"hires_steps": None, "use_old_scheduling": "false"})
    assert hires_steps is None
    assert use_old is False

    _, use_old_true = unpack_conditioning_options((), {"use_old_scheduling": "yes"})
    assert use_old_true is True


def test_normalize_conditioning_arguments_converts_positional_strings():
    hires_steps, use_old, args, kwargs = normalize_conditioning_arguments((42, "0"), {})
    assert hires_steps == 42
    assert use_old is False
    assert args == (42, False)
    assert kwargs == {}


def test_normalize_conditioning_arguments_updates_keyword_flags():
    hires_steps, use_old, args, kwargs = normalize_conditioning_arguments((), {"use_old_scheduling": "true"})
    assert hires_steps is None
    assert use_old is True
    assert args == ()
    assert kwargs == {"use_old_scheduling": True}


def test_normalize_conditioning_arguments_preserves_other_kwargs():
    _, _, args, kwargs = normalize_conditioning_arguments((10,), {"seed": 123})
    assert args == (10,)
    assert kwargs == {"seed": 123}
