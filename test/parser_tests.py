import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'AUTOMATIC1111', 'stable-diffusion-webui'))

from lib_prompt_fusion.prompt_parser import parse_prompt
from lib_prompt_fusion.interpolation_tensor import InterpolationTensorBuilder
from lib_prompt_fusion.prompt_parser_compat import convert_legacy_schedules, requires_legacy_prompt_parser
from modules import prompt_parser, prompt_parser_old


def run_functional_tests(total_steps=100):
    for i, (given, expected) in enumerate(functional_parse_test_cases):
        expr = parse_prompt(given)
        tensor_builder = InterpolationTensorBuilder()
        expr.extend_tensor(tensor_builder, (0, total_steps), total_steps, dict(), is_hires=False, use_old_scheduling=False)

        actual = tensor_builder.get_prompt_database()

        if type(expected) is set:
            assert set(actual) == expected, f"{actual} != {expected}"
        else:
            assert len(actual) == 1 and actual[0] == expected, f"'{actual[0]}' != '{expected}'"


def test_requires_legacy_parser_for_prompt_fusion_tokens():
    prompts = [
        '[[a:b:1,2]:b:]',
        '[top level:interpolatin:lik a pro:1,3,5:linear]',
        '[a:b:c::mean]',
        '[(nested attention:2.0):abc:,]',
    ]

    for prompt in prompts:
        assert requires_legacy_prompt_parser([prompt]), prompt

    assert not requires_legacy_prompt_parser(['plain prompt'])


def test_prompt_fusion_interpolation_collapses_in_new_parser():
    prompt = '[[a:b:1,2]:b:]'

    schedule = prompt_parser.get_learned_conditioning_prompt_schedules([prompt], 10)[0]
    assert len(schedule) == 1


def test_convert_legacy_schedules_preserves_segments():
    legacy_schedule = [
        [
            prompt_parser_old.ScheduledPromptConditioning(end_at_step=3, cond='cond-a'),
            prompt_parser_old.ScheduledPromptConditioning(end_at_step=10, cond='cond-b'),
        ]
    ]

    converted = convert_legacy_schedules(legacy_schedule)

    assert len(converted) == 1
    assert all(isinstance(s, prompt_parser.ScheduledPromptConditioning) for s in converted[0])
    assert [s.cond for s in converted[0]] == ['cond-a', 'cond-b']
    assert [s.end_at_step for s in converted[0]] == [3, 10]


def test_attention_interpolation_prompts_use_legacy_parser():
    expr = parse_prompt('(cats:0.5, 1.5)')
    tensor_builder = InterpolationTensorBuilder()
    expr.extend_tensor(
        tensor_builder,
        (0, 20),
        20,
        dict(),
        is_hires=False,
        use_old_scheduling=False,
    )

    prompts = tensor_builder.get_prompt_database()
    assert len(prompts) == 1
    prompt = prompts[0]

    assert requires_legacy_prompt_parser([prompt])

    prompt_schedule = prompt_parser_old.get_learned_conditioning_prompt_schedules([prompt], 20)[0]
    assert len(prompt_schedule) > 1

    legacy_schedule = [
        [
            prompt_parser_old.ScheduledPromptConditioning(end_at_step=step, cond=text)
            for step, text in prompt_schedule
        ]
    ]

    converted = convert_legacy_schedules(legacy_schedule)

    assert len(converted[0]) == len(prompt_schedule)
    assert [s.end_at_step for s in converted[0]] == [step for step, _ in prompt_schedule]


functional_parse_test_cases = [
    ('single',)*2,
    ('some space separated text',)*2,
    ('(legacy weighted prompt:-2.1)',)*2,
    ('mixed (legacy weight:3.6) and text',)*2,
    ('legacy [range begin:0] thingy',)*2,
    ('legacy [range end::3] thingy',)*2,
    ('legacy [[nested range::3]:2] thingy',)*2,
    ('legacy [[nested range:2]::3] thingy',)*2,
    ('sugar [range:,abc:3] thingy',)*2,
    ('sugar [[(weight interpolation:0,12):0]::1] thingy', 'sugar [[(weight interpolation:0.0):0]::1] thingy'),
    ('sugar [[(weight interpolation:0,12):0]::2] thingy', 'sugar [[[(weight interpolation:0.0)::1][(weight interpolation:12.0):1]:0]::2] thingy'),
    ('sugar [[(weight interpolation:0,12):0]::3] thingy', 'sugar [[[(weight interpolation:0.0)::1][[(weight interpolation:6.0):1]::2][(weight interpolation:12.0):2]:0]::3] thingy'),
    ('legacy [from:to:2] thingy',)*2,
    ('legacy [negative weight]',)*2,
    ('legacy (positive weight)',)*2,
    ('[abc:1girl:2]',)*2,
    ('[::]',)*2,
    ('[a:b:]',)*2,
    ('[[a:b:1,2]:b:]', {'[a:b:]', '[b:b:]'}),
    ('1girl',)*2,
    ('dashes-in-text',)*2,
    ('text, separated with, comas',)*2,
    ('{prompt}',)*2,
    ('[abc|def ghi|jkl]',)*2,
    ('merging this AND with this',)*2,
    (':',)*2,
    (r'portrait \(object\)',)*2,
    (r'\[escaped square\]',)*2,
    (r'\$var = abc',)*2,
    (r'\\$ arst',)*2,
    (r'$$ arst',)*2,
    ('$var = abc', ''),
    ('$a = prompt value\n$a', 'prompt value'),
    ('$a = prompt value\n$b = $a\n$b', 'prompt value'),
    ('$a = (multiline\nprompt\nvalue:1.0)\n$a', '(multiline prompt value:1.0)'),
    ('$a = ($aa = nested variable\nmultiline\n$aa:1.0)\n$a', '(multiline nested variable:1.0)'),
    ('a [b:c:-1, 10] d', {'a b d', 'a c d'}),
    ('a [b:c:5, 6] d', {'a b d', 'a c d'}),
    ('a [b:c:0.25, 0.5] d', {'a b d', 'a c d'}),
    ('a [b:c:.25, .5] d', {'a b d', 'a c d'}),
    ('a [b:c:,] d', {'a b d', 'a c d'}),
    ('0[1.0:1.1:,]2[3.0:3.1:,]4', {
        '0 1.0 2 3.0 4', '0 1.1 2 3.0 4',
        '0 1.0 2 3.1 4', '0 1.1 2 3.1 4',
    }),
    ('0[1.0:1.1:1.2:,.5,]2[3.0:3.1:,]4', {
        '0 1.0 2 3.0 4', '0 1.0 2 3.1 4',
        '0 1.1 2 3.0 4', '0 1.1 2 3.1 4',
        '0 1.2 2 3.0 4', '0 1.2 2 3.1 4',
    }),
    ('[0.0:0.1:,][1.0:1.1:,][2.0:2.1:,]', {
        '0.0 1.0 2.0', '0.0 1.0 2.1',
        '0.1 1.0 2.0', '0.1 1.0 2.1',
        '0.0 1.1 2.0', '0.0 1.1 2.1',
        '0.1 1.1 2.0', '0.1 1.1 2.1',
    }),
    ('[top level:interpolatin:lik a pro:1,3,5:linear]', {'top level', 'interpolatin', 'lik a pro'}),
    ('[[nested:expr:,]:abc:,]', {'nested', 'expr', 'abc'}),
    ('[(nested attention:2.0):abc:,]', {'(nested attention:2.0)', 'abc'}),
    ('[[nested editing:15]:abc:,]', {'[nested editing:15]', 'abc'}),
    ('[[nested interpolation:abc:,]:12]', {'[nested interpolation:12]', '[abc:12]'}),
    ('[[nested interpolation:abc:,]::7]', {'[nested interpolation::7]', '[abc::7]'}),
    ('$attention = 1.5\n(prompt:$attention)', '(prompt:1.5)'),
    ('$a = 0\n$b = 12\n[[(prompt:$a,$b):0]::2]', '[[[(prompt:0.0)::1][(prompt:12.0):1]:0]::2]'),
    ('$step = 5\n[legacy:editing:$step]', '[legacy:editing:5]'),
    ('$begin = 2\n$end = 7\n[prompt:interpolation:$begin, $end]', {'prompt', 'interpolation'}),
    ('$a($b, $c) = prompt with $b, prompt with $c\n$a(cat, dog)', 'prompt with cat , prompt with dog'),
    ('$a($b) = prompt with $b\n$c($d) = yeay $a($d)\n$c(dog)', 'yeay prompt with dog'),
    ('$a = a lot of animals\n$b($c) = I love $c\n$b($a)', 'I love a lot of animals'),
    ('$a($b) = prompt with $b\n$c($d) = yeay $d\n$a($c(dog))', 'prompt with yeay dog'),
    ('[a|b|c]', '[a|b|c]'),
    ('[a|b|c:]', '[a|b|c]'),
    ('[a|b|c:1]', {'a', 'b', 'c'}),
    ('[a|b|c:2]', {'a', 'b', 'c'}),
    ('[a|b|c:0.5]', {'a', 'b', 'c'}),
    ('[a|b|c:1.1]', {'a', 'b', 'c'}),
    ('[[[Imperial Yellow|Amber]:[Ruby|Plum|Bronze]:9]::39]',)*2,
    ('[a:b:c::mean]', {'a', 'b', 'c'}),
    ('[a:b:c:,,:mean]', {'a', 'b', 'c'}),
    ('[a:b:c: 1, 2, 3:mean]', {'a', 'b', 'c'}),
]


def run_tests():
    run_functional_tests()
    test_requires_legacy_parser_for_prompt_fusion_tokens()
    test_prompt_fusion_interpolation_collapses_in_new_parser()
    test_convert_legacy_schedules_preserves_segments()
    test_attention_interpolation_prompts_use_legacy_parser()
