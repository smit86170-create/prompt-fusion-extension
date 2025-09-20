# prompt_parser_patched_superhybrid.py
from __future__ import annotations

import re
from collections import namedtuple
import lark
import random
from functools import lru_cache
from itertools import product
from typing import Sequence, Tuple

import os
import logging
logger = logging.getLogger(__name__)  # не настраиваем basicConfig в библиотеке


# ──────────────────────────────────────────────────────────────────────────────
# Фиче-флаги (переопределяемые через env)
# ──────────────────────────────────────────────────────────────────────────────

def _env_bool(name: str, default: str = "0") -> bool:
    v = str(os.getenv(name, default)).strip().lower()
    return v not in ("0", "", "false", "no", "off")

def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable with a fallback.

    Attempts to convert the environment variable *name* to ``int``. If the
    variable is unset or cannot be parsed as an integer, ``default`` is
    returned and a warning is logged.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid value for %s=%r, using default %d", name, raw, default)
        return default

SAFE_EMPTY = " "

ALLOW_EMPTY_ALTERNATE          = _env_bool("ALLOW_EMPTY_ALTERNATE", "1")
EXPAND_ALTERNATE_PER_STEP      = _env_bool("EXPAND_ALTERNATE_PER_STEP", "1")
GROUP_COMBO_LIMIT              = _env_int("GROUP_COMBO_LIMIT", 100)
DEDUP_SCHEDULE_STEPS           = _env_bool("DEDUP_SCHEDULE_STEPS", "0")
GROUP_COMBO_FALLBACK           = os.getenv("GROUP_COMBO_FALLBACK", "truncate").strip().lower()  # "truncate"|"literal"|"sample"
SUPPRESS_STANDALONE_COLON      = _env_bool("SUPPRESS_STANDALONE_COLON", "1")
CACHE_SIZE                     = _env_int("PROMPT_PARSER_CACHE_SIZE", 4096)


# ──────────────────────────────────────────────────────────────────────────────
# Общие числовые шаблоны / регэкспы (чтобы не дублировать)
# ──────────────────────────────────────────────────────────────────────────────
NUMERIC_RE = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
# «Без знака» — удобно для шаблонов вида [+|-]<число>
NUMERIC_NOSIGN_RE = r"(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
RE_NUMERIC = re.compile(NUMERIC_RE)
RE_NUMERIC_FULL = re.compile(rf"^(?:{NUMERIC_RE})$")

# ──────────────────────────────────────────────────────────────────────────────
# Вспомогалки по тексту/пробелам для единообразия вывода
# ──────────────────────────────────────────────────────────────────────────────

_re_ws_collapse = re.compile(r"[ \t\r\n]+")
_re_unescape_literals = re.compile(r"\\([:\[\]\(\)\{\}\|!\\])")

@lru_cache(maxsize=CACHE_SIZE)
def _collapse_spaces(s: str, keep_edges: bool = False) -> str:
    """Сжать повторяющиеся пробелы/переводы строк в один пробел.

    Если ``keep_edges`` установлен в ``True``, ведущие и хвостовые пробелы сохраняются,
    иначе итоговая строка обрезается по краям.
    """
    collapsed = _re_ws_collapse.sub(" ", s)
    if keep_edges:
        return collapsed
    return collapsed.strip()

def _unescape_literals(s: str) -> str:
    """Разэкранировать часто встречающиеся литералы в промптах."""
    if not s:
        return s
    return _re_unescape_literals.sub(r"\1", s)

def _strip_outer_parens_once(s: str) -> str:
    """Снять ОДИН раз внешние круглые скобки, если они действительно обрамляют весь текст."""
    if not s:
        return s
    t = s.strip()
    if len(t) >= 2 and t[0] == "(" and t[-1] == ")":
        # быстрая проверка баланса на одном уровне
        depth = 0
        for i, ch in enumerate(t):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(t) - 1:
                    # есть ещё символы после закрытия — это не внешняя пара
                    return s
        if depth == 0:
            return t[1:-1].strip()
    return s

def _norm_join(*parts: str) -> str:
    """Аккуратно склеить префикс/контент/суффикс → одинарные пробелы, края подрезать."""
    return _collapse_spaces("".join(parts))

def _norm_join_keep_edges(*parts: str) -> str:
    """Как _norm_join, но без .strip() по краям (используется локально при сборке)."""
    s = _re_ws_collapse.sub(" ", "".join(parts))
    # Collapses whitespace without trimming leading or trailing spaces.
    return s


# ──────────────────────────────────────────────────────────────────────────────
# Склейка префикса/ядра/суффикса с корректной обработкой пробелов
# ──────────────────────────────────────────────────────────────────────────────
def _smart_space_trim(s: str) -> str:
    return _re_ws_collapse.sub(" ", s)


def _concat_prefix_text_suffix(prefix: str, text: str, suffix: str) -> str:
    """
    Склейка префикса/ядра/суффикса c бережной обработкой пробелов:
    - Если text пустой, сохраняем крайние пробелы и гарантируем ОДИН пробел
      между непустыми prefix/suffix (если они соприкасаются без него).
    - Ничего не .strip()-аем по краям (важно для кейсов с хвостовым пробелом: 'a ').
    """
    # Вспомогалка: нужен ли принудительный пробел между куском a и b
    def _need_space(a: str, b: str) -> bool:
        return bool(a) and bool(b) and (not a[-1].isspace()) and (not b[0].isspace()) and a[-1].isalnum() and b[0].isalnum()

    # Пустая центральная часть — склеиваем prefix/suffix и гарантируем один пробел, если они «слипаются»
    if text.strip() == "":
        if _need_space(prefix, suffix):
            return _smart_space_trim(prefix + " " + suffix)
        return _smart_space_trim(prefix + suffix)

    # Непустая центральная часть — обеспечиваем разделители по краям ядра
    left = prefix
    mid  = text
    right = suffix
    if _need_space(left, mid):
        left += " "
    if _need_space(mid, right):
        mid += " "
    return _smart_space_trim(left + mid + right)


# ──────────────────────────────────────────────────────────────────────────────
# Предкомпилированные regex'ы для fast-path
# ──────────────────────────────────────────────────────────────────────────────

# [ ... ] : N [reverse|r]  с префиксом/суффиксом
RE_BRACKET_AFTER = re.compile(
    rf'(?s)^(.*)\[(.*?)\]\s*:\s*({NUMERIC_RE})\s*(?:(?P<rev>r|reverse)\b)?\s*(?P<post>.*)$'
)
# Явные диапазоны: "[a:b]:10 1-4,5-7 [reverse]"
RE_BRACKET_AFTER_WITH_RANGES = re.compile(rf'(?s)^\s*\[(?P<inner>.*?)\]\s*:\s*(?P<num>{NUMERIC_RE})\s+'
    r'(?P<ranges>(?:\d+%?\s*-\s*\d+%?(?:\s*,\s*)?)+)'
    r'(?:\s*(?P<rev>r|reverse))?\s*$'
)

# Общий префикс для reverse/r в постфиксе
RE_REVERSE_PREFIX = re.compile(r'^\s*(?:r|reverse)\b\s*')

# ──────────────────────────────────────────────────────────────────────────────
# Безопасный сплит по ':' только на верхнем уровне
# ──────────────────────────────────────────────────────────────────────────────

def _split_top_level_colon_all(s: str, keep_empty: bool) -> list[str]:
    """
    Разбить строку по ':' только на ВЕРХНЕМ уровне:
    игнорировать ':' внутри круглых (), фигурных {}, а ТАКЖЕ квадратных [].
    Учитывать экранирование '\\'.
    """
    parts: list[str] = []
    buf: list[str] = []
    depth_paren = 0  # (...)
    depth_brace = 0  # {...}
    depth_brack = 0  # [...]
    i = 0
    while i < len(s):
        ch = s[i]
        # экранирование
        if ch == '\\':
            if i + 1 < len(s):
                buf.append(ch); buf.append(s[i + 1])
                i += 2
                continue

        # уровни скобок
        if ch == '(':
            depth_paren += 1
        elif ch == ')':
            depth_paren = max(0, depth_paren - 1)
        elif ch == '{':
            depth_brace += 1
        elif ch == '}':
            depth_brace = max(0, depth_brace - 1)
        elif ch == '[':
            depth_brack += 1
        elif ch == ']':
            depth_brack = max(0, depth_brack - 1)

        # разрыв только на самом верхнем уровне
        if ch == ':' and depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            seg = "".join(buf)
            seg_trim = seg.strip()
            if keep_empty:
                parts.append(seg_trim if seg_trim != "" else "")
            else:
                if seg_trim != "":
                    parts.append(seg_trim)
            buf = []
        else:
            buf.append(ch)
        i += 1

    seg = "".join(buf)
    seg_trim = seg.strip()
    if keep_empty:
        parts.append(seg_trim if seg_trim != "" else "")
    else:
        if seg_trim != "":
            parts.append(seg_trim)
    return parts


def _split_top_level_colon(s: str) -> list[str]:
    return _split_top_level_colon_all(s, keep_empty=False)

def _split_top_level_colon_keep_empty(s: str) -> list[str]:
    return _split_top_level_colon_all(s, keep_empty=True)

# ──────────────────────────────────────────────────────────────────────────────
# Грамматика Lark
# ──────────────────────────────────────────────────────────────────────────────

# динамическое правило для alternate
_alt_rule = r' "[" prompt ("|" prompt)* "]" '                 # без пустых опций
if ALLOW_EMPTY_ALTERNATE:
    # разрешить пустые опции
    _alt_rule = r' "[" prompt ("|" [prompt])+ "]" '

# NB: расширили класс одиночных знаков в start, чтобы '!' не валил парсер для plain'ов/compound'ов
_grammar = r"""
!start: (prompt | /[][():,!]/+)*

prompt: (scheduled | emphasized | grouped
        | alternate | alternate_distinct
        | alternate1 | alternate2
        | top_level_sequence3 | top_level_sequence | sequence
        | compound | weighted | numbered | and_rule
        | plain | WHITESPACE)*

!emphasized: "(" prompt ")"
           | "(" prompt ":" prompt ")"
           | "(" prompt ":" NUMBER ")"


!weighted: (plain | compound) ":" NUMBER

scheduled: "[" [prompt (":" prompt)*] "]" ":" NUMBER (WHITESPACE* step_range_list)? (WHITESPACE* reverse_flag)?
reverse_flag: "reverse" | "r"
step_range_list: step_range (WHITESPACE* "," WHITESPACE* step_range)*
step_range: NUMBER "-" NUMBER        -> range_abs
          | NUMBER "%" "-" NUMBER "%" -> range_pct



alternate: """ + _alt_rule + r"""
!alternate_distinct: "[" prompt ("|" prompt)* "]!"
alternate1: (prompt) "|" (prompt)+
alternate2: (plain | compound) ("|" (plain | compound))+

grouped: "{" ((NUMBER_Q | prompt | sequence | grouped) ("," | "|")?)+ "}"

top_level_sequence: prompt ("::" sequence)+ "!!" ("," plain)?
top_level_sequence3: prompt ":::" (sequence | plain) (WHITESPACE* "," WHITESPACE* (sequence | plain))* "!!!" (WHITESPACE* "," WHITESPACE* (plain | sequence))*
sequence: prompt "::" prompt ("," | WHITESPACE)* nested_sequence* ("!" | ";")?
nested_sequence: "::" prompt ("," | WHITESPACE)* ("!" | ";" | "~")?

compound: /[a-zA-Z0-9]+(_[a-zA-Z0-9]+)+/
numbered: NUMBER_Q ("!" | "_")? (grouped | sequence | compound | and_rule | plain | alternate | alternate_distinct | alternate2 | alternate1)

and_rule: (plain | compound | weighted | emphasized) ("&" (plain | compound | weighted | emphasized))+
 WHITESPACE: /\s+/
plain: /([^\\[\]\{\}\(\),&:!|]|\\.)+/

%import common.SIGNED_NUMBER -> NUMBER
%import common.INT -> NUMBER_Q
"""

schedule_parser = lark.Lark(_grammar, start="start")

# ──────────────────────────────────────────────────────────────────────────────
# Общие вспомогалки для fast-path "[...]:N" (и CollectSteps, и get_schedule)
# ──────────────────────────────────────────────────────────────────────────────

def _clamp(x: int, steps: int) -> int:
    return max(1, min(int(x), steps))

def _to_end_step(num: float, steps: int) -> int:
    """num ≤ 1.0 — это доля, > 1.0 — абсолютный шаг."""
    if num <= 1.0:
        return _clamp(round(num * steps), steps)
    return _clamp(round(num), steps)

def _build_bracket_after_schedules(
    pre: str, prompts: list[str], boundary_end: int, post: str, steps: int
) -> list[list[int, str]]:
    """
    Семантика:
      • если НЕТ префикса/суффикса → как для [p1:p2:...:N] (интервалы ВНУТРИ [1..N]);
      • если ЕСТЬ префикс/суффикс → «после N»: до N — пролог (pre+post), далее хвост.
    """
    if not (pre.strip() or post.strip()):
        return _build_bracket_inner_schedules(pre, prompts, boundary_end, post, steps)

    schedules: list[list[int, str]] = []
    boundary_end = int(boundary_end)
    steps = int(steps)

    tail = max(0, steps - boundary_end)

    # Пролог до N (если есть хвост)
    if tail > 0:
        schedules.append([boundary_end, _concat_prefix_text_suffix(pre, "", post)])

    if not prompts:
        schedules.append([steps, _concat_prefix_text_suffix(pre, "", post)])
        return schedules

    if tail <= 0:
        schedules.append([steps, _concat_prefix_text_suffix(pre, prompts[-1], post)])
    else:
        # Строго монотонные границы по floor, без round-слипания
        k = max(1, len(prompts))
        prev_end = boundary_end
        for i, p in enumerate(prompts, start=1):
            end = boundary_end + (tail * i) // k  # floor
            if end <= prev_end:
                end = min(steps, prev_end + 1)
            schedules.append([end, _concat_prefix_text_suffix(pre, p, post)])
            prev_end = end

    # Схлопнуть повторы (end,text)
    out: list[list[int, str]] = []
    for e, t in schedules:
        if out and out[-1][0] == e and out[-1][1] == t:
            continue
        out.append([int(e), t])
    return out


def _build_bracket_inner_schedules(
    pre: str, prompts: list[str], boundary_end: int, post: str, steps: int
) -> list[list[int, str]]:
    """
    A [p1:p2:...:N] Z — интервалы внутри [1..N] + хвост.
    Особый случай: при ОДНОМ prompt до границы показываем только pre+post,
    а после границы — pre+prompt+post (ожидание тестов).
    Для >=2 prompt'ов интервалы совпадают с логикой CollectSteps.visit_scheduled.
    """
    schedules: list[list[int, str]] = []

    if not prompts:
        return [[steps, _concat_prefix_text_suffix(pre, "", post)]]

    # 1 prompt — до boundary пустая центральная часть, после — сам prompt
    if len(prompts) == 1:
        schedules.append([int(boundary_end), _concat_prefix_text_suffix(pre, "", post)])
        if boundary_end < steps:
            schedules.append([int(steps), _concat_prefix_text_suffix(pre, prompts[0], post)])
        return schedules

    # >=2 prompt — равномерно делим диапазон [1..boundary_end]
    num_prompts = len(prompts)
    step_size = float(boundary_end) / num_prompts

    intervals: list[tuple[int, int]] = []
    for i in range(num_prompts):
        start = _clamp(int(round(i * step_size)) + 1, steps)
        end = _clamp(int(round((i + 1) * step_size)), steps)
        # допускаем одношаговые интервалы
        if start <= end:
            intervals.append((start, end))

    if intervals:
        # Пролог ДО первого интервала добавляем ТОЛЬКО если есть префикс/суффикс.
        # Для «чистых» форм ([...]:N) тесты ожидают начинать сразу с первого интервала.
        if (pre.strip() or post.strip()):
            pre_end = max(1, intervals[0][0] - 1)
            schedules.append([pre_end, _concat_prefix_text_suffix(pre, "", post)])

    for i, (_st, en) in enumerate(intervals[:num_prompts]):
        schedules.append([en, _concat_prefix_text_suffix(pre, prompts[i], post)])

    if intervals and intervals[-1][1] < steps:
        schedules.append([steps, _concat_prefix_text_suffix(pre, prompts[-1], post)])

    # Fallback: если из-за округлений интервалов не получилось вовсе
    if not intervals:
        if (pre.strip() or post.strip()):
            schedules.append([int(boundary_end), _concat_prefix_text_suffix(pre, "", post)])
        schedules.append([int(steps), _concat_prefix_text_suffix(pre, prompts[-1], post)])



    # Схлопнем повторы по (end_step, text)
    out = []
    for e, t in schedules:
        if out and out[-1][0] == e and out[-1][1] == t:
            continue
        out.append([e, t])

    return out



# ──────────────────────────────────────────────────────────────────────────────
# Унификация resolve_tree
# ──────────────────────────────────────────────────────────────────────────────

def resolve_tree(tree: lark.Tree | lark.Token, keep_spacing: bool = True) -> str:
    if isinstance(tree, lark.Tree):
        parts = []
        for child in tree.children:
            if isinstance(child, lark.Token) and child.type == "WHITESPACE":
                parts.append(" ")
                continue
            parts.append(resolve_tree(child, keep_spacing))
        result = "".join(str(c) for c in parts if c is not None)
        if keep_spacing:
            # схлопываем все пробельные последовательности до одиночного пробела,
            # но сохраняем крайние пробелы, если они были
            leading = len(result) - len(result.lstrip())
            trailing = len(result) - len(result.rstrip())
            core = _re_ws_collapse.sub(" ", result.strip())
            return (" " * leading) + core + (" " * trailing)
        else:
            return _re_ws_collapse.sub("", result)
    return str(tree)


# ──────────────────────────────────────────────────────────────────────────────
# Комплексность/перевод узлов → текст
# ──────────────────────────────────────────────────────────────────────────────
_RE_UNESCAPED_ALT_OR_BANG = re.compile(r'(?<!\\)[|!]')

def _needs_complex_parse(inner: str, full: str) -> bool:
    """
    True, если fast-path лучше отключить и отдать парсеру/Visitor:
      - вложенные квадратные скобки внутри inner,
      - НЕэкранированные '|' или '!' в полной строке,
      - TL-последовательности с '::'.
    """
    if "[" in inner or "]" in inner:
        return True
    if _RE_UNESCAPED_ALT_OR_BANG.search(full or ""):
        return True
    # '::' считаем «сложностью» только ВНЕ квадратных скобок fast-path'а
    # (для случая "[...:N]" внутри [] это нормальный разделитель).
    if "::" in (full or "") and "::" not in (inner or ""):
        return True
    return False

def _to_text(x) -> str:
    """Единый путь узел/строку → финальный текст, как у Visitor."""
    s = resolve_tree(x, keep_spacing=True) if not isinstance(x, str) else x
    return _unescape_literals(s)

# ──────────────────────────────────────────────────────────────────────────────
# Transformer: преобразование дерева в текст (для TL последовательностей и т.п.)
# ──────────────────────────────────────────────────────────────────────────────

class ScheduleTransformer(lark.Transformer):
    def __init__(self, total_steps: int, current_step: int = 1, seed: int | None = 42):
        super().__init__()
        self.total_steps = total_steps
        self.current_step = current_step
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else random

    def start(self, args):
        s = "".join(str(arg) for arg in args if arg)
        # owner::a::b!!, extra -> owner -> owner: a, b, extra
        if "::" in s and "!!" in s and all(ch not in s for ch in "[]()"):
            left, trailing = s.split("!!", 1)
            owner, rest = left.split("::", 1)
            descriptors = [x.strip(" ,~!;") for x in rest.split("::") if x.strip(" ,~!;")]
            seq_text = f"{owner.strip()}: {', '.join(descriptors)}" if descriptors else owner.strip()
            trailing_text = [t.strip(" ,") for t in trailing.split(",") if t.strip(" ,")]
            out = f"{owner.strip()} -> {seq_text}"
            if trailing_text:
                out += f", {', '.join(trailing_text)}"
            return out
        # owner::a::b!  ->  owner: a, b
        if "::" in s and (s.endswith("!") or s.endswith(";")) and all(ch not in s for ch in "[]()"):
            owner, rest = s.split("::", 1)
            rest = rest[:-1]
            descriptors = [x.strip(" ,~!;") for x in rest.split("::") if x.strip(" ,~!;")]
            return f"{owner.strip()}: {', '.join(descriptors)}"
        return s

    def prompt(self, args):
        return "".join(str(arg) for arg in args if arg)

    def plain(self, args):
        return args[0].value

    def compound(self, args):
        return "_".join(str(arg) for arg in args)

    def and_rule(self, args):
        import lark as _l
        parts = []
        for child in args:
            if isinstance(child, _l.Tree) and getattr(child, "data", None) in ("weighted", "emphasized"):
                parts.append(self.transform(child))  # получим '(token:1.2)'
            else:
                s = resolve_tree(child, keep_spacing=True)
                if s:
                    parts.append(s)
        return " and ".join(p for p in parts if p)

    def weighted(self, args):
        """
        Преобразуем 'token:NUMBER' -> '(token:NUMBER)' с тем же форматированием, как в emphasized.
        """
        import lark as _l
        # левая часть (plain|compound)
        left = resolve_tree(args[0], keep_spacing=True).strip()

        # правая часть — число; достаём как текст с подстраховкой
        tail_txt = "".join(
            (a.value if isinstance(a, _l.Token) else resolve_tree(a, keep_spacing=True))
            for a in args[1:]
        )
        mnum = RE_NUMERIC.search(tail_txt or "")
        if mnum:
            num_txt = mnum.group(0)
            if "." not in num_txt and "e" not in num_txt.lower():
                try:
                    weight_str = f"{float(num_txt):.1f}"
                except Exception:
                    weight_str = "1.1"
            else:
                weight_str = num_txt
        else:
            weight_str = "1.1"

        return f"({left}:{weight_str})"

    def grouped(self, args):
        return ", ".join(resolve_tree(arg, keep_spacing=True) for arg in args if resolve_tree(arg).strip(" ,|"))

    def alternate(self, args):
        # Учитываем ПУСТЫЕ опции между '|' (например, "[a||c]")
        vals = []
        last_was_sep = True  # стартуем так, чтобы ведущий '|' дал пустую опцию

        for arg in args:
            s = resolve_tree(arg, keep_spacing=True)
            tok = (s or "").strip()
            if tok in ("|", ",", "[", "]", "!"):
                if last_was_sep:
                    vals.append("")  # пустая опция между двумя разделителями
                last_was_sep = True
                continue
            vals.append(s)
            last_was_sep = False

        if last_was_sep:
            vals.append("")  # хвостовой разделитель -> пустая опция

        return vals[(self.current_step - 1) % len(vals)] if vals else "empty_prompt"


    def alternate_distinct(self, args):
        options = []
        for arg in args:
            s = resolve_tree(arg, keep_spacing=True)
            # фильтруем разделители
            if s.strip() in ("|", ","):
                continue
            if s:
                options.append(s)
            else:
                options.append("")  # пустая опция допустима при ALLOW_EMPTY_ALTERNATE
        return self.rng.choice(options) if options else "empty_prompt"

    def alternate1(self, args):
        # "a | b" — поддержка простых разделителей с возможными пустыми вариантами
        options = []
        last_was_sep = True

        for arg in args:
            s = resolve_tree(arg, keep_spacing=True)
            tok = (s or "").strip()
            if tok in ("|", ","):
                if last_was_sep:
                    options.append("")  # пустая альтернатива
                last_was_sep = True
                continue
            options.append(s)
            last_was_sep = False

        if last_was_sep:
            options.append("")

        return self.rng.choice(options) if options else "empty_prompt"


    def alternate2(self, args):
        options = [resolve_tree(a, keep_spacing=True).strip() for a in args if resolve_tree(a, keep_spacing=True).strip()]
        suffix = options[0].split("_", 1)[1] if options and "_" in options[0] else ""
        combined = [(o if "_" in o or not suffix else f"{o}_{suffix}") for o in options]
        return "|".join(combined) if combined else "empty_prompt"

    def numbered(self, args):
        quantity = int(args[0])
        distinct = False
        if len(args) > 1:
            mark = str(args[1])
            distinct = mark in ("!", "_")
        # В Transformer дети уже преобразованы → приводим к строке.
        target = args[-1]
        target_str = "" if target is None else str(target)
        # Допускаем, что источником был alternate; если пришла единичная строка — берём её как единственный вариант.
        options = [s for s in (p.strip() for p in target_str.split("|")) if s] or ([target_str] if target_str != "" else [])

        if not options:
            return "empty_prompt"

        if distinct:
            seen = []
            for opt in options:
                if opt not in seen:
                    seen.append(opt)
                if len(seen) >= quantity:
                    break
            selected = seen if len(seen) >= quantity else seen + options[:max(0, quantity - len(seen))]
        else:
            selected = self.rng.choices(options, k=quantity)

        return ", ".join(selected)

    def sequence(self, args, parent=None):
        owner = resolve_tree(args[0], keep_spacing=True) if parent is None else parent
        descriptors = [resolve_tree(arg, keep_spacing=True).strip(" ,~!;") for arg in args[1:] if resolve_tree(arg, keep_spacing=True).strip(" ,~!;")]
        return f"{owner}: {', '.join(descriptors)}"

    def top_level_sequence(self, args):
        owner = resolve_tree(args[0], keep_spacing=True).strip()
        sequences = []
        trailing_text = []
        for child in args[1:]:
            s = child if isinstance(child, str) else resolve_tree(child, keep_spacing=True)
            if not s:
                continue
            s = s.strip()
            if ':' in s:
                head, rhs = s.split(':', 1)
                parts = [head.strip()] + [p.strip() for p in rhs.split(',') if p.strip()]
                for part in parts:
                    if part and part != owner:
                        sequences.append(f"{owner}: {part}")
            else:
                t = s.strip(' ,')
                if t:
                    trailing_text.append(t)
        text = f"{owner} -> {', '.join(sequences)}"
        if trailing_text:
            text += f", {', '.join(trailing_text)}"
        return text

    def top_level_sequence3(self, args):
        owner = resolve_tree(args[0], keep_spacing=True).strip()
        before = []
        after = []
        after_bang = False

        for child in args[1:]:
            # Токен-граница "!!!"
            if isinstance(child, lark.Token):
                if child.type == "WHITESPACE":
                    continue
                if str(child) == "!!!":
                    after_bang = True
                    continue
                if str(child) == ",":
                    continue

            if not after_bang:
                # Левая часть: "owner::: (sequence|plain) (, ...)*"
                if isinstance(child, lark.Tree) and child.data == "sequence":
                    # Принудительно используем owner в левой части
                    txt = self.sequence(child.children, parent=owner)
                    if txt:
                        before.append(txt)
                else:
                    s = resolve_tree(child, keep_spacing=True).strip(" ,")
                    if s:
                        before.append(f"{owner}: {s}")
            else:
                # Правая часть после "!!!": сохраняем собственных владельцев
                if isinstance(child, lark.Tree) and child.data == "sequence":
                    txt = self.sequence(child.children)  # owner берём из самого sequence
                    if txt:
                        after.append(txt)
                else:
                    s = resolve_tree(child, keep_spacing=True).strip(" ,")
                    if s:
                        after.append(s)

        text = f"{owner} -> {', '.join(before)}"
        if after:
            text += f", {', '.join(after)}"
        return text

        
    def nested_sequence(self, args):
        def _is_term(x): return isinstance(x, str) and x in ('!', ';', '~')
        has_term = bool(args and _is_term(args[-1]))
        payload = args[:-1] if has_term else args
        elements = [resolve_tree(arg, keep_spacing=True).strip(" ,~!;")
                    for arg in payload
                    if resolve_tree(arg, keep_spacing=True).strip(" ,~!;")]
        terminator = args[-1] if has_term else None
        if terminator == "~":
            return self.rng.choice(elements) if elements else "empty_prompt"
        return ", ".join(elements)

    def emphasized(self, args):
        """
        (cat) -> (cat:1.1)
        (cat:2) -> (cat:2.0)
        (cat: dog) -> (cat:1.1)
        ((bird:2):3) -> содержит "(bird:2.0)"
        (  fox  :  1.25 ) -> (fox:1.25)
        (wolf : (2)) -> (wolf:2.0)

        Доп. правило: если внутри уже "(...:w)" и рассчитанный внешний вес == 1.1,
        не добавляем второй слой — возвращаем как есть.
        """
        prompt_text = ""
        weight_str: str | None = None

        # Собираем сырой текст аргументов
        raw_parts: list[str] = []
        for a in args:
            if isinstance(a, lark.Token):
                raw_parts.append(a.value)
            elif isinstance(a, lark.Tree):
                raw_parts.append(resolve_tree(a, keep_spacing=True))
            else:
                raw_parts.append(str(a))

        # Уберём служебные токены '(', ')', ':' — оставим полезные куски
        parts = [p.strip() for p in raw_parts if p not in (":", "(", ")", "", None)]
        if parts:
            prompt_text = parts[0].strip()

        if len(parts) >= 2:
            mnum = RE_NUMERIC.search(parts[1])
            if mnum:
                num_txt = mnum.group(0)
                if "." not in num_txt and "e" not in num_txt.lower():
                    try:
                        weight_str = f"{float(num_txt):.1f}"
                    except Exception:
                        weight_str = "1.1"
                else:
                    weight_str = num_txt
            else:
                weight_str = "1.1"
        else:
            weight_str = "1.1"

        # ★ Если уже имеем "(...:w)" как текст и внешний вес дефолтный — не оборачиваем повторно
        #   Это устраняет артефакт вида "((cat:1.2):1.1)" в Visitor.
        # Внутри ScheduleTransformer.emphasized, перед return: 
        pt = prompt_text.strip()
        # (существующее правило «не оборачивать второй раз при 1.1» — оставить)
        if (
            weight_str in ("1.1", "1.10")
            and len(pt) >= 5 and pt[0] == "(" and pt[-1] == ")" and ":" in pt
        ):
            return pt

        # ★ добавка: единообразно разэкранируем литералы в prompt_text
        prompt_text = _unescape_literals(prompt_text)

        return f"({prompt_text}:{weight_str})"




# ──────────────────────────────────────────────────────────────────────────────
# Visitor: сборка расписаний
# ──────────────────────────────────────────────────────────────────────────────

class CollectSteps(lark.Visitor):
    def __init__(self, steps, prefix="", suffix="", use_scheduling=True, seed=None):
        super().__init__()
        self.steps = steps
        self.prefix = prefix
        self.suffix = suffix
        self.use_scheduling = use_scheduling
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else random
        self.schedules = []

    # — fast-path для "[...]:N [reverse]" на уровне prompt
    def visit_prompt(self, tree):
        full = resolve_tree(tree, keep_spacing=True)
        # (A) Диапазоны после числа: "[a:b]:10 1-4,6-8 [reverse]"
        m_rng = RE_BRACKET_AFTER_WITH_RANGES.match(full)
        if m_rng and '|' not in full:
            inner = m_rng.group("inner")
            ranges_txt = m_rng.group("ranges")
            rev = m_rng.group("rev")
            def _to_steps_local(txt: str) -> int:
                s = txt.strip()
                if s.endswith('%'):
                    try:
                        return max(1, min(int(round(float(s[:-1]) / 100.0 * self.steps)), self.steps))
                    except Exception:
                        return 1
                try:
                    return max(1, min(int(round(float(s))), self.steps))
                except Exception:
                    return 1
            prompts = [_unescape_literals(p.strip()) for p in _split_top_level_colon_keep_empty(inner)]
            # reverse только для «чистых» форм
            if rev and not (self.prefix.strip() or self.suffix.strip()):
                prompts = list(reversed(prompts))
            ranges = []
            first_is_percent = None
            for part in ranges_txt.split(','):
                if '-' in part:
                    a, b = part.split('-', 1)
                    if first_is_percent is None:
                        first_is_percent = a.strip().endswith('%')
                    ra, rb = _to_steps_local(a), _to_steps_local(b)
                    if ra < rb:
                        ranges.append((ra, rb))
            schedules = []
            if ranges:
                pre_end = max(1, ranges[0][0] - 1)
                if first_is_percent:
                    if ranges[0][0] > 1:
                        schedules.append([pre_end, _concat_prefix_text_suffix(self.prefix, "", self.suffix)])
                else:
                    schedules.append([max(1, pre_end), _concat_prefix_text_suffix(self.prefix, "", self.suffix)])

            for i, (_st, en) in enumerate(ranges[:len(prompts)]):
                schedules.append([min(en, self.steps), _concat_prefix_text_suffix(self.prefix, prompts[i], self.suffix)])
            if ranges and ranges[-1][1] < self.steps and prompts:
                schedules.append([self.steps, _concat_prefix_text_suffix(self.prefix, prompts[-1], self.suffix)])
            return (
                [[e, _apply_and(_collapse_spaces(t, keep_edges=True))] for e, t in schedules]
                or [[self.steps, _apply_and(_collapse_spaces(_unescape_literals(inner), keep_edges=True))]]
            )

        # (B) «Чистый» "[...]:N [reverse]" без диапазонов — только если ровно одна пара []
        if full.count('[') == 1 and full.count(']') == 1 and not _RE_UNESCAPED_ALT_OR_BANG.search(full):
            m = RE_BRACKET_AFTER.match(full)
            if m:
                pre, inner, boundary_txt, rev_flag, post = m.groups()
                # NEW: сложный inner → отдаём грамматике
                if _needs_complex_parse(inner or "", full):
                    return self._default_visit(tree)
                # Если после числа идут диапазоны (1-4, 10%-40% и т.п.) — отдаём грамматике
                if re.search(r'\b\d+%?\s*-\s*\d+%?', post or ""):
                    return self._default_visit(tree)
                if '(' in inner or ')' in inner:
                    # даже если в inner есть скобки, мы теперь безопасно сплитим по верхнему уровню
                    pass
                try:
                    boundary_f = float(boundary_txt)
                    prompts = [_unescape_literals(p.strip()) for p in _split_top_level_colon_keep_empty(inner)]
                    # Реверс — ТОЛЬКО для «чистого» вида (без префикса/суффикса)
                    if rev_flag and not ((pre or "").strip() or (post or "").strip()):
                        prompts = list(reversed(prompts))
                    boundary = _to_end_step(boundary_f, self.steps)
                    schedules = _build_bracket_after_schedules(
                        self.prefix + pre, prompts, boundary, post + self.suffix, self.steps
                    )
                    return [[e, _apply_and(_collapse_spaces(t, keep_edges=True))] for e, t in schedules]
                except ValueError:
                    pass
        return self._default_visit(tree)


    def visit(self, tree):
        if isinstance(tree, lark.Tree):
            method_name = f"visit_{tree.data}"
            method = getattr(self, method_name, self._default_visit)
            return method(tree)
        elif isinstance(tree, lark.Token):
            return self._visit_token(tree)
        return []

    def visit_start(self, tree):
        full = resolve_tree(tree, keep_spacing=True)
        # ── FAST-PATH (1): "[a:b] : N RANGES [reverse]" — ОБРАБАТЫВАЕМ ПЕРВЫМ ──
        m_rng = RE_BRACKET_AFTER_WITH_RANGES.match(full)
        if m_rng and '|' not in full:
            inner = m_rng.group("inner")
            ranges_txt = m_rng.group("ranges")
            rev = m_rng.group("rev")

            def _to_steps_local(txt: str) -> int:
                s = txt.strip()
                if s.endswith('%'):
                    try:
                        return max(1, min(int(round(float(s[:-1]) / 100.0 * self.steps)), self.steps))
                    except Exception:
                        return 1
                try:
                    return max(1, min(int(round(float(s))), self.steps))
                except Exception:
                    return 1

            prompts = [_unescape_literals(p.strip()) for p in _split_top_level_colon_keep_empty(inner)]
            if rev and not (self.prefix.strip() or self.suffix.strip()):
                prompts = list(reversed(prompts))

            ranges = []
            first_is_percent = None
            for part in ranges_txt.split(','):
                if '-' in part:
                    a, b = part.split('-', 1)
                    if first_is_percent is None:
                        first_is_percent = a.strip().endswith('%')
                    ra, rb = _to_steps_local(a), _to_steps_local(b)
                    if ra < rb:
                        ranges.append((ra, rb))

            schedules = []
            if ranges:
                pre_end = max(1, ranges[0][0] - 1)
                if first_is_percent:
                    if ranges[0][0] > 1:
                        schedules.append([pre_end, _concat_prefix_text_suffix(self.prefix, "", self.suffix)])
                else:
                    schedules.append([max(1, pre_end), _concat_prefix_text_suffix(self.prefix, "", self.suffix)])

            for i, (_st, en) in enumerate(ranges[:len(prompts)]):
                schedules.append([min(en, self.steps), _concat_prefix_text_suffix(self.prefix, prompts[i], self.suffix)])
            if ranges and ranges[-1][1] < self.steps and prompts:
                schedules.append([self.steps, _concat_prefix_text_suffix(self.prefix, prompts[-1], self.suffix)])

            return (
                [[e, _apply_and(_collapse_spaces(t, keep_edges=True))] for e, t in schedules]
                or [[self.steps, _apply_and(_collapse_spaces(_unescape_literals(inner), keep_edges=True))]]
            )

        # ── FAST-PATH (2): "pre [a:b:c] : N [reverse] post" (без диапазонов) ──
        m_ba = RE_BRACKET_AFTER.match(full)
        if m_ba and not _RE_UNESCAPED_ALT_OR_BANG.search(full) and full.count('[') == 1 and full.count(']') == 1:
            pre, inner, boundary_txt, rev_flag, post = m_ba.groups()
            # NEW: сложный кейс — отдать визитору (единая семантика)
            if _needs_complex_parse(inner or "", full):
                return self._default_visit(tree)
            try:
                boundary_f = float(boundary_txt)
            except Exception:
                boundary_f = 1.0

            # Если после числа есть диапазоны — не делаем fast-path, а отдаём грамматике
            import re as _re
            if _re.search(r'\b\d+%?\s*-\s*\d+%?', post or ""):
                return self._default_visit(tree)

            # Если regex «проглотил» reverse в post — вытащим его префиксно
            if not rev_flag and post:
                mrev = RE_REVERSE_PREFIX.match(post or "")
                if mrev:
                    rev_flag = 'reverse'
                    post = (post or "")[mrev.end():]

            prompts = [_unescape_literals(p.strip()) for p in _split_top_level_colon_keep_empty(inner)]
            # Реверс — ТОЛЬКО для «чистого» вида (без префикса/суффикса)
            if rev_flag and not ((pre or "").strip() or (post or "").strip()):
                prompts = list(reversed(prompts))
            boundary = _to_end_step(boundary_f, self.steps)

            schedules = _build_bracket_after_schedules(
                self.prefix + (pre or ""), prompts, boundary, (post or "") + self.suffix, self.steps
            )
            return [[e, _apply_and(_collapse_spaces(t, keep_edges=True))] for e, t in schedules]

        # 0) owner::a::b!!, trailing
        if "::" in full and "!!" in full and all(ch not in full for ch in '[]()'):
            left, trailing = full.split("!!", 1)
            owner, rest = left.split("::", 1)
            descriptors = [x.strip(' ,~!;') for x in rest.split('::') if x.strip(' ,~!;')]
            sequences = [f"{owner.strip()}: {d}" for d in descriptors]
            trailing_text = [t.strip(' ,') for t in trailing.split(',') if t.strip(' ,')]
            out = f"{owner.strip()} -> {', '.join(sequences)}"
            if trailing_text:
                out += f", {', '.join(trailing_text)}"
            return [[self.steps, _collapse_spaces(self.prefix + out + self.suffix, keep_edges=True)]]

        # 1) owner::a::b!
        if '::' in full and (full.endswith('!') or full.endswith(';')) and all(ch not in full for ch in '[]()'):
            owner, rest = full.split('::', 1)
            rest = rest[:-1]
            descriptors = [x.strip(' ,~!;') for x in rest.split('::') if x.strip(' ,~!;')]
            text = f"{owner.strip()}: {', '.join(descriptors)}"
            return [[self.steps, _collapse_spaces(self.prefix + text + self.suffix, keep_edges=True)]]

        # 2) [a:b:...:N] — допускаем скобки внутри и безопасно сплитим,
        # запускаем fast-path только если в строке ровно одна пара []
        if full.count('[') == 1 and full.count(']') == 1 and '|' not in full:
            try:
                lb, rb = full.index('['), full.rindex(']')
                pre, inner, post = full[:lb], full[lb+1:rb], full[rb+1:]
                parts = _split_top_level_colon_keep_empty(inner)
                if len(parts) >= 2 and re.fullmatch(r'[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?', parts[-1]):
                    boundary_f = float(parts[-1])
                    prompts = [_unescape_literals(p.strip()) for p in parts[:-1]]
                    if prompts:
                        # Поддержка "...] reverse" как префиксного токена в постфиксе.
                        mrev = RE_REVERSE_PREFIX.match(post or "")
                        if mrev:
                            post = (post or "")[mrev.end():]
                            # Реверс только для «чистой» формы (как в get_schedule)
                            if not pre.strip() and not post.strip() and not (self.prefix.strip() or self.suffix.strip()):
                                prompts = list(reversed(prompts))
                        boundary = _to_end_step(boundary_f, self.steps)
                        schedules = _build_bracket_inner_schedules(self.prefix + pre, prompts, boundary, post + self.suffix, self.steps)
                        return [[e, _apply_and(_collapse_spaces(t, keep_edges=True))] for e, t in schedules]
            except ValueError:
                pass

        return self._default_visit(tree)

    def _default_visit(self, tree):
        # 1) Собираем расписание для каждого ребёнка без внешних аффиксов
        child_scheds = []
        for child in tree.children:
            if isinstance(child, lark.Token):
                if child.type == "WHITESPACE":
                    # пробел — константное расписание
                    child_scheds.append([[self.steps, " "]])
                else:
                    child_scheds.append([[self.steps, str(child)]])
            else:
                sub = CollectSteps(
                    self.steps,
                    prefix="",   # важно: без префикса/суффикса на уровне детей
                    suffix="",
                    use_scheduling=self.use_scheduling,
                    seed=self.seed,
                ).visit(child)
                if not sub:
                    # на всякий случай — literal fallback
                    sub = [[self.steps, resolve_tree(child, keep_spacing=True)]]
                child_scheds.append(sub)

        if not child_scheds:
            return [[self.steps, _collapse_spaces(self.prefix + self.suffix, keep_edges=True)]]

        # 2) Объединённые границы всех детей
        boundaries = sorted({int(e) for sched in child_scheds for (e, _) in sched})

        def pick_text(sched, step):
            for e, t in sched:
                if step <= int(e):
                    return t
            return sched[-1][1]

        # 3) Склейка «картинки шага» = конкатенация текущих текстов детей
        out = []
        for end in boundaries:
            parts = [pick_text(s, end) for s in child_scheds]
            text = self.prefix + "".join(parts) + self.suffix
            text = _apply_and(_collapse_spaces(text, keep_edges=True))
            if not out or out[-1][1] != text:
                out.append([end, text])

        return out


    def _visit_token(self, token):
        if token.type == "WHITESPACE":
            return []
        return [[self.steps, _collapse_spaces(self.prefix + str(token) + self.suffix, keep_edges=True)]]

    def visit_plain(self, tree):
        text = resolve_tree(tree, keep_spacing=True)
        return [[self.steps, _collapse_spaces(self.prefix + text + self.suffix, keep_edges=True)]]

    def visit_top_level_sequence3(self, tree):
        transformer = ScheduleTransformer(self.steps, 1, self.seed)
        text = transformer.transform(tree)
        return [[self.steps, _collapse_spaces(self.prefix + text + self.suffix, keep_edges=True)]]

    def visit_top_level_sequence(self, tree):
        # Аналогично visit_top_level_sequence3, но для 'owner::a::b!!, tail'
        transformer = ScheduleTransformer(self.steps, 1, self.seed)
        text = transformer.transform(tree)
        return [[self.steps, _collapse_spaces(self.prefix + text + self.suffix, keep_edges=True)]]

    def visit_scheduled(self, tree):
        if not tree.children:
            return [[self.steps, _collapse_spaces(self.prefix + "empty_prompt" + self.suffix, keep_edges=True)]]

        import lark as _l
        # Берём только поддеревья prompt (исключаем служебные узлы)
        prompts = []
        for node in tree.children:
            if isinstance(node, _l.Tree):
                data = getattr(node, "data", None)
                if data == "prompt":
                    prompts.append(node)
                elif data in ("step_range_list", "reverse_flag"):
                    continue

        number_node = next((p for p in tree.children if isinstance(p, _l.Token) and p.type == "NUMBER"), None)
        step_range_list = next((p for p in tree.children if isinstance(p, _l.Tree) and getattr(p, "data", None) == "step_range_list"), None)
        is_reverse = any(isinstance(p, _l.Tree) and getattr(p, "data", None) == "reverse_flag" for p in tree.children)

        try:
            weight = float(number_node.value) if number_node is not None else 1.0
        except Exception:
            weight = 1.0

        if not prompts:
            return [[self.steps, _collapse_spaces(self.prefix + "empty_prompt" + self.suffix, keep_edges=True)]]

        def _clamp_step(x: int) -> int:
            return max(1, min(x, self.steps))

        step_intervals: list[tuple[int,int]] = []
        first_is_percent: bool | None = None

        if step_range_list:
            # Явные диапазоны
            for sr in step_range_list.children:
                if not isinstance(sr, _l.Tree):
                    continue
                data = getattr(sr, "data", None)
                if data not in ("range_abs", "range_pct", "step_range"):
                    continue

                nums = [resolve_tree(ch, keep_spacing=False)
                        for ch in sr.children
                        if isinstance(ch, _l.Token) and ch.type == "NUMBER"]
                if len(nums) >= 2:
                    start_txt, end_txt = nums[0], nums[-1]
                else:
                    rng_txt = resolve_tree(sr, keep_spacing=False)
                    m = re.match(r'^\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+))%?\s*-\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+))%?\s*$', rng_txt)
                    if not m:
                        continue
                    start_txt, end_txt = m.group(1), m.group(2)

                is_percent = (data == "range_pct") or ('%' in resolve_tree(sr, keep_spacing=False))
                if first_is_percent is None:
                    first_is_percent = is_percent

                def _to_steps(txt: str, is_pct: bool) -> int:
                    try:
                        val = float(txt)
                        if is_pct:
                            val = val / 100.0 * self.steps
                        return int(round(val))
                    except Exception:
                        return 1

                start_step = _clamp_step(_to_steps(start_txt, is_percent))
                end_step   = _clamp_step(_to_steps(end_txt,   is_percent))
                if start_step <= end_step:
                    step_intervals.append((start_step, end_step))
        else:
            # "[...]: N" — «после границы»
            boundary = _clamp_step(int(round(weight * self.steps)) if weight <= 1.0 else int(round(weight)))
            prompt_texts = [_unescape_literals(resolve_tree(p, keep_spacing=True)) for p in prompts]
            has_affixes = bool(self.prefix.strip() or self.suffix.strip())
            if is_reverse and not has_affixes:
                prompt_texts = prompt_texts[::-1]
            schedules = _build_bracket_after_schedules(self.prefix, prompt_texts, boundary, self.suffix, self.steps)
            return [[e, _apply_and(_collapse_spaces(t, keep_edges=True))] for e, t in schedules]

        # ★ Фикс: пролог добавляем ТОЛЬКО если первый старт > 1 (и для процентов, и для чисел)
        schedules: list[list[int, str]] = []
        if step_intervals:
            first_start = step_intervals[0][0]
            if first_start > 1:
                pre_end = max(1, first_start - 1)
                schedules.append([pre_end, _concat_prefix_text_suffix(self.prefix, "", self.suffix)])

        if is_reverse:
            prompts = prompts[::-1]
            step_intervals = step_intervals[::-1]

        for i, (start, end) in enumerate(step_intervals[:len(prompts)]):
            end = min(end, self.steps)
            if start <= end:
                p = prompts[i]
                if isinstance(p, _l.Tree):
                    child_schedules = self.visit(p)
                    child_texts = [_unescape_literals(s[1]) for s in child_schedules] \
              or [_unescape_literals(resolve_tree(p, keep_spacing=True))]
                else:
                    child_texts = [resolve_tree(p, keep_spacing=True)]
                for txt in child_texts:
                    schedules.append([end, _concat_prefix_text_suffix(self.prefix, txt, self.suffix)])

        if step_intervals and step_intervals[-1][1] < self.steps:
            tail_text = _unescape_literals(resolve_tree(prompts[-1], keep_spacing=True))
            schedules.append([self.steps, _concat_prefix_text_suffix(self.prefix, tail_text, self.suffix)])

        if not schedules:
            return [[self.steps, _collapse_spaces(self.prefix + resolve_tree(tree, keep_spacing=True) + self.suffix, keep_edges=True)]]

        return [[e, _apply_and(_collapse_spaces(t, keep_edges=True))] for e, t in schedules]


    def visit_alternate(self, tree):
        # Собираем значения с учётом пустых опций между разделителями
        vals = []
        last_sep = True
        for child in tree.children:
            if isinstance(child, lark.Token) and child.type == "WHITESPACE":
                continue
            tok = str(child).strip()
            if tok in ("|", ",", "[", "]", "!"):
                if last_sep:
                    vals.append("")  # пустая опция между разделителями
                last_sep = True
                continue
            child_schedules = self.visit(child)
            if child_schedules:
                vals.append(child_schedules[0][1])
            else:
                vals.append(resolve_tree(child, keep_spacing=True))
            last_sep = False
        if last_sep:
            vals.append("")  # хвостовой разделитель -> пустая опция
        if not vals:
            return [[self.steps, _collapse_spaces(self.prefix + "empty_prompt" + self.suffix, keep_edges=True)]]
        if EXPAND_ALTERNATE_PER_STEP:
            schedules = []
            for step in range(1, self.steps + 1):
                choice = vals[(step - 1) % len(vals)]
                schedules.append([step, _collapse_spaces(self.prefix + choice + self.suffix, keep_edges=True)])
            return [[e, _apply_and(t)] for e, t in schedules]
        choice = self.rng.choice(vals)
        return [[self.steps, _collapse_spaces(self.prefix + choice + self.suffix, keep_edges=True)]]

    def visit_alternate_distinct(self, tree):
        import lark as _l
        options = []
        for child in tree.children:
            if isinstance(child, _l.Token):
                tok = str(child).strip()
                if tok in ("|", ",", "[", "]", "!"):
                    continue
                if tok:
                    options.append([tok])
                else:
                    options.append([""])
                continue
            child_schedules = self.visit(child)
            child_options = [sched[1] for sched in child_schedules if sched[1] != ""]
            if child_options:
                options.append(child_options)
            else:
                txt = resolve_tree(child, keep_spacing=True)
                options.append([txt])
        flat = [opt for group in options for opt in group]
        if not flat:
            return [[self.steps, _collapse_spaces(self.prefix + "empty_prompt" + self.suffix, keep_edges=True)]]
        selected = self.rng.choice(flat)
        return [[self.steps, _collapse_spaces(self.prefix + selected + self.suffix, keep_edges=True)]]

    def visit_alternate1(self, tree):
        # Собираем варианты, включая потенциально пустые
        import lark as _l
        options = []
        last_was_sep = True
        for child in tree.children:
            if isinstance(child, _l.Token):
                tok = str(child).strip()
                if tok in ("|", ","):
                    if last_was_sep:
                        options.append("")
                    last_was_sep = True
                    continue
                if child.type == "WHITESPACE":
                    continue
                s = str(child)
            else:
                s = resolve_tree(child, keep_spacing=True)
            if s is None:
                continue
            tok = s.strip()
            if tok in ("|", ","):
                if last_was_sep:
                    options.append("")
                last_was_sep = True
                continue
            options.append(s)
            last_was_sep = False
        if last_was_sep:
            options.append("")

        if not options:
            return [[self.steps, _collapse_spaces(self.prefix + "empty_prompt" + self.suffix, keep_edges=True)]]

        if EXPAND_ALTERNATE_PER_STEP:
            schedules = []
            for step in range(1, self.steps + 1):
                choice = options[(step - 1) % len(options)]
                schedules.append([step, _collapse_spaces(self.prefix + choice + self.suffix, keep_edges=True)])
            return [[e, _apply_and(t)] for e, t in schedules]
        else:
            choice = self.rng.choice(options)
            return [[self.steps, _collapse_spaces(self.prefix + choice + self.suffix, keep_edges=True)]]


    def visit_alternate2(self, tree):
        opts = []
        for c in tree.children:
            s = resolve_tree(c, keep_spacing=True).strip()
            if not s or s in ("|", ","):
                continue
            opts.append(s)
        options = opts
        suffix = options[0].split("_", 1)[1] if options and "_" in options[0] else ""
        combined = []
        for opt in options:
            combined.append(opt if "_" in opt or not suffix else f"{opt}_{suffix}")
        text = "|".join(combined) if combined else "empty_prompt"
        return [[self.steps, _collapse_spaces(self.prefix + text + self.suffix, keep_edges=True)]]

    def visit_weighted(self, tree):
        tr = ScheduleTransformer(self.steps, 1, self.seed)
        text = tr.transform(tree)  # '(token:1.2)'
        return [[self.steps, _collapse_spaces(self.prefix + text + self.suffix, keep_edges=True)]]

    def visit_grouped(self, tree):
        # Собираем варианты по дочерним узлам (сохраняем пустые элементы)
        all_options = []
        for child in tree.children:
            if isinstance(child, lark.Token) and child.type == "WHITESPACE":
                continue
            child_schedules = self.visit(child)
            # берём тексты без обрезки пустых, чтобы пустые варианты учитывались
            child_opts = [sched[1] for sched in child_schedules]
            if not child_opts:
                child_opts = [resolve_tree(child, keep_spacing=True)]
            all_options.append(child_opts)

        # Оценка числа комбинаций:
        total_combos = 1
        for opts in all_options:
            total_combos *= max(1, len(opts))

        if total_combos > GROUP_COMBO_LIMIT:
            mode = GROUP_COMBO_FALLBACK
            if mode == "literal":
                # Возвращаем псевдолитерал, начинающийся с "{[" чтобы сохранить сигнал исходной формы
                inner = ", ".join(resolve_tree(c, keep_spacing=True) for c in tree.children if not (isinstance(c, lark.Token) and c.type == "WHITESPACE"))
                original = "{[" + inner + "]}"
                return [[self.steps, _collapse_spaces(self.prefix + original + self.suffix, keep_edges=True)]]
            elif mode == "sample":
                k = GROUP_COMBO_LIMIT
                lens = [max(1, len(opts)) for opts in all_options]
                seen: set[tuple[int, ...]] = set()
                out = []
                max_tries = k * 10
                tries = 0
                while len(seen) < min(k, total_combos) and tries < max_tries:
                    idx = tuple(self.rng.randrange(n) for n in lens)
                    tries += 1
                    if idx in seen:
                        continue
                    seen.add(idx)
                    combo = [all_options[d][i] if all_options[d] else "" for d, i in enumerate(idx)]
                    text = ", ".join(combo).strip()
                    if text:
                        out.append([self.steps, _collapse_spaces(self.prefix + text + self.suffix, keep_edges=True)])
                if out:
                    return out
                # иначе сваливаемся на усечение (truncate)

        out = []
        for i, combo in enumerate(product(*all_options)):
            if i >= GROUP_COMBO_LIMIT:
                break
            text = ", ".join(combo).strip()
            if text:
                out.append([self.steps, _collapse_spaces(self.prefix + text + self.suffix, keep_edges=True)])
        return out or [[self.steps, _collapse_spaces(self.prefix + "empty_prompt" + self.suffix, keep_edges=True)]]

    def visit_sequence(self, tree):
        transformer = ScheduleTransformer(self.steps, 1, self.seed)
        text = transformer.transform(tree)
        return [[self.steps, _collapse_spaces(self.prefix + text + self.suffix, keep_edges=True)]]

    def visit_nested_sequence(self, tree):
        elements = [resolve_tree(c, keep_spacing=True).strip(" ,~!;")
                    for c in tree.children[:-1]
                    if resolve_tree(c, keep_spacing=True).strip(" ,~!;")]
        terminator = tree.children[-1] if tree.children and isinstance(tree.children[-1], lark.Token) else None
        if terminator and terminator.value == "~":
            text = self.rng.choice(elements) if elements else "empty_prompt"
        else:
            text = f"[{' | '.join(elements)}]"
        return [[self.steps, _collapse_spaces(self.prefix + text + self.suffix, keep_edges=True)]]

    def visit_numbered(self, tree):
        quantity = int(tree.children[0])
        distinct = False
        if len(tree.children) > 1:
            mark = tree.children[1]
            try:
                if str(mark) in ('!', '_'):
                    distinct = True
            except Exception:
                pass
        target = tree.children[-1]

        import lark as _l
        options = []
        def add_opts(node):
            if isinstance(node, _l.Tree):
                if getattr(node, "data", None) in ("alternate","alternate1","alternate2","prompt"):
                    for ch in node.children:
                        add_opts(ch)
                else:
                    txt = resolve_tree(node, keep_spacing=True)
                    options.append(txt)
            elif isinstance(node, _l.Token):
                txt = str(node)
                options.append(txt)

        add_opts(target)
        if not options:
            child_schedules = self.visit(target)
            options = [s[1] for s in child_schedules]
        if not options:
            return [[self.steps, _collapse_spaces(self.prefix + "empty_prompt" + self.suffix, keep_edges=True)]]

        if distinct:
            seen = []
            for opt in options:
                if opt not in seen:
                    seen.append(opt)
                if len(seen) == quantity:
                    break
            if len(seen) < quantity:
                seen += self.rng.choices(options, k=quantity - len(seen))
            selected = seen
        else:
            selected = self.rng.choices(options, k=quantity)

        return [[self.steps, _collapse_spaces(self.prefix + ", ".join(selected) + self.suffix, keep_edges=True)]]

    def visit_and_rule(self, tree):
        import lark as _l
        parts = []
        tr = ScheduleTransformer(self.steps, 1, self.seed)
        for child in tree.children:
            if isinstance(child, _l.Tree) and getattr(child, "data", None) in ("weighted", "emphasized"):
                parts.append(tr.transform(child))
            else:
                s = resolve_tree(child, keep_spacing=True)
                if s:
                    parts.append(s)
        text = " and ".join(parts)
        return [[self.steps, _collapse_spaces(self.prefix + text + self.suffix, keep_edges=True)]]

    def visit_emphasized(self, tree):
        # на уровне визитора возвращаем уже отформатированный "(text:weight)"
        tr = ScheduleTransformer(self.steps, 1, self.seed)
        text = tr.transform(tree)
        return [[self.steps, _collapse_spaces(self.prefix + text + self.suffix, keep_edges=True)]]

    def __call__(self, tree):
        self.schedules = self.visit(tree)
        # dedup по паре (step,text)
        uniq = []
        seen = set()
        for end_step, text in self.schedules or []:
            key = (int(end_step), text)
            if key not in seen:
                uniq.append([int(end_step), text])
                seen.add(key)
        # Паритет с get_schedule: последний сегмент эксплицитно тянется до self.steps
        if uniq:
            try:
                last_end = int(uniq[-1][0])
            except Exception:
                last_end = None
            if isinstance(last_end, int) and last_end < int(self.steps):
                uniq.append([int(self.steps), uniq[-1][1]])
        return uniq or [[self.steps, _collapse_spaces(self.prefix + resolve_tree(tree, keep_spacing=True) + self.suffix, keep_edges=True)]]


# ──────────────────────────────────────────────────────────────────────────────
# Внешние утилиты для расписаний
# ──────────────────────────────────────────────────────────────────────────────

def at_step_from_schedule(step: int, schedule: Sequence[Sequence[int | str]]) -> str:
    if not schedule:
        return ""
    for end_step, text in schedule:
        try:
            if step <= int(end_step):
                return text
        except Exception:
            continue
    return schedule[-1][1]

def at_step(step: int, prompt_or_schedule, *, steps: int | None = None,
            seed: int | None = 42, use_visitor: bool = True) -> str:
    if isinstance(prompt_or_schedule, list) and prompt_or_schedule and isinstance(prompt_or_schedule[0], list):
        return at_step_from_schedule(step, prompt_or_schedule)
    prompt = str(prompt_or_schedule)
    if steps is None:
        raise ValueError("steps is required when passing a prompt string to at_step(...)")
    sched = get_schedule(prompt, steps, use_scheduling=True, seed=seed, use_visitor=use_visitor)
    return at_step_from_schedule(step, sched)

@lru_cache(maxsize=CACHE_SIZE)
def _apply_and(text: str) -> str:
    # 1) оператор & -> ' and ' (окружённый пробелами)
    text = re.sub(r'\s*&\s*', ' and ', text)

    # 2) Нормализуем пробелы вокруг двоеточия в шаблоне 'token : number' -> 'token:number'
    # Только после буквы/подчёркивания (не цифры) и поддержка китайского двоеточия
    text = re.sub(
        rf'(?<=[^\W\d_])\s*[:：]\s*(?=(?:{NUMERIC_RE})\b)',
        ':',
        text,
    )

    # 3) Гарантируем одиночный пробел вокруг слова-оператора AND (не часть слова, не AND_PERP и т.п.)
    #    Примеры: ')and(' -> ') and (', 'cat  AND   dog' -> 'cat and dog'
    text = re.sub(r'(?i)\s*(?<![\w_])and(?![\w_])\s*', ' and ', text)

    # 4) Схлопнуть множественные пробелы
    return _re_ws_collapse.sub(" ", text)




# ──────────────────────────────────────────────────────────────────────────────
# Главный API построения расписания из строки
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=CACHE_SIZE)
def get_schedule(prompt: str, steps: int, use_scheduling: bool, seed: int | None, use_visitor: bool = True):
    import re as _re

    if not str(prompt).strip():
        return [[steps, SAFE_EMPTY]]

    # Литеральные \n и \t → реальные
    if "\\n" in prompt or "\\t" in prompt:
        prompt = prompt.replace("\\n", "\n").replace("\\t", "\t")

    # ➜ Новый шаг: приводим слово AND к & (и верхний, и нижний регистр)
    prompt = _re.sub(r'(?i)\bAND\b', '&', prompt)
    # ── Быстрые пути ──────────────────────────────────────────────────────────

    # TL3: owner::: ... !!!
    m_tl3 = _re.match(r'(?s)^\s*([^:\[\]\{\}\(\)]+?):::(.+?)!!!(?:,\s*(.*))?\s*$', prompt)
    if m_tl3:
        owner, rest, trailing = m_tl3.groups()
        parts = [p.strip() for p in rest.split(',') if p.strip()]
        seq_texts = []
        for seg in parts:
            seg = seg.rstrip('!;').strip()
            toks = [t.strip() for t in seg.split('::') if t.strip()]
            if toks:
                label, descs = toks[0], toks[1:]
                s = f"{owner.strip()}: {label}"
                if descs:
                    s += f", {', '.join(descs)}"
                seq_texts.append(s)
        trailing_texts = []
        if trailing:
            for t in trailing.split(','):
                t = t.strip()
                if not t:
                    continue
                mseq = _re.match(r'^\s*([^:\[\]\{\}\(\)]+?)::(.+?)([!;])\s*$', t)
                if mseq:
                    seq_owner, rest2, _term = mseq.groups()
                    descs = [x.strip(' ,~!;') for x in rest2.split('::') if x.strip(' ,~!;')]
                    trailing_texts.append(f"{seq_owner.strip()}: {', '.join(descs)}")
                else:
                    trailing_texts.append(t)
        text = f"{owner.strip()} -> {', '.join(seq_texts)}"
        if trailing_texts:
            text += f", {', '.join(trailing_texts)}"
        return [[steps, _apply_and(_collapse_spaces(text, keep_edges=True))]]

    # Numbered альтернативы в []
    m_num_alt = _re.match(r'^\s*(\d+)\s*([!_])?\s*\[([^\]]+)\]\s*$', str(prompt))
    if m_num_alt:
        import random as _rnd
        qty_txt, mark, inner = m_num_alt.groups()
        quantity = int(qty_txt)
        # split по НЕэкранированному '|', затем разэкранируем литералы
        options = [_unescape_literals(x.strip()) for x in _re.split(r'(?<!\\)\|', inner)]
        options = [opt if opt != "" else SAFE_EMPTY for opt in options]
        options_unique = list(dict.fromkeys(options)) or [SAFE_EMPTY]
        if mark:
            if quantity <= len(options_unique):
                chosen = options_unique[:quantity]
            else:
                need = quantity - len(options_unique)
                pad = (options_unique * ((need + len(options_unique) - 1)//len(options_unique)))[:need]
                chosen = options_unique + pad
        else:
            rng = _rnd.Random(seed) if seed is not None else _rnd
            chosen = rng.choices(options_unique, k=quantity)
        return [[steps, _apply_and(_collapse_spaces(', '.join(chosen), keep_edges=True))]]


    # Явные диапазоны: "[...]:N a-b,c-d [r]" — ДОЛЖНЫ идти ПЕРЕД общим "[...]:N"
    m_ranges = RE_BRACKET_AFTER_WITH_RANGES.match(prompt)
    if m_ranges:
        inner = m_ranges.group("inner")
        ranges_txt = m_ranges.group("ranges")
        rev = m_ranges.group("rev")

        def _to_steps_local(txt: str) -> int:
            s = txt.strip()
            if s.endswith('%'):
                try:
                    return _clamp(round(float(s[:-1]) / 100.0 * steps), steps)
                except Exception:
                    return 1
            try:
                return _clamp(round(float(s)), steps)
            except Exception:
                return 1

        prompts = [_unescape_literals(p.strip()) for p in _split_top_level_colon_keep_empty(inner)]
        if rev:
            prompts = list(reversed(prompts))

        ranges = []
        for part in ranges_txt.split(','):
            if '-' in part:
                a, b = part.split('-', 1)
                ra, rb = _to_steps_local(a), _to_steps_local(b)
                if ra < rb:
                    ranges.append((ra, rb))

        schedules = []
        if ranges and ranges[0][0] > 1:
            schedules.append([ranges[0][0] - 1, ""])
        for i, (start, end) in enumerate(ranges[:len(prompts)]):
            schedules.append([min(end, steps), prompts[i]])
        if ranges and ranges[-1][1] < steps and prompts:
            schedules.append([steps, prompts[-1]])
        return (
            [[e, _apply_and(_collapse_spaces(t, keep_edges=True))] for e, t in schedules]
            or [[steps, _apply_and(_collapse_spaces(_unescape_literals(inner), keep_edges=True))]]
        )

    # НОВЫЙ fast-path: "[...:N]" (число/доля *внутри* скобок)
    m_inner = _re.match(r'(?s)^(.*)\[(.*?)\](.*)$', prompt)
    if m_inner and prompt.count('[') == 1 and prompt.count(']') == 1:
        pre, inner, post = m_inner.groups()
        # не конфликтуем со сложными конструкциями
        # В этом fast-path нас интересует сложность ТОЛЬКО внутри inner.
        if not _needs_complex_parse(inner or "", inner or ""):
            parts = _split_top_level_colon_keep_empty(inner)
            if len(parts) >= 2 and RE_NUMERIC_FULL.fullmatch(parts[-1]):
                try:
                    boundary_f = float(parts[-1])
                except Exception:
                    boundary_f = None
                if boundary_f is not None:
                    prompts = [_unescape_literals(p.strip()) for p in parts[:-1]]
                    # reverse в постфиксе (как префиксный токен)
                    mrev = RE_REVERSE_PREFIX.match(post or "")
                    rev_flag = bool(mrev)
                    if mrev:
                        post = (post or "")[mrev.end():]
                    if rev_flag and not (pre.strip() or post.strip()):
                        prompts = list(reversed(prompts))
                    boundary = _to_end_step(boundary_f, steps)
                    schedules = _build_bracket_inner_schedules(pre, prompts, boundary, post, steps)
                    return [[e, _apply_and(_collapse_spaces(t, keep_edges=True))] for e, t in schedules]

    # "[...]:N" с префиксом/суффиксом — общий случай
    m_with_pre = RE_BRACKET_AFTER.match(prompt)
    if m_with_pre and prompt.count('[') == 1 and prompt.count(']') == 1:
        pre, inner, boundary_txt = m_with_pre.group(1), m_with_pre.group(2), m_with_pre.group(3)
        rev_token = m_with_pre.group('rev')
        post = m_with_pre.group('post') or ""
        # Если в post есть явные диапазоны (например "6-8" или "10%-40%") — отдаем грамматике,
        # иначе fast-path сломает семантику.
        if _re.search(r'\b\d+%?\s*-\s*\d+%?', post):
            pass  # не возвращаем — ниже сработает общий парсер
        elif _needs_complex_parse(inner or "", prompt):
            pass  # отдаём грамматике/Visitor
        else:
            try:
                boundary = _to_end_step(float(boundary_txt), steps)
                inner_prompts = [_unescape_literals(p.strip()) for p in _split_top_level_colon_keep_empty(inner)]

                # reverse мог прийти отдельным токеном ИЛИ как начало post (префикс)
                m_rev = RE_REVERSE_PREFIX.match(post or "")
                rev_here = bool(rev_token) or bool(m_rev)
                if m_rev:
                    post = (post or "")[m_rev.end():]

                # Реверс ТОЛЬКО для «чистой» формы (без префикса/суффикса)
                if rev_here and not ((pre or "").strip() or (post or "").strip()):
                    inner_prompts = list(reversed(inner_prompts))

                schedules = _build_bracket_after_schedules(pre, inner_prompts, boundary, post, steps)
                return [[e, _apply_and(_collapse_spaces(t, keep_edges=True))] for e, t in schedules]
            except Exception:
                pass


    # ── Обычный разбор ─────────────────────────────────────────────────────────

    try:
        tree = schedule_parser.parse(prompt)
    except lark.exceptions.LarkError as e:
        logger.warning("Prompt parse failed: '%s' — %s", prompt, e)
        return [[steps, _collapse_spaces(prompt, keep_edges=True)]]

    collector = CollectSteps(steps, use_scheduling=use_scheduling, seed=seed)
    schedules = collector(tree)
    try:
        schedules.sort(key=lambda x: int(x[0]))
    except Exception:
        pass

    if DEDUP_SCHEDULE_STEPS:
        try:
            schedules = _dedup_schedules(schedules)
        except Exception:
            pass

    if not schedules:
        return [[steps, _collapse_spaces(prompt, keep_edges=True)]]

    if not use_visitor:
        rebuilt = []
        for end_step, _ in schedules:
            transformer = ScheduleTransformer(total_steps=steps, current_step=end_step, seed=seed)
            text = transformer.transform(tree)
            rebuilt.append([end_step, _apply_and(_collapse_spaces(text, keep_edges=True))])
        return rebuilt

    return [[e, _apply_and(_collapse_spaces(t, keep_edges=True))] for e, t in schedules]


# ──────────────────────────────────────────────────────────────────────────────
# Сервиска
# ──────────────────────────────────────────────────────────────────────────────

def _dedup_schedules(schedules: list[list[int, str]], joiner: str = ", ") -> list[list[int, str]]:
    if not schedules:
        return schedules
    last_by_step: dict[int, str] = {}
    for end, text in schedules:
        try:
            key = int(end)
        except Exception:
            key = end
        last_by_step[key] = text
    # Устойчивая сортировка: сначала числовые ключи по возрастанию, потом строковые.
    return [[k, v] for k, v in sorted(
        last_by_step.items(),
        key=lambda kv: (0, int(kv[0])) if isinstance(kv[0], int) else (1, str(kv[0]))
    )]

# ──────────────────────────────────────────────────────────────────────────────
# parse_prompt_attention — внимательность/веса
# ──────────────────────────────────────────────────────────────────────────────

re_attention = re.compile(rf"""
\\\(|
\\\)|
\\\[|
\\\]|
\\\\|
\\|
\(|
\[|
:\s*({NUMERIC_RE})\s*\)|
\)|
]|
[^\\()\[\]:\s]+|
\s+|
:
""", re.X)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

@lru_cache(maxsize=CACHE_SIZE)
def parse_prompt_attention(text):
    text = str(text or "")
    if not text.strip():
        return ((SAFE_EMPTY, 1.0),)

    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # FIX: нормализуем и реальный таб, и литерал таба до пробелов ДО токенизации
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = text.replace('\\n', ' ').replace('\\t', ' ')

    res = []
    round_brackets = []
    square_brackets = []
    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    pending_colon = False
    pending_colon_had_space = False
    last_token_index = -1

    orig_text = text  # для ремонтного прохода ниже
    for m in re_attention.finditer(text):
        text_match = m.group(0)
        wgrp = m.group(1)  # число для ( ... : <num> )

        if text_match.startswith('\\'):
            res.append([text_match[1:], 1.0])
            last_token_index = len(res) - 1

        elif text_match == '(':
            round_brackets.append(len(res)); pending_colon = False

        elif text_match == '[':
            square_brackets.append(len(res)); pending_colon = False

        elif wgrp is not None and round_brackets:
            try:
                multiply_range(round_brackets.pop(), float(wgrp))
            except ValueError:
                multiply_range(round_brackets.pop(), 1.0)
            pending_colon = False

        elif text_match == ')' and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
            pending_colon = False

        elif text_match == ']' and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
            pending_colon = False

        elif text_match == ':' and not round_brackets and not square_brackets:
            # Всегда "откладываем" решение: это вес или обычное двоеточие?
            pending_colon = True
            pending_colon_had_space = False
            if not SUPPRESS_STANDALONE_COLON:
                res.append([':', 1.0])
                last_token_index = len(res) - 1

        else:
            chunk = text_match
            
            # Если ждём число после ':' и пришли только пробелы — продолжаем ждать
            if pending_colon and not (round_brackets or square_brackets):
                if chunk.strip() == "":
                    # запомнили, что после ':' был пробел(ы)
                    pending_colon_had_space = True
                    continue

            if pending_colon and not (round_brackets or square_brackets):
                mnum = re.match(r"^\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)", chunk)
                if mnum:
                    j = last_token_index
                    while j >= 0 and res[j][0] in (':', 'BREAK'):
                        j -= 1
                    if j >= 0:
                        try:
                            res[j][1] = float(mnum.group(1))
                            chunk = chunk[mnum.end():]
                        except ValueError:
                            pass
                    pending_colon = False
                else:
                    # это НЕ вес. Вернём двоеточие (и, при необходимости, пробел) к предыдущему токену.
                    j = last_token_index
                    while j >= 0 and res[j][0] in (':', 'BREAK'):
                        j -= 1
                    if j >= 0:
                        res[j][0] = res[j][0] + ":" + (" " if pending_colon_had_space else "")
                    else:
                        res.append([':', 1.0])
                        last_token_index = len(res) - 1
                        if pending_colon_had_space:
                            res.append([' ', 1.0])
                            last_token_index = len(res) - 1
                    pending_colon = False


            parts = re.split(re_break, chunk)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                if part == "":
                    continue
                # НЕ трогаем пробелы — оставляем как есть,
                # обработаем слияние на последнем шаге
                res.append([part, 1.0])
                last_token_index = len(res) - 1

    # --- ВОССТАНОВЛЕНИЕ ВИСЯЩЕГО ДВОЕТОЧИЯ В КОНЦЕ СТРОКИ --------------------
    # Если дошли до конца с ожиданием числа (pending_colon=True), но число так и не встретилось,
    # трактуем ':' как литерал и приклеиваем его к предыдущему токену (с учётом пробела),
    # даже если SUPPRESS_STANDALONE_COLON включён.
    if pending_colon and not (round_brackets or square_brackets):
        j = last_token_index
        while j >= 0 and res[j][0] in (':', 'BREAK'):
            j -= 1
        if j >= 0:
            res[j][0] = res[j][0] + ":" + (" " if pending_colon_had_space else "")
        else:
            res.append([':', 1.0])
            if pending_colon_had_space:
                res.append([' ', 1.0])
        pending_colon = False

    if round_brackets or square_brackets:
        pass

    if not res:
        return ((SAFE_EMPTY, 1.0),)

    # --- Ремонтный проход для шаблона "word : number" (с пробелами) -----------
    # Если по какой-то причине вес не приклеился в основном цикле — гарантируем это здесь.
    rx_fix = re.compile(rf'(\b[^\s:()\[\]{{}}]+)\s*:\s*({NUMERIC_RE})')
    used = {}
    for mm in rx_fix.finditer(orig_text):
        word = mm.group(1)
        try:
            wt = float(mm.group(2))
        except ValueError:
            continue
        k = used.get(word, 0)
        # найдём k-ю по счёту неподправленную запись этого слова
        cnt = 0
        for i, (t, w) in enumerate(res):
            if t == word:
                if cnt == k:
                    res[i][1] = wt
                    used[word] = k + 1
                    break
                cnt += 1


    # --- Сшивка "word  +0.2" / "word  -0.1" через ПРОБЕЛ ---
    rx_delta_token = re.compile(rf'^\s*([-+]{NUMERIC_NOSIGN_RE})\s*$')
    i = 0
    while i + 1 < len(res):
        t, w = res[i]
        if t not in (':', 'BREAK') and w == 1.0:
            # пропускаем любые пробелы после слова
            j = i + 1
            had_space = False
            while j < len(res) and res[j][1] == 1.0 and res[j][0].strip() == "":
                had_space = True
                j += 1
            if j < len(res) and res[j][1] == 1.0:
                m = rx_delta_token.match(res[j][0])
                if m:
                    # трактуем как дельту веса для ПРЕДЫДУЩЕГО слова
                    try:
                        res[i][1] = float(m.group(1))
                    except ValueError:
                        pass
                    # удалить дельту и промежуточные пробелы; один пробел вернуть при необходимости
                    del res[i+1:j+1]
                    if had_space:
                        res.insert(i+1, [" ", 1.0])
                    continue
        i += 1

    # второй проход: inline-веса word:weight ИЛИ word(+/-)delta
    rx_inline = re.compile(
        r'(?:'
        r'(\b[^\s:(){{}}\[\]]+)\s*:\s*'      # g1: word
        rf'({NUMERIC_RE})'                   # g2: weight
        r')|(?:'
        r'(\b[^\s:(){{}}\[\]]+)\s*'         # g3: word
        rf'([-+](?:{NUMERIC_NOSIGN_RE}))'   # g4: +/-delta — знак строго один
        r')'
    )

    new_res = []
    for txt, w in res:
        if txt == "BREAK" or w != 1.0:
            new_res.append([txt, w])
            continue
        pos = 0; changed = False
        for mm in rx_inline.finditer(txt):
            pre = txt[pos:mm.start()]
            if pre:
                new_res.append([pre, 1.0])
            if mm.group(1) is not None:
                word, wt = mm.group(1), float(mm.group(2))
            else:
                word, wt = mm.group(3), float(mm.group(4))
            new_res.append([word, wt]); changed = True
            pos = mm.end()
        if changed:
            tail = txt[pos:]
            if tail:
                new_res.append([tail, 1.0])
        else:
            new_res.append([txt, w])
    res = new_res

    # убрать ведущий "AND " (но сохранить пробелы после)
    norm = []
    for t, w in res:
        if t == "BREAK":
            norm.append([t, w]); continue
        m = re.match(r'^(\s*)AND\s+(.*)$', t, flags=re.I)
        if m:
            t = m.group(1) + m.group(2)
        norm.append([t, w])
    res = norm

    # финальное схлопывание соседей с одинаковым весом — ВСТАВЛЯЕМ ПРОБЕЛ при склейке
    def _smart_concat(a: str, b: str):
        if not a:
            return b
        if not b:
            return a

        ar = a.rstrip()
        # После , . ; пробел можно убрать, но ':' трогаем отдельно
        if ar and ar[-1] in ",.;":
            a = ar
            b = b.lstrip()
            return a + b

        # если уже есть пробел на стыке — не добавляем
        if a.endswith(" ") or b.startswith(" "):
            return a + b

        # если оба текстовые и алфанумерика — добавляем одиночный пробел
        if a[-1].isalnum() and b[0].isalnum():
            return a + " " + b

        return a + b


    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1] and res[i][0] not in (':','BREAK') and res[i + 1][0] not in (':','BREAK'):
            res[i][0] = _smart_concat(res[i][0], res[i + 1][0])
            res.pop(i + 1)
        else:
            i += 1

    # Сжать повторные пробелы внутри каждого токена
    res = [[_collapse_spaces(t), w] if t != "BREAK" else [t, w] for t, w in res]
    # Удалить пустые, если вдруг остались
    res = [[t, w] for t, w in res if t or t == "BREAK"]

    # Если остались только BREAK — вернём безопасный токен
    if all(t == "BREAK" for t, _ in res):
        return ((SAFE_EMPTY, 1.0),)

    return tuple((t, w) for t, w in res)


# ──────────────────────────────────────────────────────────────────────────────
# Мультиконд — как в старых версиях, но с дедупом текстов расписаний
# ──────────────────────────────────────────────────────────────────────────────

ScheduledPromptConditioning = namedtuple("ScheduledPromptConditioning", ["end_at_step", "cond"])

class SdConditioning(list):
    def __init__(self, prompts, is_negative_prompt=False, width=None, height=None, copy_from=None):
        super().__init__()
        self.extend(prompts)
        if copy_from is None:
            copy_from = prompts
        self.is_negative_prompt = is_negative_prompt or getattr(copy_from, 'is_negative_prompt', False)
        self.width = width or getattr(copy_from, 'width', None)
        self.height = height or getattr(copy_from, 'height', None)

def get_learned_conditioning_prompt_schedules(
    prompts: list[str],
    base_steps: int,
    hires_steps: int | None = None,
    use_old_scheduling: bool = False,
    seed: int | None = 42,
    use_visitor: bool = True,
):
    steps = hires_steps if (hires_steps is not None and not use_old_scheduling) else base_steps
    use_scheduling = (hires_steps is None) or use_old_scheduling
    prompt_schedules = [get_schedule(p, steps, use_scheduling, seed, use_visitor=use_visitor) for p in prompts]
    return prompt_schedules

def get_learned_conditioning(model, prompts: SdConditioning | list[str], steps, hires_steps=None, use_old_scheduling=False):
    res = []
    prompt_schedules = get_learned_conditioning_prompt_schedules(prompts, steps, hires_steps, use_old_scheduling)
    cache = {}
    for prompt, prompt_schedule in zip(prompts, prompt_schedules):
        if not prompt_schedule:
            raise ValueError(f"Empty schedule for prompt '{prompt}'")
        cached = cache.get(prompt, None)
        if cached is not None:
            res.append(cached); continue
        texts = SdConditioning([x[1] for x in prompt_schedule], copy_from=prompts)
        conds = model.get_learned_conditioning(texts)
        cond_schedule = []
        for i, (end_at_step, _) in enumerate(prompt_schedule):
            if isinstance(conds, dict):
                cond = {k: v[i] for k, v in conds.items()}
            else:
                cond = conds[i]
            cond_schedule.append(ScheduledPromptConditioning(end_at_step, cond))
        cache[prompt] = cond_schedule
        res.append(cond_schedule)
    return res

# === DROP-IN REPLACEMENT ===
class ComposableScheduledPromptConditioning:
    def __init__(self, schedules, weight=1.0):
        self.schedules: list[ScheduledPromptConditioning] = schedules
        self.weight: float = weight

class MulticondLearnedConditioning:
    def __init__(self, shape, batch):
        self.shape: tuple = shape
        self.batch: list[list[ComposableScheduledPromptConditioning]] = batch

def get_multicond_learned_conditioning(model, prompts, steps, hires_steps=None, use_old_scheduling=False):
    effective_steps = hires_steps if hires_steps is not None and not use_old_scheduling else steps
    prompt_schedules = get_learned_conditioning_prompt_schedules(prompts, effective_steps, hires_steps, use_old_scheduling)
    if not prompt_schedules or any(not sch for sch in prompt_schedules):
        raise ValueError("Empty schedule for at least one prompt")
    conds_list, prompt_flat_list, prompt_indexes = get_multicond_prompt_list(prompts)

    text_to_index: dict[str, int] = {}
    unique_texts: list[str] = []
    schedule_text_indices_per_prompt: list[list[int]] = []

    for schedule in prompt_schedules:
        indices_for_prompt: list[int] = []
        for end_at_step, text in schedule:
            if text not in text_to_index:
                text_to_index[text] = len(unique_texts)
                unique_texts.append(text)
            indices_for_prompt.append(text_to_index[text])
        schedule_text_indices_per_prompt.append(indices_for_prompt)

    texts_conditioning = SdConditioning(unique_texts, copy_from=prompts)
    model_conds = model.get_learned_conditioning(texts_conditioning)

    def get_i_cond(i: int):
        if isinstance(model_conds, dict):
            return {k: v[i] for k, v in model_conds.items()}
        return model_conds[i]

    res_batch: list[list[ComposableScheduledPromptConditioning]] = []
    for prompt_idx, (schedule, cond_parts) in enumerate(zip(prompt_schedules, conds_list)):
        composable_parts: list[ComposableScheduledPromptConditioning] = []
        conds_for_steps = []
        for local_step_idx, (end_at_step, _text) in enumerate(schedule):
            i_global = schedule_text_indices_per_prompt[prompt_idx][local_step_idx]
            cond_at_step = get_i_cond(i_global)
            conds_for_steps.append(ScheduledPromptConditioning(int(end_at_step), cond_at_step))
        for _flat_index, weight in cond_parts:
            composable_parts.append(ComposableScheduledPromptConditioning(conds_for_steps, weight))
        res_batch.append(composable_parts)

    if isinstance(model_conds, dict):
        ca = model_conds.get('crossattn')
        if isinstance(ca, list) and ca:
            shape = getattr(ca[0], 'shape', None) or (0,)
        else:
            shape = getattr(ca, 'shape', None) or (0,)
    else:
        shape = getattr(model_conds, 'shape', None) or (0,)

    return MulticondLearnedConditioning(shape, res_batch)


re_AND = re.compile(r"\bAND\b(?!_PERP|_SALT|_TOPK)", re.I)
# Вес в конце подпрампта: "text : 1.2" (якорь на конец строки, поддержка китайского двоеточия)
RE_END_WEIGHT = re.compile(
    r"^(?P<text>.*?)"
    rf"(?:[:：]\s*(?P<w>{NUMERIC_RE}))?"
    r"\s*$"
)

def get_multicond_prompt_list(prompts: SdConditioning | list[str]):
    res_indexes = []
    prompt_indexes = {}
    prompt_flat_list = SdConditioning(prompts)
    prompt_flat_list.clear()

    for prompt in prompts:
        subprompts = re_AND.split(prompt)
        indexes = []
        for subprompt in subprompts:
            s = subprompt if isinstance(subprompt, str) else str(subprompt)

            # 1) Нормализация скобочных весов: "(cat:2.0)" или просто "(cat)"
            m_emph = re.match(
                r'^\s*\(\s*(.*?)\s*(?::\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)\s*)?\)\s*$',
                s
            )
            if m_emph:
                text = m_emph.group(1)
                weight = m_emph.group(2) if m_emph.group(2) is not None else 1.1
            else:
                # 2) Надёжный разбор "word:weight" (вес — именно в конце подпрампта)
                s_clean = (s or "").strip().rstrip(",;")
                m_end = RE_END_WEIGHT.match(s_clean)
                if m_end:
                    text, weight = m_end.group("text"), m_end.group("w")
                else:
                    text, weight = (s_clean, None)

            # ── Ремонт: иногда регэксп оставляет '1' в тексте и отдаёт вес как '.2'.
            #            Если вес начинается с точки и текст оканчивается на цифру —
            #            переносим эту цифру(ы) в начало веса: '...1' + '.2' → '1.2'.
            if weight is not None and weight.startswith(".") and text and text[-1].isdigit():
                j = len(text) - 1
                while j >= 0 and text[j].isdigit():
                    j -= 1
                digits = text[j + 1:]          # хвостовые цифры
                text = text[:j + 1].rstrip()   # отрезаем их из текста
                weight = digits + weight       # склеиваем '1' + '.2' -> '1.2'

            text = (text or "").strip()
            # Нормализуем артефакты вроде "dog:" -> "dog"
            if text.endswith(":") or text.endswith("："):
                text = text[:-1].rstrip()
            if not text.strip():
                text = SAFE_EMPTY
            try:
                weight = float(weight) if weight is not None else 1.0
            except Exception:
                weight = 1.0
            index = prompt_indexes.get(text, None)
            if index is None:
                index = len(prompt_flat_list)
                prompt_flat_list.append(text)
                prompt_indexes[text] = index
            indexes.append((index, weight))
        res_indexes.append(indexes)

    return res_indexes, prompt_flat_list, prompt_indexes


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогалки для реконструкции батчей
# ──────────────────────────────────────────────────────────────────────────────

class DictWithShape(dict):
    def __init__(self, x, shape=None):
        super().__init__()
        self.update(x)
        self._shape = shape
    @property
    def shape(self):
        if self._shape is not None:
            return self._shape
        any_val = self.get("crossattn")
        if any_val is None and self:
            any_val = next(iter(self.values()))
        return getattr(any_val, "shape", None)

def reconstruct_cond_batch(c: list[list[ScheduledPromptConditioning]], current_step):
    import torch
    if not c or not c[0]:
        raise ValueError("Empty conditioning schedule")
    param = c[0][0].cond
    if param is None:
        raise ValueError("Invalid conditioning parameter")
    is_dict = isinstance(param, dict)
    if is_dict:
        dict_cond = param
        res = {k: torch.zeros((len(c),) + v.shape, device=getattr(v, "device", "cpu"), dtype=getattr(v, "dtype", torch.float32)) for k, v in dict_cond.items()}
        res = DictWithShape(res, (len(c),) + dict_cond.get('crossattn', next(iter(dict_cond.values()))).shape)
    else:
        res = torch.zeros((len(c),) + param.shape, device=getattr(param, "device", "cpu"), dtype=getattr(param, "dtype", torch.float32))

    for i, cond_schedule in enumerate(c):
        target_index = 0
        for current, entry in enumerate(cond_schedule):
            if current_step <= int(entry.end_at_step):
                target_index = current
                break
        if is_dict:
            for k, v in cond_schedule[target_index].cond.items():
                res[k][i] = v
        else:
            res[i] = cond_schedule[target_index].cond
    return res

def stack_conds(tensors):
    import torch
    token_count = max([x.shape[0] for x in tensors])
    for i in range(len(tensors)):
        if tensors[i].shape[0] != token_count:
            last_vector = tensors[i][-1: ]
            last_vector_repeated = last_vector.repeat([token_count - tensors[i].shape[0], 1])
            tensors[i] = torch.vstack([tensors[i], last_vector_repeated])
    return torch.stack(tensors)

def reconstruct_multicond_batch(c: MulticondLearnedConditioning, current_step):
    import torch
    if not c.batch or not c.batch[0]:
        raise ValueError("Empty multicond batch")
    param = c.batch[0][0].schedules[0].cond
    if param is None:
        raise ValueError("Invalid conditioning parameter")
    tensors = []
    conds_list = []
    for composable_prompts in c.batch:
        conds_for_batch = []
        for composable_prompt in composable_prompts:
            target_index = 0
            for current, entry in enumerate(composable_prompt.schedules):
                try:
                    end_at_step = int(entry.end_at_step)
                except ValueError:
                    end_at_step = current_step
                if current_step <= end_at_step:
                    target_index = current
                    break
            conds_for_batch.append((len(tensors), composable_prompt.weight))
            tensors.append(composable_prompt.schedules[target_index].cond)
        conds_list.append(conds_for_batch)

    if isinstance(tensors[0], dict):
        keys = list(tensors[0].keys())
        stacked = {k: stack_conds([x[k] for x in tensors]) for k in keys}
        stacked = DictWithShape(stacked, stacked.get('crossattn', next(iter(stacked.values()))).shape)
    else:
        stacked = stack_conds(tensors).to(device=getattr(param, "device", "cpu"), dtype=getattr(param, "dtype", torch.float32))
    return conds_list, stacked


# ──────────────────────────────────────────────────────────────────────────────
# Тестовые самопроверки
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import doctest, random
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
    random.seed(42)

    g = lambda p: get_learned_conditioning_prompt_schedules([p], 10)[0]

    assert g("test") == [[10, 'test']]
    assert g("a [b:3]") == [[3, 'a'], [10, 'a b']]
    assert g("[(a:2):3]") == [[3, ''], [10, '(a:2)']]

    big = "{[a|b|c|d|e|f|g],[h|i|j|k|l|m|n],[o|p|q|r|s|t|u]}"
    res = g(big)
    assert len(res) <= GROUP_COMBO_LIMIT

    g2 = lambda p: get_learned_conditioning_prompt_schedules([p], 6)[0]
    one = g2("[cat|dog|fox]!")
    assert len(set([txt for _, txt in one])) == 1
    
    # reverse для «чистой» формы со ступенчатым делением внутри скобок
    g3 = lambda p: get_learned_conditioning_prompt_schedules([p], 6)[0]
    assert g3("[a:b:4] reverse") == [[2, 'b'], [4, 'a'], [6, 'a']]

    # Нумерованные альтернативы с нормализацией пробелов/AND
    g4 = lambda p: get_learned_conditioning_prompt_schedules([p], 5)[0]
    txt = g4("3[_a|b| |c]")[0][1]
    assert isinstance(txt, str) and len(txt) > 0

    # Пустые сегменты внутри [...:N] (чистая форма и reverse)
    g5 = lambda p: get_learned_conditioning_prompt_schedules([p], 6)[0]
    assert g5("[a::4]") == [[2, 'a'], [4, ''], [6, '']]
    assert g5("[a::4] reverse") == [[2, ''], [4, 'a'], [6, 'a']]

    # Префикс/суффикс + пустой сегмент: добавляется пролог до первого интервала
    g6 = lambda p: get_learned_conditioning_prompt_schedules([p], 6)[0]
    assert g6("X [a::4] Y") == [[1, 'X Y'], [2, 'X a Y'], [4, 'X Y'], [6, 'X Y']]

    # --- новые проверки табов/переводов строки ---
    pa1 = parse_prompt_attention("dog\\tcat")
    joined1 = "".join(t for t, w in pa1 if t != "BREAK").strip()
    assert joined1 == "dog cat", f"Expected 'dog cat', got {joined1!r}"

    pa2 = parse_prompt_attention("dog\tcat")
    joined2 = "".join(t for t, w in pa2 if t != "BREAK").strip()
    assert joined2 == "dog cat", f"Expected 'dog cat', got {joined2!r}"

    print("All quick integration tests passed!")
