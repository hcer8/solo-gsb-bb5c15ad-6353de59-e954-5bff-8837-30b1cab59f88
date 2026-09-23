"""Optional ``checkRules`` regression checks for the ``/merge`` response.

A request may carry rules constraining which direct child fields of an
*object scope* may coexist:

* ``scope`` is a JSON Pointer (RFC 6901) naming an object, or a pointer whose
  last ``*`` (and any other ``*`` tokens along the way) expand over the keys
  of one object level -- a ``*`` token matches a single object key only, it
  never matches array indices;
* ``fields`` names at least two distinct direct child fields of every object
  the scope expands to;
* ``mode`` ``all_or_none`` requires either all named fields to be present or
  none of them, while ``at_most_one`` allows at most one of them.

A field counts as present whenever the object carries it -- a JSON ``null``
value is present.

The checks are pure *regression* checks. Each rule is evaluated in both the
request ``base`` and the final adjudicated ``merged`` document, aligned by
rule and concrete scope pointer, and only violations that appear freshly in
``merged`` are reported: a violation already present in base is never
re-reported, and a scope deleted from ``merged`` (or no longer an object)
produces nothing.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

from .merger import _array_index, escape_token, parse_pointer

# A document value may be any JSON value.
JsonValue = Optional[Any]


class RawViolation(NamedTuple):
    """One regression violation found in the final merged document."""

    rule_index: int
    scope: str  # concrete RFC 6901 pointer of the violating object
    fields: list[str]  # named fields actually present there, rule order


class CheckSpec(NamedTuple):
    """One compiled, validated check rule."""

    index: int
    scope: tuple[str, ...]
    fields: tuple[str, ...]
    mode: str


def compile_check_rules(rules: list[Any]) -> list[CheckSpec]:
    """Validate and index raw check rules.

    Each *rules* element is the already type-validated ``CheckRule`` model;
    pointer syntax and the field-name constraints (at least two distinct
    strings) are the API-level ``400`` checks and raise :class:`ValueError`
    here.
    """
    specs: list[CheckSpec] = []
    for index, rule in enumerate(rules):
        try:
            scope = parse_pointer(rule.scope)
        except ValueError as exc:
            raise ValueError(
                f"invalid checkRules[{index}] scope pointer {rule.scope!r}: {exc}"
            ) from exc
        fields = tuple(rule.fields)
        if len(fields) < 2:
            raise ValueError(
                f"invalid checkRules[{index}] fields: at least two distinct "
                "field names are required"
            )
        if len(set(fields)) != len(fields):
            raise ValueError(
                f"invalid checkRules[{index}] fields: field names must not repeat"
            )
        specs.append(CheckSpec(index, scope, fields, rule.mode))
    return specs


def evaluate_check_rules(
    base: JsonValue, merged: JsonValue, specs: list[CheckSpec]
) -> list[RawViolation]:
    """Return the merged-only violations of *specs*.

    Each rule is expanded against both documents and the concrete scopes are
    aligned by rendered pointer; only scopes that violate the rule in
    *merged* while not violating it in *base* are reported. Output is sorted
    by rule index and concrete scope pointer.
    """
    violations: list[RawViolation] = []
    for spec in specs:
        base_scopes = set(_expand(base, spec.scope))
        merged_scopes = set(_expand(merged, spec.scope))
        for tokens in sorted(
            base_scopes | merged_scopes, key=lambda t: _render(t)
        ):
            base_fields = _present_fields(_descend(base, tokens), spec.fields)
            merged_fields = _present_fields(
                _descend(merged, tokens), spec.fields
            )
            if _violates(merged_fields, spec) and not _violates(
                base_fields, spec
            ):
                violations.append(
                    RawViolation(spec.index, _render(tokens), merged_fields)
                )
    violations.sort(key=lambda v: (v.rule_index, v.scope))
    return violations


def _violates(present: list[str], spec: CheckSpec) -> bool:
    """Whether a scope with *present* named fields breaks *spec*.

    ``all_or_none`` is broken when the fields appear partially (some but not
    all); ``at_most_one`` is broken when two or more appear.
    """
    count = len(present)
    if spec.mode == "all_or_none":
        return 0 < count < len(spec.fields)
    return count >= 2


def _present_fields(obj: Any, fields: tuple[str, ...]) -> list[str]:
    """Rule-declared fields carried by object *obj*, in declaration order.

    Only a JSON object has child fields; a missing scope or any non-object
    value reports no present fields. Presence includes a real ``null``.
    """
    if not isinstance(obj, dict):
        return []
    return [name for name in fields if name in obj]


def _expand(document: Any, pattern: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Expand a scope pattern over *document* to concrete token tuples.

    A ``*`` token enumerates the keys of one object level (never array
    indices); literal tokens follow object members and, per RFC 6901,
    concrete array indices. Branches that do not resolve (missing key,
    out-of-range index, a non-object in the way) yield nothing.
    """
    concrete: list[tuple[str, ...]] = []

    def walk(node: Any, tokens: tuple[str, ...], acc: tuple[str, ...]) -> None:
        if not tokens:
            concrete.append(acc)
            return
        token, rest = tokens[0], tokens[1:]
        if isinstance(node, dict):
            if token == "*":
                for key in node:
                    walk(node[key], rest, acc + (key,))
            elif token in node:
                walk(node[token], rest, acc + (token,))
            return
        if isinstance(node, list) and token != "*":
            index = _array_index(token)
            if index is not None and 0 <= index < len(node):
                walk(node[index], rest, acc + (token,))

    walk(document, pattern, ())
    return concrete


def _descend(document: Any, tokens: tuple[str, ...]) -> Any:
    """Value at concrete *tokens* in *document*, or ``None`` when absent.

    The returned ``None`` deliberately conflates "absent" with a real JSON
    ``null``: neither is an object scope, so no check fields can be present
    at it either way.
    """
    node: Any = document
    for token in tokens:
        if isinstance(node, dict):
            if token not in node:
                return None
            node = node[token]
        elif isinstance(node, list):
            index = _array_index(token)
            if index is None or index >= len(node):
                return None
            node = node[index]
        else:
            return None
    return node


def _render(tokens: tuple[str, ...]) -> str:
    """Render concrete scope tokens as an RFC 6901 pointer string."""
    if not tokens:
        return ""
    return "/" + "/".join(escape_token(token) for token in tokens)
