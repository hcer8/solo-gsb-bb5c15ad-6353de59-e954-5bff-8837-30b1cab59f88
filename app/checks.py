"""Optional ``checkRules`` constraint regression checks for ``POST /merge``.

The check is pure post-processing on top of the ordinary merge result and
never changes it: the same merge, protection and adjudication semantics
produce ``merged`` first, and afterwards every rule is evaluated twice --
once against the request ``base`` and once against the final, adjudicated
``merged`` -- so only violations that are *new in merged* are reported. A
violation already present in base that persists, or an object scope that
disappears from merged, is not reported.

A rule has:

* a ``scope`` -- an RFC 6901 JSON Pointer locating the object (or objects,
  when a ``*`` token expands over one single level of object keys) whose
  direct children are constrained;
* ``fields`` -- at least two distinct direct child field names;
* a ``mode``:

  - ``all_or_none``: either every named field is present or none is; a
    scope carrying some but not all of them violates the rule;
  - ``at_most_one``: at most one named field may be present; a scope
    carrying two or more violates the rule.

Field presence follows JSON membership: an existing key counts, including a
key whose value is ``null``; an absent key does not. Only concrete
locations that resolve to a JSON object are scopes: a literal pointer
segment that cannot be resolved, or a resolved non-object value, yields no
scope (and never an error). Literal segments follow RFC 6901 (object member
name, or an array index); the ``*`` wildcard matches object keys only -- it
never expands into arrays or scalars and never matches more than one level.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

# Reuse the engine's RFC 6901 pointer dialect (parsing and token escaping)
# so configured and reported pointers match the ones the rest of the
# service speaks.
from .merger import escape_token, parse_pointer

JsonValue = Optional[Any]


class CheckRule(NamedTuple):
    """One parsed and validated check rule."""

    scope: tuple[str, ...]
    fields: tuple[str, ...]
    mode: str  # "all_or_none" | "at_most_one"


class RuleViolation(NamedTuple):
    """One violation that is newly present in merged."""

    rule: int
    scope: tuple[str, ...]
    fields: list[str]


def parse_rules(rules: list[Any]) -> list[CheckRule]:
    """Parse rule scopes into reference-token tuples.

    Field cardinality/uniqueness and mode are checked by the caller so the
    HTTP layer can attach a ``checkRules[index]`` context to every 400; this
    helper only parses the scope pointers and propagates their errors.
    """
    parsed: list[CheckRule] = []
    for rule in rules:
        tokens = parse_pointer(rule.scope)
        parsed.append(
            CheckRule(scope=tokens, fields=tuple(rule.fields), mode=rule.mode)
        )
    return parsed


def evaluate_check_rules(
    base: JsonValue, merged: JsonValue, rules: list[CheckRule]
) -> list[RuleViolation]:
    """Return the violations of *rules* that newly appear in *merged*.

    Each rule expands its object scopes in *base* and in *merged*
    independently; the concrete scope pointers are then aligned and a
    violation is reported only when merged violates the rule and the same
    scope in base did not. Scopes absent from merged (deleted, or no longer
    objects) never contribute.
    """
    violations: list[RuleViolation] = []
    for index, rule in enumerate(rules):
        base_scopes = _expand_object_scopes(base, rule.scope)
        merged_scopes = _expand_object_scopes(merged, rule.scope)
        for scope, merged_obj in merged_scopes.items():
            merged_present = _present_fields(merged_obj, rule.fields)
            if not _violates(rule.mode, len(merged_present), len(rule.fields)):
                continue
            base_obj = base_scopes.get(scope)
            if base_obj is not None:
                # The scope already existed in base: report only when base
                # satisfied the rule and merged no longer does. A violation
                # that was already present and simply persists is skipped.
                base_present = _present_fields(base_obj, rule.fields)
                if _violates(rule.mode, len(base_present), len(rule.fields)):
                    continue
            violations.append(
                RuleViolation(rule=index, scope=scope, fields=merged_present)
            )
    # Deterministic order: rule index first, then the rendered scope
    # pointer (the same text ordering used for reported conflicts).
    violations.sort(key=lambda v: (v.rule, render_scope(v.scope)))
    return violations


def render_scope(tokens: tuple[str, ...]) -> str:
    """Render concrete scope tokens back into an RFC 6901 pointer."""
    return "".join("/" + escape_token(token) for token in tokens)


def _violates(mode: str, present: int, total: int) -> bool:
    """Whether *present* of *total* named fields breaks the rule mode."""
    if mode == "all_or_none":
        # Some but not all: the "all present" and "none present" extremes
        # both satisfy the rule.
        return 0 < present < total
    # at_most_one: zero or one present is fine.
    return present > 1


def _present_fields(obj: dict, fields: tuple[str, ...]) -> list[str]:
    """Configured *fields* that are actual members of *obj*, sorted.

    Membership -- not value -- is what counts, so a key holding JSON ``null``
    is present just like any other value.
    """
    return sorted(field for field in fields if field in obj)


def _expand_object_scopes(
    root: JsonValue, tokens: tuple[str, ...]
) -> dict[tuple[str, ...], dict]:
    """Resolve *tokens* from *root* to every concrete object scope.

    The returned mapping keys are the concrete token tuples reached (literal
    segments kept verbatim, ``*`` replaced by one object key); its values
    are the resolved JSON objects. Locations that resolve to a non-object,
    and branches a literal segment cannot follow, are absent from the map.
    """
    scopes: dict[tuple[str, ...], dict] = {}
    _expand(root, tokens, (), scopes)
    return scopes


def _expand(
    node: Any,
    tokens: tuple[str, ...],
    prefix: tuple[str, ...],
    scopes: dict[tuple[str, ...], dict],
) -> None:
    if not tokens:
        # Only a concrete JSON object is an actual scope.
        if isinstance(node, dict):
            scopes[prefix] = node
        return
    token, rest = tokens[0], tokens[1:]
    if token == "*":
        # A wildcard matches one single level of object keys. Arrays and
        # scalars have no object-key children, so the pattern ends there.
        if isinstance(node, dict):
            for key, child in node.items():
                _expand(child, rest, prefix + (key,), scopes)
        return
    child = _resolve_literal(node, token)
    if child is not _UNRESOLVED:
        _expand(child, rest, prefix + (token,), scopes)


# Sentinel for "the literal token does not resolve here" -- distinct from a
# successfully resolved member whose value is JSON null.
_UNRESOLVED: Any = object()


def _resolve_literal(node: Any, token: str) -> Any:
    """Resolve one literal RFC 6901 reference token against *node*."""
    if isinstance(node, dict):
        return node[token] if token in node else _UNRESOLVED
    if isinstance(node, list):
        index = _rfc6901_array_index(token)
        if index is not None and 0 <= index < len(node):
            return node[index]
    return _UNRESOLVED


def _rfc6901_array_index(token: str) -> Optional[int]:
    """RFC 6901 array index syntax: ``"0"`` or digits without a leading zero."""
    if token == "0":
        return 0
    if token and token[0] in "123456789" and token.isdigit():
        return int(token)
    return None
