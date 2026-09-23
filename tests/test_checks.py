"""Unit tests for the optional checkRules constraint regression engine."""

from __future__ import annotations

import pytest

from app.checks import (
    CheckRule,
    evaluate_check_rules,
    parse_rules,
    render_scope,
)
from app.merger import parse_pointer
from app.models import CheckRule as RequestCheckRule


def _rule(scope: str, fields, mode: str) -> CheckRule:
    return CheckRule(parse_pointer(scope), tuple(fields), mode)


def _violations(base, merged, rules):
    return evaluate_check_rules(base, merged, rules)


def _locators(result):
    return [(v.rule, render_scope(v.scope), v.fields) for v in result]


# ------------------------------------------------------------- all_or_none


def test_all_or_none_none_present_is_compliant():
    result = _violations(
        {"x": {}}, {"x": {"z": 1}},
        [_rule("/x", ["a", "b"], "all_or_none")],
    )
    assert result == []


def test_all_or_none_all_present_is_compliant():
    result = _violations(
        {"x": {}}, {"x": {"a": 1, "b": None}},
        [_rule("/x", ["a", "b"], "all_or_none")],
    )
    assert result == []


def test_all_or_none_some_present_is_a_violation():
    result = _violations(
        {"x": {}}, {"x": {"a": 1, "z": 2}},
        [_rule("/x", ["a", "b"], "all_or_none")],
    )
    assert _locators(result) == [(0, "/x", ["a"])]


def test_all_or_none_null_field_counts_as_present():
    result = _violations(
        {"x": {}}, {"x": {"a": None}},
        [_rule("/x", ["a", "b"], "all_or_none")],
    )
    assert _locators(result) == [(0, "/x", ["a"])]


def test_all_or_none_all_null_fields_is_compliant():
    result = _violations(
        {"x": {}}, {"x": {"a": None, "b": None}},
        [_rule("/x", ["a", "b"], "all_or_none")],
    )
    assert result == []


# ------------------------------------------------------------- at_most_one


def test_at_most_one_zero_present_is_compliant():
    result = _violations(
        {"x": {}}, {"x": {}},
        [_rule("/x", ["a", "b", "c"], "at_most_one")],
    )
    assert result == []


def test_at_most_one_one_present_is_compliant():
    result = _violations(
        {"x": {}}, {"x": {"b": None}},
        [_rule("/x", ["a", "b", "c"], "at_most_one")],
    )
    assert result == []


def test_at_most_one_two_present_is_a_violation():
    result = _violations(
        {"x": {}}, {"x": {"c": 1, "a": 2}},
        [_rule("/x", ["a", "b", "c"], "at_most_one")],
    )
    assert _locators(result) == [(0, "/x", ["a", "c"])]


def test_at_most_one_three_present_lists_them_sorted():
    result = _violations(
        {"x": {}}, {"x": {"c": 1, "a": 2, "b": 3}},
        [_rule("/x", ["a", "b", "c"], "at_most_one")],
    )
    assert _locators(result) == [(0, "/x", ["a", "b", "c"])]


# ------------------------------------------------------- regression logic


def test_violation_already_in_base_is_not_reported_even_when_fields_change():
    # Base already violates (only a); merged still violates (only a) and
    # changes unrelated content: no new violation.
    result = _violations(
        {"x": {"a": 1}},
        {"x": {"a": 9, "z": 2}},
        [_rule("/x", ["a", "b"], "all_or_none")],
    )
    assert result == []


def test_violation_in_base_with_different_present_set_still_not_reported():
    # Base violates with {a}; merged violates with {b}: still a
    # pre-existing violation at the same scope, not a regression.
    result = _violations(
        {"x": {"a": 1}},
        {"x": {"b": 1}},
        [_rule("/x", ["a", "b"], "all_or_none")],
    )
    assert result == []


def test_scope_deleted_from_merged_is_not_reported():
    result = _violations(
        {"x": {"a": 1}}, {},
        [_rule("/x", ["a", "b"], "all_or_none")],
    )
    assert result == []


def test_scope_becoming_non_object_is_not_reported():
    result = _violations(
        {"x": {"a": 1}}, {"x": 5},
        [_rule("/x", ["a", "b"], "all_or_none")],
    )
    assert result == []


def test_previously_violating_scope_fixed_reports_nothing():
    result = _violations(
        {"x": {"a": 1}},
        {"x": {"a": 1, "b": 2}},
        [_rule("/x", ["a", "b"], "all_or_none")],
    )
    assert result == []


def test_compliant_base_becoming_violating_is_reported():
    result = _violations(
        {"x": {"a": 1, "b": 2}},
        {"x": {"a": 1}},
        [_rule("/x", ["a", "b"], "all_or_none")],
    )
    assert _locators(result) == [(0, "/x", ["a"])]


def test_new_scope_added_by_merged_is_evaluated():
    result = _violations(
        {}, {"new": {"a": 1}},
        [_rule("/new", ["a", "b"], "all_or_none")],
    )
    assert _locators(result) == [(0, "/new", ["a"])]


def test_new_scope_added_compliant_is_not_reported():
    result = _violations(
        {}, {"new": {"a": 1, "b": 2}},
        [_rule("/new", ["a", "b"], "all_or_none")],
    )
    assert result == []


def test_violation_reports_merged_present_fields_not_base_fields():
    # Compliant base (both present); merged keeps a and replaces b with a
    # value still present, but a one-sided change removes b -> merged only a.
    result = _violations(
        {"x": {"a": 1, "b": 2}},
        {"x": {"a": 3}},
        [_rule("/x", ["a", "b"], "all_or_none")],
    )
    assert _locators(result) == [(0, "/x", ["a"])]


# ----------------------------------------------------------- scope shapes


def test_root_scope_empty_pointer():
    result = _violations(
        {}, {"a": 1},
        [_rule("", ["a", "b"], "all_or_none")],
    )
    assert _locators(result) == [(0, "", ["a"])]


def test_literal_pointer_into_array_index():
    result = _violations(
        {"arr": [{"a": 1}]},
        {"arr": [{"a": 1, "b": 2}]},
        [_rule("/arr/0", ["a", "b"], "at_most_one")],
    )
    assert _locators(result) == [(0, "/arr/0", ["a", "b"])]


def test_literal_index_matching_base_only_is_aligned():
    # The scope exists in base and merged at the same concrete pointer; a
    # wildcard never expands over the array, but the literal index aligns.
    result = _violations(
        {"arr": [{}]},
        {"arr": [{"a": 1}]},
        [_rule("/arr/0", ["a", "b"], "all_or_none")],
    )
    assert _locators(result) == [(0, "/arr/0", ["a"])]


def test_unresolved_literal_yields_no_scope():
    result = _violations(
        {"arr": [{"a": 1}]},
        {"arr": [{"a": 1, "b": 2}]},
        [_rule("/arr/9", ["a", "b"], "at_most_one")],
    )
    assert result == []


def test_non_object_literal_scope_yields_no_scope():
    result = _violations(
        {"x": 1}, {"x": 2},
        [_rule("/x", ["a", "b"], "all_or_none")],
    )
    assert result == []


def test_missing_member_yields_no_scope():
    result = _violations(
        {}, {"x": {"a": 1}},
        [_rule("/y/z", ["a", "b"], "all_or_none")],
    )
    assert result == []


# --------------------------------------------------------------- wildcard


def test_wildcard_expands_one_level_of_object_keys():
    result = _violations(
        {"services": {"s1": {}, "s2": {}}},
        {"services": {"s1": {"a": 1}, "s2": {}}},
        [_rule("/services/*", ["a", "b"], "all_or_none")],
    )
    assert _locators(result) == [(0, "/services/s1", ["a"])]


def test_wildcard_does_not_expand_into_arrays():
    result = _violations(
        {"arr": [{}]},
        {"arr": [{"a": 1, "b": 2}]},
        [_rule("/arr/*", ["a", "b"], "at_most_one")],
    )
    assert result == []


def test_wildcard_does_not_traverse_multiple_levels():
    # The direct children of the expanded scope are checked; a field two
    # levels below the wildcard is invisible to the rule.
    result = _violations(
        {"s": {}},
        {"s": {"k": {"nested": {"a": 1}}}},
        [_rule("/s/*", ["a", "b"], "all_or_none")],
    )
    assert result == []


def test_wildcard_matches_multiple_sibling_scopes_and_sorts():
    result = _violations(
        {"s": {"z": {}, "a": {}, "m": {}}},
        {"s": {"z": {"a": 1}, "a": {"a": 1}, "m": {"a": 1}}},
        [_rule("/s/*", ["a", "b"], "all_or_none")],
    )
    assert _locators(result) == [
        (0, "/s/a", ["a"]),
        (0, "/s/m", ["a"]),
        (0, "/s/z", ["a"]),
    ]


def test_multiple_wildcard_segments():
    result = _violations(
        {"env": {"prod": {"feature": {}}}},
        {"env": {"prod": {"feature": {"f1": {"x": 1}}}}},
        [_rule("/env/*/feature/*", ["x", "y"], "all_or_none")],
    )
    assert _locators(result) == [(0, "/env/prod/feature/f1", ["x"])]


def test_wildcard_scopes_aligned_with_base_by_concrete_pointer():
    # s1 violates in base already -> skipped; s2 is newly violating.
    result = _violations(
        {"s": {"s1": {"a": 1}, "s2": {}}},
        {"s": {"s1": {"a": 1, "z": 9}, "s2": {"a": 1}}},
        [_rule("/s/*", ["a", "b"], "all_or_none")],
    )
    assert _locators(result) == [(0, "/s/s2", ["a"])]


def test_wildcard_scope_removed_from_merged_not_reported():
    result = _violations(
        {"s": {"gone": {"a": 1}}},
        {"s": {}},
        [_rule("/s/*", ["a", "b"], "all_or_none")],
    )
    assert result == []


def test_wildcard_key_is_rendered_with_rfc6901_escaping():
    result = _violations(
        {"s": {}},
        {"s": {"a/b": {"a": 1}, "~z": {}}},
        [_rule("/s/*", ["a", "b"], "all_or_none")],
    )
    assert _locators(result) == [(0, "/s/a~1b", ["a"])]


# ------------------------------------------------- multiple rules & ordering


def test_results_sorted_by_rule_index_then_scope():
    rules = [
        _rule("/s/*", ["a", "b"], "all_or_none"),
        _rule("/s/m", ["a", "b"], "all_or_none"),
    ]
    result = _violations(
        {"s": {"z": {}, "m": {}, "a": {}}},
        {"s": {"z": {"a": 1}, "m": {"a": 1}, "a": {"a": 1}}},
        rules,
    )
    assert _locators(result) == [
        (0, "/s/a", ["a"]),
        (0, "/s/m", ["a"]),
        (0, "/s/z", ["a"]),
        (1, "/s/m", ["a"]),
    ]


def test_same_scope_violating_two_rules_reports_both():
    rules = [
        _rule("/x", ["a", "b"], "at_most_one"),
        _rule("/x", ["a", "c"], "at_most_one"),
    ]
    result = _violations(
        {"x": {}}, {"x": {"a": 1, "b": 2, "c": 3}}, rules
    )
    assert _locators(result) == [
        (0, "/x", ["a", "b"]),
        (1, "/x", ["a", "c"]),
    ]


def test_unlisted_fields_are_ignored():
    result = _violations(
        {"x": {}},
        {"x": {"other": 1, "another": 2}},
        [_rule("/x", ["a", "b"], "at_most_one")],
    )
    assert result == []


# ------------------------------------------------------------- parse_rules


def test_parse_rules_from_request_models():
    models = [
        RequestCheckRule(scope="/x", fields=["a", "b"], mode="all_or_none"),
        RequestCheckRule(scope="", fields=["c", "d", "e"], mode="at_most_one"),
    ]
    parsed = parse_rules(models)
    assert parsed[0].scope == ("x",)
    assert parsed[0].fields == ("a", "b")
    assert parsed[0].mode == "all_or_none"
    assert parsed[1].scope == ()
    assert parsed[1].mode == "at_most_one"


def test_parse_rules_rejects_malformed_scope():
    model = RequestCheckRule(scope="no-leading-slash", fields=["a", "b"],
                             mode="all_or_none")
    with pytest.raises(ValueError):
        parse_rules([model])


def test_render_root_scope_is_empty_string():
    assert render_scope(()) == ""
