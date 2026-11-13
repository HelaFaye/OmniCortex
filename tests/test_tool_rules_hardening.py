import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import tool_rules as rules


def test_argument_traversal_has_a_node_limit(monkeypatch):
    monkeypatch.setattr(rules, "MAX_ARGUMENT_NODES", 20)
    args = {"values": [{} for _ in range(100)]}
    with pytest.raises(rules.RuleRegexTimeout, match="nodes"):
        rules.cond_holds(
            {"regex": ".*", "path": "*"}, args, time.monotonic() + 2
        )


def test_argument_traversal_has_a_depth_limit(monkeypatch):
    monkeypatch.setattr(rules, "MAX_ARGUMENT_DEPTH", 8)
    nested = "leaf"
    for _ in range(20):
        nested = [nested]
    with pytest.raises(rules.RuleRegexTimeout, match="depth"):
        rules.cond_holds(
            {"regex": "leaf", "path": "*"},
            {"value": nested},
            time.monotonic() + 2,
        )


def test_argument_traversal_checks_deadline():
    with pytest.raises(rules.RuleRegexTimeout, match="budget"):
        rules.cond_holds(
            {"regex": ".*", "path": "*"},
            {"value": "text"},
            time.monotonic() - 1,
        )


def test_argument_value_count_is_bounded():
    args = {"values": [str(i) for i in range(rules.MAX_ARGUMENT_VALUES + 1)]}
    with pytest.raises(rules.RuleRegexTimeout, match="values"):
        rules.cond_holds(
            {"regex": "^never-match$", "path": "values.*"},
            args,
            time.monotonic() + 2,
        )


def test_rules_load_fails_closed_on_invalid_json_shapes(tmp_path):
    path = tmp_path / "rules.json"
    rule_set = rules.Rules(path)

    path.write_text("[]")
    assert rule_set.load() == rules.blank()

    path.write_text(json.dumps({
        "default": "deny",
        "categories": [],
        "rules": [],
    }))
    loaded = rule_set.load()
    assert loaded["default"] == "deny"
    assert loaded["categories"] == rules.blank()["categories"]
    assert loaded["rules"] == []


def test_rules_load_ignores_malformed_individual_rule_fields(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(json.dumps({
        "rules": [
            {"id": "bad-rate", "action": "allow", "max_per_hour": 999999},
            {"id": "good", "action": "deny"},
        ]
    }))
    loaded = rules.Rules(path).load()
    assert [item["id"] for item in loaded["rules"]] == ["good"]


def test_max_per_hour_is_bounded():
    with pytest.raises(ValueError, match="max_per_hour"):
        rules.validate_rule({
            "id": "too-many",
            "action": "allow",
            "max_per_hour": rules.MAX_RATE_HISTORY + 1,
        })


def test_future_rate_history_does_not_block_allow(tmp_path):
    now = time.time()
    rule_set = rules.Rules(tmp_path / "rules.json")
    rule_set.replace({
        "default": "ask",
        "categories": {},
        "rules": [{
            "id": "future",
            "action": "allow",
            "max_per_hour": 1,
            "recent": [now + 7200],
        }],
    })
    result = rule_set.decide("server", "tool", {}, record=False, now=now)
    assert result["action"] == "allow"


def test_rule_file_size_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(rules, "MAX_RULE_FILE_BYTES", 16)
    path = tmp_path / "rules.json"
    path.write_text(" " * 17)
    assert rules.Rules(path).load() == rules.blank()


def test_global_rule_lock_wait_obeys_decision_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(rules, "RULE_MATCH_BUDGET_SECONDS", 0.02)
    rule_set = rules.Rules(tmp_path / "rules.json")

    rules._LOCK.acquire()
    try:
        result = rule_set.decide("server", "tool", {})
    finally:
        rules._LOCK.release()

    assert result["action"] == "ask"
    assert "lock is busy" in result["reason"]



def test_argument_path_length_is_bounded():
    path = "a" * (rules.MAX_ARGUMENT_PATH_CHARS + 1)
    with pytest.raises(ValueError, match="argument path exceeds"):
        rules.validate_rule({
            "action": "allow",
            "args": [{"path": path, "regex": "safe"}],
        })


def test_argument_wildcards_must_be_complete_path_segments():
    for path in ("files.*.path", "*", "files.0.path"):
        validated = rules.validate_rule({
            "action": "allow",
            "args": [{"path": path, "regex": "safe"}],
        })
        assert validated["args"][0]["path"] == path

    for path in ("files*.path", "files.*suffix", "files.**.path"):
        with pytest.raises(ValueError, match="bad argument path"):
            rules.validate_rule({
                "action": "allow",
                "args": [{"path": path, "regex": "safe"}],
            })


def test_rule_from_call_rejects_empty_or_excessive_exact_arguments():
    with pytest.raises(ValueError, match="non-empty"):
        rules.rule_from_call("args", "server", "tool", {})

    too_many = {
        f"arg{i}": i
        for i in range(rules.MAX_CONDITIONS + 1)
    }
    with pytest.raises(ValueError, match="at most"):
        rules.rule_from_call("args", "server", "tool", too_many)


def test_rule_from_call_rejects_unrepresentable_argument_keys():
    for args in (
        {"nested.key": "value"},
        {"bad key": "value"},
        {1: "value"},
    ):
        with pytest.raises(ValueError, match="safe exact-match path"):
            rules.rule_from_call("args", "server", "tool", args)


def test_rule_from_call_rejects_truncated_or_oversized_argument_values():
    too_long = {"value": "x" * rules.MAX_MATCH_CHARS}
    with pytest.raises(ValueError, match="text limit"):
        rules.rule_from_call("args", "server", "tool", too_long)

    regex_too_long = {"value": "." * (rules.MAX_REGEX // 2 + 1)}
    with pytest.raises(ValueError, match="complex"):
        rules.rule_from_call("args", "server", "tool", regex_too_long)


def test_rule_from_call_preserves_exact_supported_arguments():
    args = {"command": "echo hello", "count": 3}
    rule = rules.rule_from_call("args", "server", "tool", args)
    validated = rules.validate_rule(rule)

    assert validated["action"] == "allow"
    assert [condition["path"] for condition in validated["args"]] == [
        "command",
        "count",
    ]
    assert rules.rule_matches(
        validated, "server", "tool", ["unknown"], args
    )
    assert not rules.rule_matches(
        validated, "server", "tool", ["unknown"],
        {"command": "echo hello", "count": 4},
    )
