from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from oc_cli import add_trust_remote_code_argument
import autotrace


def _trust_value(argv):
    parser = argparse.ArgumentParser()
    add_trust_remote_code_argument(parser)
    return parser.parse_args(argv).trust_remote_code


def test_remote_code_is_disabled_by_default():
    assert _trust_value([]) is False


@pytest.mark.parametrize("flag", ["--trust-remote-code", "--trust_remote_code"])
def test_remote_code_requires_explicit_opt_in(flag):
    assert _trust_value([flag]) is True


@pytest.mark.parametrize("flag", ["--no-trust-remote-code", "--no_trust_remote_code"])
def test_legacy_negative_flag_remains_safe(flag):
    assert _trust_value([flag]) is False


def test_trust_flags_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        _trust_value(["--trust-remote-code", "--no_trust_remote_code"])


def test_autotrace_remote_code_is_disabled_by_default():
    args = autotrace.parse_args([
        "--upstream", "http://127.0.0.1:8080",
        "--binary", "llama-cett-dump",
        "--gguf", "model.gguf",
        "--tokenizer", "example/tokenizer",
        "--n-layers", "32",
    ])
    assert args.trust_remote_code is False


def test_autotrace_remote_code_can_be_explicitly_enabled():
    args = autotrace.parse_args([
        "--upstream", "http://127.0.0.1:8080",
        "--binary", "llama-cett-dump",
        "--gguf", "model.gguf",
        "--tokenizer", "example/tokenizer",
        "--n-layers", "32",
        "--trust-remote-code",
    ])
    assert args.trust_remote_code is True


@pytest.mark.parametrize("trust", [False, True])
def test_autotrace_passes_explicit_trust_setting_to_transformers(monkeypatch, trust):
    seen = {}

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            seen["model_id"] = model_id
            seen.update(kwargs)
            return object()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = _AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(autotrace, "CFG", {
        "tokenizer": "example/tokenizer",
        "trust_remote_code": trust,
    })

    assert autotrace._get_tokenizer() is not None
    assert seen == {
        "model_id": "example/tokenizer",
        "trust_remote_code": trust,
    }


def test_all_legacy_hf_cli_tools_use_shared_opt_in():
    cli_files = [
        "extract_activations.py", "extract_activations_gguf.py",
        "make_answer_tokens.py", "trace_sample.py", "export_lora.py",
        "intervene_model.py", "apply_profile.py", "merge_selective.py",
        "score_tokens.py", "tune_scale.py", "neuron_attrib.py",
    ]
    for name in cli_files:
        source = (SCRIPTS / name).read_text(encoding="utf-8")
        assert "add_trust_remote_code_argument" in source, name
        assert 'add_argument("--no_trust_remote_code"' not in source, name


def test_shared_model_loader_default_is_false():
    import ast

    tree = ast.parse((SCRIPTS / "oc_common.py").read_text(encoding="utf-8"))
    loader = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "load_model")
    names = [arg.arg for arg in loader.args.args]
    defaults = dict(zip(names[-len(loader.args.defaults):], loader.args.defaults))
    trust_default = defaults["trust_remote_code"]
    assert isinstance(trust_default, ast.Constant) and trust_default.value is False


def test_runtime_sources_do_not_unconditionally_enable_remote_code():
    import ast

    for folder in (ROOT / "scripts", ROOT / "viz"):
        for source in folder.rglob("*.py"):
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                for keyword in node.keywords:
                    if (keyword.arg == "trust_remote_code"
                            and isinstance(keyword.value, ast.Constant)
                            and keyword.value.value is True):
                        pytest.fail(
                            f"unconditional trust_remote_code=True at "
                            f"{source.relative_to(ROOT)}:{node.lineno}"
                        )
