"""The paper/latest protocols: resolution, classification, and CLI defaults."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ndcb import protocols
from ndcb.cli import build_parser
from ndcb.protocols import PROTOCOLS, classify, judge_model_id, normalise_model
from ndcb.score import score_dir


def test_two_protocols_with_the_documented_settings():
    assert PROTOCOLS["paper"].max_tokens == 500
    assert PROTOCOLS["paper"].judge == "claude-opus-4-8"
    assert PROTOCOLS["latest"].max_tokens == 2000
    assert PROTOCOLS["latest"].judge == "claude-fable-5-2"
    assert protocols.DEFAULT_PROTOCOL == "latest"


@pytest.mark.parametrize("protocol,api,expected", [
    ("paper", "bedrock", "us.anthropic.claude-opus-4-8"),
    ("paper", "anthropic", "claude-opus-4-8"),
    ("latest", "bedrock", "us.anthropic.claude-fable-5-2"),
    ("latest", "anthropic", "claude-fable-5-2"),
])
def test_judge_id_per_backend(protocol, api, expected):
    assert judge_model_id(protocol, api) == expected


def test_openai_backend_cannot_be_a_protocol_judge():
    with pytest.raises(ValueError, match="--judge-model"):
        judge_model_id("paper", "openai")


def test_unknown_protocol():
    with pytest.raises(ValueError, match="unknown protocol"):
        protocols.get("v2")


@pytest.mark.parametrize("model_id", [
    "us.anthropic.claude-opus-4-8", "global.anthropic.claude-opus-4-8",
    "eu.anthropic.claude-opus-4-8", "anthropic.claude-opus-4-8", "claude-opus-4-8",
])
def test_judge_ids_normalise_across_providers(model_id):
    assert normalise_model(model_id) == "claude-opus-4-8"


@pytest.mark.parametrize("tokens,judges,expected", [
    ([500], ["us.anthropic.claude-opus-4-8"], "paper"),
    ([500], ["claude-opus-4-8"], "paper"),                 # Anthropic API judge
    ([2000], ["global.anthropic.claude-fable-5-2"], "latest"),
    ([2000], ["us.anthropic.claude-opus-4-8"], "custom"),  # Opus 5.5's setup
    ([500], ["us.anthropic.claude-fable-5-2"], "custom"),
    ([500, 2000], ["claude-opus-4-8"], "custom"),           # mixed limits
    ([500], ["claude-opus-4-8", "claude-fable-5-2"], "custom"),
    ([None], ["claude-opus-4-8"], "custom"),               # limit unknown
    ([1000], ["claude-opus-4-8"], "custom"),
])
def test_classify(tokens, judges, expected):
    assert classify(tokens, judges) == expected


# --- CLI: the protocol supplies defaults, explicit flags override ---------- #
def _parse(*argv):
    return build_parser().parse_args(list(argv))


def test_cli_run_defaults_to_latest_and_leaves_tokens_to_the_protocol():
    a = _parse("run", "--model", "m", "--out", "o")
    assert a.protocol == "latest" and a.max_tokens is None


def test_cli_accepts_paper_protocol_on_run_and_judge():
    assert _parse("run", "--model", "m", "--out", "o", "--protocol", "paper").protocol == "paper"
    assert _parse("judge", "--out", "o", "--protocol", "paper").protocol == "paper"


def test_cli_rejects_unknown_protocol():
    with pytest.raises(SystemExit):
        _parse("run", "--model", "m", "--out", "o", "--protocol", "v2")


def test_cli_run_resolves_protocol_settings(monkeypatch):
    from ndcb import cli
    seen = {}
    monkeypatch.setattr("ndcb.run.generate", lambda **kw: seen.update(kw) or {})
    cli.main(["run", "--model", "m", "--out", "o", "--protocol", "paper"])
    assert seen["max_tokens"] == 500 and seen["protocol"] == "paper"
    cli.main(["run", "--model", "m", "--out", "o", "--max-tokens", "750"])
    assert seen["max_tokens"] == 750 and seen["protocol"] == "latest"


def test_cli_judge_resolves_protocol_judge(monkeypatch):
    from ndcb import cli
    seen = {}
    monkeypatch.setattr("ndcb.run.judge", lambda **kw: seen.update(kw) or {})
    cli.main(["judge", "--out", "o", "--protocol", "paper", "--api", "anthropic"])
    assert seen["judge_model"] == "claude-opus-4-8"
    cli.main(["judge", "--out", "o"])
    assert seen["judge_model"] == "us.anthropic.claude-fable-5-2"
    cli.main(["judge", "--out", "o", "--judge-model", "x"])
    assert seen["judge_model"] == "x"


# --- scorecards record the protocol --------------------------------------- #
def _write_run(d: Path, max_tokens, judge):
    for f in ("normative", "descriptive"):
        gens, evals = [], []
        for i in range(3):
            for r in range(3):
                key = {"scenario_id": f"S{i}", "run": r}
                gens.append({**key, "ok": True, "response": "x", "max_tokens": max_tokens})
                evals.append({**key, "ok": True, "judge_model": judge,
                              "judge_response": "<direction>equal</direction>"})
        for name, rows in ((f"hiring_{f}_results.jsonl", gens),
                           (f"hiring_{f}_bias_eval.jsonl", evals)):
            (d / name).write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.mark.parametrize("max_tokens,judge,expected", [
    (500, "us.anthropic.claude-opus-4-8", "paper"),
    (2000, "us.anthropic.claude-fable-5-2", "latest"),
    (2000, "us.anthropic.claude-opus-4-8", "custom"),
])
def test_scorecard_records_protocol(tmp_path, max_tokens, judge, expected):
    _write_run(tmp_path, max_tokens, judge)
    sc = score_dir(tmp_path, ci_method="wilson")
    assert sc["protocol"] == expected
    assert sc["generation_max_tokens"] == [max_tokens]


def test_frozen_paper_cards_are_labelled_paper():
    root = Path(__file__).resolve().parents[1] / "results" / "paper"
    cards = [json.loads(p.read_text()) for p in root.glob("*.json")]
    assert cards and all(c["protocol"] == "paper" for c in cards)
