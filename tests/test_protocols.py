"""The paper/latest protocols: resolution, classification, and CLI defaults."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ndcb import protocols
from ndcb.cli import build_parser
from ndcb.protocols import PROTOCOLS, classify, judge_model_id, normalise_model
from ndcb.score import score_dir


def test_two_protocols_with_the_documented_defaults():
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
    assert seen["max_tokens"] == 750 and seen["protocol"] == "custom"


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


# --- per-model paper settings --------------------------------------------- #
@pytest.mark.parametrize("model_id", [
    "us.anthropic.claude-opus-4-6-v1", "claude-opus-4-6",
    "us.anthropic.claude-sonnet-4-6", "global.anthropic.claude-sonnet-4-6",
])
def test_paper_claude_46_runs_with_thinking(model_id):
    g = PROTOCOLS["paper"].generation(model_id)
    assert (g.max_tokens, g.temperature, g.thinking_budget) == (2500, 1.0, 1024)


@pytest.mark.parametrize("model_id", ["openai.gpt-5.5", "gpt-5.5"])
def test_paper_gpt55_has_no_limit_and_medium_effort(model_id):
    g = PROTOCOLS["paper"].generation(model_id)
    assert g.max_tokens is None and g.temperature is None
    assert g.reasoning_effort == "medium"


@pytest.mark.parametrize("model_id", [
    "qwen.qwen3-32b-v1:0", "openai.gpt-oss-120b-1:0",
    "mistral.mistral-large-3-675b-instruct", "us.anthropic.claude-opus-5-5",
])
def test_paper_other_models_use_500_at_temperature_0(model_id):
    g = PROTOCOLS["paper"].generation(model_id)
    assert (g.max_tokens, g.temperature, g.thinking_budget, g.reasoning_effort) == (500, 0.0, None, None)


@pytest.mark.parametrize("model_id", [
    "us.anthropic.claude-opus-4-6-v1", "openai.gpt-5.5", "qwen.qwen3-32b-v1:0",
])
def test_latest_uses_one_setting_for_every_model(model_id):
    g = PROTOCOLS["latest"].generation(model_id)
    assert (g.max_tokens, g.temperature, g.thinking_budget, g.reasoning_effort) == (2000, 0.0, None, None)


def test_extra_body_per_backend():
    t = PROTOCOLS["paper"].generation("claude-opus-4-6")
    assert t.extra_body("bedrock") == {"additionalModelRequestFields": {
        "thinking": {"type": "enabled", "budget_tokens": 1024}}}
    assert t.extra_body("anthropic") == {"thinking": {"type": "enabled", "budget_tokens": 1024}}
    with pytest.raises(ValueError):
        t.extra_body("openai")
    r = PROTOCOLS["paper"].generation("gpt-5.5")
    assert r.extra_body("openai") == {"reasoning_effort": "medium"}
    assert r.extra_body("bedrock") == {"additionalModelRequestFields": {"reasoning_effort": "medium"}}
    assert PROTOCOLS["latest"].generation("x").extra_body("bedrock") is None


def test_cli_paper_protocol_applies_per_model_settings(monkeypatch):
    from ndcb import cli
    seen = {}
    monkeypatch.setattr("ndcb.run.generate", lambda **kw: seen.update(kw) or {})
    cli.main(["run", "--protocol", "paper", "--model", "us.anthropic.claude-sonnet-4-6", "--out", "o"])
    assert seen["max_tokens"] == 2500 and seen["temperature"] == 1.0
    assert seen["extra_body"]["additionalModelRequestFields"]["thinking"]["budget_tokens"] == 1024
    assert seen["protocol"] == "paper"
    cli.main(["run", "--protocol", "paper", "--model", "openai.gpt-5.5", "--out", "o"])
    assert seen["max_tokens"] is None and seen["temperature"] is None
    assert seen["protocol"] == "paper"


@pytest.mark.parametrize("flag", [["--temperature", "0.5"], ["--no-temperature"],
                                  ["--extra-body", "{}"], ["--max-tokens", "500"]])
def test_cli_any_generation_override_is_custom(monkeypatch, flag):
    from ndcb import cli
    seen = {}
    monkeypatch.setattr("ndcb.run.generate", lambda **kw: seen.update(kw) or {})
    cli.main(["run", "--model", "m", "--out", "o", *flag])
    assert seen["protocol"] == "custom"


def test_cli_judge_override_is_custom_unless_it_is_the_protocol_judge(monkeypatch):
    from ndcb import cli
    seen = {}
    monkeypatch.setattr("ndcb.run.judge", lambda **kw: seen.update(kw) or {})
    cli.main(["judge", "--out", "o", "--judge-model", "x"])
    assert seen["protocol"] == "custom"
    cli.main(["judge", "--out", "o", "--protocol", "paper",
              "--judge-model", "global.anthropic.claude-opus-4-8"])
    assert seen["protocol"] == "paper"


@pytest.mark.parametrize("gens,judges,expected", [
    ({"paper"}, {"paper"}, "paper"),
    ({"latest"}, {"latest"}, "latest"),
    ({"latest"}, {"paper"}, "custom"),      # the Opus/Sonnet 4.6 runs
    ({"custom"}, {"custom"}, "custom"),
    ({"paper", "latest"}, {"paper"}, "custom"),
])
def test_from_records(gens, judges, expected):
    assert protocols.from_records(gens, judges, [2000], ["claude-fable-5-2"]) == expected


def test_from_records_falls_back_for_old_records():
    assert protocols.from_records({None}, {None}, [2000], ["claude-fable-5-2"]) == "latest"


# --- client: no token limit ----------------------------------------------- #
def test_no_limit_is_omitted_from_requests(monkeypatch):
    from ndcb.client import ChatClient
    c = ChatClient("gpt-5.5", api="openai", api_key="dummy", max_tokens=None)
    assert "max_completion_tokens" not in c._payload("hi")


def test_anthropic_requires_a_limit():
    from ndcb.client import ChatClient
    with pytest.raises(ValueError, match="token limit"):
        ChatClient("m", api="anthropic", api_key="dummy", max_tokens=None)
