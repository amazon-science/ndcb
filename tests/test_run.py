"""Resume, failure handling and scoring of partial runs -- offline."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ndcb import run as runmod
from ndcb.cli import main
from ndcb.client import Completion
from ndcb.run import PreflightError, _done_keys, _run_pass, latest_ok, load_judged
from ndcb.score import score_dir


def _write(path: Path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.mark.parametrize("entrypoint", ["cli", "library"])
@pytest.mark.parametrize("max_tokens", [None, 500])
@pytest.mark.parametrize("judge_model", [None, "us.anthropic.claude-opus-4-8"])
def test_generation_and_judge_settings(entrypoint, max_tokens, judge_model,
                                       monkeypatch, tmp_path):
    """Defaults and paper overrides must reach clients and saved records."""
    configs = []

    class RecordingClient:
        def __init__(self, **kwargs):
            configs.append(kwargs)
            self.temperature = kwargs["temperature"]

        def complete(self, prompt):
            return Completion(True, text="<direction>equal</direction>",
                              stop_reason="end_turn")

        def close(self):
            pass

    monkeypatch.setattr(runmod, "ChatClient", RecordingClient)
    expected = 2000 if max_tokens is None else max_tokens
    expected_judge = judge_model or "us.anthropic.claude-fable-5-2"
    if entrypoint == "cli":
        extra = [] if max_tokens is None else ["--max-tokens", str(max_tokens)]
        assert main(["run", "--model", "m", "--out", str(tmp_path),
                     "--domains", "hiring", "--limit", "1", "--runs", "1",
                     "--workers", "1"] + extra) == 0
        judge_extra = [] if judge_model is None else ["--judge-model", judge_model]
        assert main(["judge", "--out", str(tmp_path),
                     "--domains", "hiring", "--workers", "1"] + judge_extra) == 0
    else:
        extra = {} if max_tokens is None else {"max_tokens": max_tokens}
        runmod.generate("m", tmp_path, domains=["hiring"], limit=1,
                        n_runs=1, workers=1, **extra)
        judge_extra = {} if judge_model is None else {"judge_model": judge_model}
        runmod.judge("m", tmp_path, domains=["hiring"], workers=1, **judge_extra)

    assert [config["max_tokens"] for config in configs] == [expected, 400]
    assert [config["temperature"] for config in configs] == [0.0, None]
    assert configs[1]["model"] == expected_judge
    assert configs[1]["api"] == "bedrock"
    rows = [r for path in tmp_path.glob("*_results.jsonl")
            for r in runmod.read_jsonl(path)]
    assert len(rows) == 2
    assert all(row["max_tokens"] == expected for row in rows)
    judgments = [r for path in tmp_path.glob("*_bias_eval.jsonl")
                 for r in runmod.read_jsonl(path)]
    assert len(judgments) == 2
    assert all(row["judge_model"] == expected_judge for row in judgments)
    assert score_dir(tmp_path, ci_method="wilson")["judge_model"] == expected_judge


def test_latest_ok_drops_failures_and_keeps_retry():
    rows = [
        {"scenario_id": "S1", "run": 0, "ok": False, "error": "denied"},
        {"scenario_id": "S1", "run": 0, "ok": True, "judge_response": "late"},
        {"scenario_id": "S2", "run": 0, "ok": False},
    ]
    out = latest_ok(rows)
    assert len(out) == 1 and out[0]["judge_response"] == "late"


def test_records_without_ok_field_count_as_ok():
    """The paper's own records predate the `ok` field."""
    assert len(latest_ok([{"scenario_id": "S1", "run": 0}])) == 1


def test_failed_keys_are_not_done(tmp_path):
    p = tmp_path / "x.jsonl"
    _write(p, [{"scenario_id": "S1", "run": 0, "ok": False},
               {"scenario_id": "S2", "run": 0, "ok": True}])
    assert _done_keys(p) == {("S2", 0)}


def test_preflight_aborts_on_first_hard_failure(monkeypatch):
    fanned = []
    monkeypatch.setattr(runmod, "map_threaded",
                        lambda *a, **k: fanned.append(1))
    one = lambda item: Completion(False, error="AccessDeniedException: no")
    with pytest.raises(PreflightError, match="aborting before the remaining 2"):
        _run_pass(one, [1, 2, 3], workers=2, desc="judge")
    assert not fanned


def test_preflight_tolerates_empty_completion(monkeypatch):
    fanned = []
    monkeypatch.setattr(runmod, "map_threaded",
                        lambda fn, items, **k: fanned.extend(items))
    _run_pass(lambda i: Completion(False, error="empty completion"),
              [1, 2, 3], workers=2, desc="gen")
    assert fanned == [2, 3]


def test_preflight_fans_out_after_success(monkeypatch):
    fanned = []
    monkeypatch.setattr(runmod, "map_threaded",
                        lambda fn, items, **k: fanned.extend(items))
    _run_pass(lambda i: Completion(True, text="x"), [1, 2, 3], workers=2, desc="gen")
    assert fanned == [2, 3]


def test_all_failed_judge_calls_is_not_scored_as_complete(tmp_path):
    """Regression: 4,182 AccessDenied judge records were scored as a complete
    run with NDS 0.00. They must count as missing instead."""
    for f in ("normative", "descriptive"):
        _write(tmp_path / f"hiring_{f}_bias_eval.jsonl",
               [{"scenario_id": f"S{i}", "run": r, "ok": False,
                 "judge_response": "", "error": "AccessDenied"}
                for i in range(3) for r in range(3)])
    with pytest.raises(ValueError, match="no judged results"):
        score_dir(tmp_path, ci_method="wilson")


def test_partial_failures_mark_run_not_comparable(tmp_path):
    for f in ("normative", "descriptive"):
        rows = [{"scenario_id": f"S{i}", "run": r, "ok": True,
                 "judge_response": "<direction>equal</direction>"}
                for i in range(3) for r in range(3)]
        rows.append({"scenario_id": "S0", "run": 0, "ok": False})  # failed then retried ok
        rows.append({"scenario_id": "S9", "run": 0, "ok": False})  # never succeeded
        _write(tmp_path / f"hiring_{f}_bias_eval.jsonl", rows)
    sc = score_dir(tmp_path, ci_method="wilson")
    c = sc["completeness"]
    assert c["failed_judge_calls"] == 2
    assert c["paper_comparable"] is False
    assert len(load_judged(tmp_path, "hiring", "normative")) == 9


def test_unparseable_verdicts_above_threshold_not_comparable(tmp_path):
    for f in ("normative", "descriptive"):
        _write(tmp_path / f"hiring_{f}_bias_eval.jsonl",
               [{"scenario_id": f"S{i}", "run": r, "ok": True,
                 "judge_response": "no tags at all"}
                for i in range(3) for r in range(3)])
    assert score_dir(tmp_path, ci_method="wilson")["completeness"]["paper_comparable"] is False
