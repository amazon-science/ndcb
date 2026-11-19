"""Generation and judging passes.

The paper's evaluation procedure, with new generation and judge defaults:

* every one of the 697 scenarios is run under 2 framings (normative,
  descriptive) x 3 runs = 4,182 generations per model;
* each generation is then labelled by the judge = 4,182 judge calls;
* candidate A (the privileged member) is always presented first -- the
  headline numbers are **not** order-counterbalanced;
* models under test: the protocol's settings (see `ndcb.protocols`):
  `latest` is 2,000 tokens at temperature 0 for every model; `paper` uses
  each model's paper settings. Reasoning models that
  reject a sampling parameter run at API defaults (the client drops
  temperature on rejection and says so in the record);
* judge: Claude Fable 5.2 on Bedrock, API defaults, max_tokens 400.
  The paper used Claude Opus 4.8; pass
  judge_model="us.anthropic.claude-opus-4-8" explicitly to use that judge.

Both passes are resumable. Records are appended to JSONL keyed by
``(scenario_id, framing, run)``; re-running skips keys already present, so an
interrupted sweep continues where it stopped instead of re-billing completed
calls.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .client import ChatClient, map_threaded
from .dataset import load_prompts, load_scenarios
from .metrics import DOMAINS, parse_direction
from .prompts import JUDGE_MAX_TOKENS, MODEL_MAX_TOKENS, get_bias_judge_prompt
from .protocols import DEFAULT_PROTOCOL, judge_model_id

#: Judge of the default protocol on Bedrock (see `ndcb.protocols`).
#: The paper protocol's judge is Claude Opus 4.8.
DEFAULT_JUDGE_MODEL = judge_model_id(DEFAULT_PROTOCOL, "bedrock")
#: Runs per scenario per framing.
DEFAULT_N_RUNS = 3
FRAMINGS = ("normative", "descriptive")


def _results_path(out_dir: Path, domain: str, framing: str) -> Path:
    return out_dir / f"{domain}_{framing}_results.jsonl"


def _eval_path(out_dir: Path, domain: str, framing: str) -> Path:
    return out_dir / f"{domain}_{framing}_bias_eval.jsonl"


def read_jsonl(path: Path) -> List[Dict]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _is_ok(record: Dict) -> bool:
    # Records from the paper's own runs predate the `ok` field; they are all
    # successful calls, so a missing field means ok.
    return record.get("ok", True) is not False


def latest_ok(records: List[Dict]) -> List[Dict]:
    """One successful record per (scenario_id, run), the latest one.

    The files are append-only, so a key retried after a failure appears more
    than once. Failed calls are kept on disk for inspection but are never
    returned here -- they must not be judged or scored as verdicts.
    """
    keep: Dict[Tuple[str, int], Dict] = {}
    for r in records:
        if _is_ok(r):
            keep[(r["scenario_id"], r.get("run", 0))] = r
    return list(keep.values())


def _done_keys(path: Path) -> Set[Tuple[str, int]]:
    """Keys that already have a successful record. Failed keys are retried."""
    return {(r["scenario_id"], r.get("run", 0)) for r in latest_ok(read_jsonl(path))}


class PreflightError(RuntimeError):
    """The first call of a pass failed in a way that will fail every call."""


def _hint(error: str) -> str:
    if "AccessDenied" in error or "not authorized" in error:
        return ("\n  hint: this account/region has no access to the model. "
                "Try another --region (the `us.` inference profiles work from "
                "any US region you have access in), or --profile.")
    if "HTTP 401" in error:
        return "\n  hint: check the API key for this provider."
    if "HTTP 404" in error:
        return "\n  hint: check --model and --base-url."
    return ""


def _run_pass(one, todo: List, workers: int, desc: str):
    """Make the first call alone; abort if it fails, otherwise fan out.

    A wrong region, a missing model grant or a bad key fails every call the
    same way. Failing on the first one costs one request instead of thousands
    and keeps the output files clean.
    """
    first = one(todo[0])
    if first is not None and not first.ok and first.error != "empty completion":
        raise PreflightError(
            f"{desc}: first call failed, aborting before the remaining "
            f"{len(todo) - 1} calls.\n  {first.error}" + _hint(first.error))
    if len(todo) > 1:
        map_threaded(one, todo[1:], workers=workers, desc=desc)


class _Appender:
    """Append-only JSONL writer that flushes every record, so a kill -9
    mid-sweep loses at most one call."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "a", encoding="utf-8")

    def write(self, record: Dict):
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())

    def close(self):
        self._f.close()


# --- generation pass ------------------------------------------------------- #
def generate(model: str, out_dir: Path, api: str = "bedrock",
             domains: Iterable[str] = DOMAINS, n_runs: int = DEFAULT_N_RUNS,
             framings: Iterable[str] = FRAMINGS,
             temperature: Optional[float] = 0.0,
             max_tokens: Optional[int] = MODEL_MAX_TOKENS, workers: int = 8,
             region: Optional[str] = None, profile: Optional[str] = None,
             base_url: Optional[str] = None, api_key: Optional[str] = None,
             extra_body: Optional[Dict] = None,
             limit: Optional[int] = None,
             protocol: Optional[str] = None) -> Dict[str, int]:
    """Run the model under test over the benchmark. Returns per-arm counts.

    `limit` caps scenarios per domain -- for smoke tests only. A limited run is
    not comparable to the paper and `score` will label it as partial.
    """
    out_dir = Path(out_dir)
    prompts = load_prompts()
    scenarios = load_scenarios()
    by_domain: Dict[str, List[Dict]] = {d: [] for d in DOMAINS}
    for s in scenarios:
        by_domain.setdefault(s["domain"], []).append(s)

    client = ChatClient(model=model, api=api, temperature=temperature,
                        max_tokens=max_tokens, region=region, profile=profile,
                        base_url=base_url, api_key=api_key,
                        extra_body=extra_body)
    counts: Dict[str, int] = {}
    try:
        for domain in domains:
            rows = by_domain.get(domain, [])
            if limit:
                rows = rows[:limit]
            for framing in framings:
                path = _results_path(out_dir, domain, framing)
                done = _done_keys(path)
                todo = [(s, r) for s in rows for r in range(n_runs)
                        if (s["scenario_id"], r) not in done]
                counts[f"{domain}/{framing}"] = len(todo)
                if not todo:
                    continue
                writer = _Appender(path)

                def one(item):
                    scenario, run_idx = item
                    key = (scenario["scenario_id"], framing)
                    if key not in prompts:
                        return None
                    prompt = prompts[key]["prompt"]
                    result = client.complete(prompt)
                    writer.write({
                        "scenario_id": scenario["scenario_id"],
                        "domain": domain,
                        "framing": framing,
                        "run": run_idx,
                        "model": model,
                        "api": api,
                        "temperature": client.temperature,
                        "max_tokens": max_tokens,
                        "protocol": protocol,
                        "question": prompt,
                        "response": result.text,
                        "stop_reason": result.stop_reason,
                        "ok": result.ok,
                        "error": result.error,
                    })
                    return result

                try:
                    _run_pass(one, todo, workers, f"{model} {domain}/{framing}")
                finally:
                    writer.close()
    finally:
        client.close()
    return counts


# --- judging pass ---------------------------------------------------------- #
def judge(model: str, out_dir: Path, judge_model: str = DEFAULT_JUDGE_MODEL,
          judge_api: str = "bedrock", domains: Iterable[str] = DOMAINS,
          framings: Iterable[str] = FRAMINGS, workers: int = 8,
          region: Optional[str] = None,
          profile: Optional[str] = None, base_url: Optional[str] = None,
          api_key: Optional[str] = None, swapped: bool = False,
          protocol: Optional[str] = None) -> Dict[str, int]:
    """Label every generation with the judge rubric. Returns per-arm counts.

    The judge runs at API defaults (`temperature=None`) with a 400-token
    limit. Sampling and thinking parameters are omitted, so the judge runs
    at its default inference settings.
    """
    out_dir = Path(out_dir)
    client = ChatClient(model=judge_model, api=judge_api, temperature=None,
                        max_tokens=JUDGE_MAX_TOKENS, region=region,
                        profile=profile, base_url=base_url, api_key=api_key)
    counts: Dict[str, int] = {}
    try:
        for domain in domains:
            for framing in framings:
                results = latest_ok(read_jsonl(_results_path(out_dir, domain, framing)))
                results = [r for r in results if r.get("response", "").strip()]
                path = _eval_path(out_dir, domain, framing)
                done = _done_keys(path)
                todo = [r for r in results
                        if (r["scenario_id"], r.get("run", 0)) not in done]
                counts[f"{domain}/{framing}"] = len(todo)
                if not todo:
                    continue
                writer = _Appender(path)

                def one(record):
                    prompt = get_bias_judge_prompt(
                        record["question"], record["response"], swapped=swapped)
                    verdict = client.complete(prompt)
                    writer.write({
                        "scenario_id": record["scenario_id"],
                        "domain": domain,
                        "framing": framing,
                        "run": record.get("run", 0),
                        "model": record.get("model", model),
                        "judge_model": judge_model,
                        "max_tokens": record.get("max_tokens"),
                        "protocol": protocol,
                        "question": record["question"],
                        "response": record["response"],
                        "judge_response": verdict.text,
                        "direction": parse_direction(verdict.text),
                        "ok": verdict.ok,
                        "error": verdict.error,
                    })
                    return verdict

                try:
                    _run_pass(one, todo, workers, f"judge {domain}/{framing}")
                finally:
                    writer.close()
    finally:
        client.close()
    return counts


def load_judged(out_dir: Path, domain: str, framing: str) -> List[Dict]:
    """Successful judged records for one arm, ready for `compute_leakage_metrics`.

    Failed judge calls are excluded, so a run whose judge calls failed shows up
    as missing data -- and is flagged as partial -- instead of being scored.
    """
    return latest_ok(read_jsonl(_eval_path(Path(out_dir), domain, framing)))


def count_failed(out_dir: Path, domain: str, framing: str) -> int:
    """Judge keys with no successful record (every attempt failed)."""
    records = read_jsonl(_eval_path(Path(out_dir), domain, framing))
    ok = {(r["scenario_id"], r.get("run", 0)) for r in latest_ok(records)}
    return len({(r["scenario_id"], r.get("run", 0)) for r in records} - ok)
