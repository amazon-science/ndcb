"""Exercise the batch runner with a fake CLI; never call a model."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from ndcb.cli import build_parser


@pytest.fixture
def batch(tmp_path):
    repo = tmp_path / "repository with spaces"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / "evaluate_bedrock_models.sh"
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "scripts" / script.name, script)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "ndcb"
    fake.write_text(f"#!{sys.executable}\n" + '''
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
with open(os.environ["BATCH_TEST_CALLS"], "a") as output:
    output.write(json.dumps(args) + "\\n")
if (args[0] == "run" and os.environ.get("BATCH_TEST_FATAL") and
        "mistral.mistral-large-3-675b-instruct" in args):
    print("AccessDeniedException: simulated model access failure", flush=True)
    sys.exit(9)
if args[0] == "judge":
    print("judging pass complete; one failed call remains recorded", flush=True)
if args[0] == "score":
    path = Path(args[args.index("--json") + 1])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "model_id": args[args.index("--model-id") + 1],
        "completeness": {"paper_comparable": False, "failed_judge_calls": 1},
    }))
    print("PARTIAL RUN -- 1 judge call failed and was not scored", flush=True)
''')
    fake.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
               BATCH_TEST_CALLS=str(calls))
    for name in ["NDCB_WORKERS", "NDCB_RUN_SUFFIX", "NDCB_JUDGE_REGION",
                 "BATCH_TEST_FATAL"]:
        env.pop(name, None)
    return script, repo, calls, env


def invoke(batch, *args):
    script, repo, calls, env = batch
    result = subprocess.run(["bash", str(script), *args], cwd=repo.parent,
                            env=env, capture_output=True, text=True)
    commands = [json.loads(line) for line in calls.read_text().splitlines()] \
        if calls.exists() else []
    return result, commands


def test_dry_run_has_no_commands_or_outputs(batch):
    result, commands = invoke(batch, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert commands == []
    assert not (batch[1] / "runs").exists()
    assert not (batch[1] / "results").exists()
    assert result.stdout.count("$ ndcb run ") == 4
    assert result.stdout.count("$ ndcb judge ") == 4
    assert result.stdout.count("$ ndcb score ") == 4


def test_batch_scores_partial_runs_and_continues_in_order(batch):
    batch[3].update(NDCB_WORKERS="2", NDCB_JUDGE_REGION="us-west-2",
                    NDCB_RUN_SUFFIX="replicate-2")
    result, commands = invoke(batch)
    assert result.returncode == 0, result.stderr
    assert [command[0] for command in commands] == ["download"] + [
        "run", "judge", "score"] * 4
    assert commands[0] == ["download", "--check"]
    model_ids = ["openai.gpt-oss-120b-1:0",
                 "mistral.mistral-large-3-675b-instruct", "qwen.qwen3-32b-v1:0",
                 "us.openai.gpt-6-astra"]
    parser = build_parser()
    for offset, model_id in zip(range(1, len(commands), 3), model_ids):
        generation, judge, score = [
            parser.parse_args(command) for command in commands[offset:offset + 3]]
        assert generation.model == score.model_id == model_id
        assert generation.api == judge.api == "bedrock"
        assert generation.max_tokens == 2000
        assert generation.temperature == 0 and not generation.no_temperature
        assert generation.runs == 3 and generation.limit is None
        assert generation.domains is None and generation.extra_body is None
        assert generation.workers == judge.workers == 2
        assert judge.judge_model == "us.anthropic.claude-fable-5-2"
        assert judge.region == "us-west-2"
        assert generation.out == judge.out == score.out
        assert score.out.endswith("-t2000-judge-fable52-replicate-2")
        assert score.ci_method == "bootstrap" and score.bootstrap_iters == 10000
        assert score.seed == 42
        card = json.loads((batch[1] / score.json).read_text())
        assert card["completeness"]["failed_judge_calls"] == 1
        assert "PARTIAL RUN" in (batch[1] / score.out / "score.log").read_text()
    assert len(list((batch[1] / "results/submissions").glob("*.json"))) == 4


def test_fatal_command_stops_before_next_stage(batch):
    batch[3]["BATCH_TEST_FATAL"] = "1"
    result, commands = invoke(batch)
    assert result.returncode == 9
    assert [command[0] for command in commands] == [
        "download", "run", "judge", "score", "run"]
    assert "mistral.mistral-large-3-675b-instruct" in commands[-1]
    assert "Batch stopped" in result.stderr
    assert len(list((batch[1] / "results/submissions").glob("*.json"))) == 1
