"""`ndcb` command line interface.

    ndcb download                 fetch or copy the benchmark data
    ndcb run                      generate responses from a model under test
    ndcb judge                    label those responses with the judge rubric
    ndcb score                    compute NER / NLR / NDS and CIs
    ndcb leaderboard              re-render the leaderboard page
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .dataset import (CHECKSUMS, DATA_FILES, cache_dir, candidate_dirs,
                      data_dir, download, load_prompts, load_scenarios, verify)
from .metrics import (DEFAULT_ALPHA, DEFAULT_BOOTSTRAP_ITERS,
                      DEFAULT_BOOTSTRAP_SEED, DOMAINS)
from .prompts import JUDGE_MAX_TOKENS, MODEL_MAX_TOKENS
from .protocols import DEFAULT_PROTOCOL, PROTOCOLS, judge_model_id
from .run import DEFAULT_N_RUNS


def _cmd_download(args) -> int:
    if args.check:
        try:
            d = data_dir()
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 1
        status = verify(d)
        print(f"data dir: {d}")
        for f, ok in status.items():
            print(f"  {'ok  ' if ok else 'BAD '} {f}")
        return 0 if all(status.values()) else 1

    dest = download(dest=args.dest, from_dir=args.from_dir,
                    base_url=args.base_url, check=not args.no_verify)
    print(f"benchmark data ready in {dest}")
    for f in DATA_FILES:
        print(f"  {f}  sha256={CHECKSUMS[f][:16]}...")
    return 0


def _cmd_info(args) -> int:
    print(f"ndcb {__version__}")
    print("searched for data in:")
    for d in candidate_dirs():
        marker = "*" if all((d / f).is_file() for f in DATA_FILES) else " "
        print(f"  {marker} {d}")
    print(f"cache dir: {cache_dir()}")
    try:
        scenarios = load_scenarios()
        prompts = load_prompts()
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"scenarios: {len(scenarios)}   prompts: {len(prompts)}")
    counts: dict = {}
    for s in scenarios:
        counts[s["domain"]] = counts.get(s["domain"], 0) + 1
    for d in DOMAINS:
        print(f"  {d:<20}{counts.get(d, 0)}")
    return 0


def _cmd_run(args) -> int:
    from .run import generate
    extra = json.loads(args.extra_body) if args.extra_body else None
    max_tokens = args.max_tokens or PROTOCOLS[args.protocol].max_tokens
    print(f"protocol {args.protocol}: max_tokens={max_tokens}")
    counts = generate(
        model=args.model, out_dir=args.out, api=args.api,
        domains=args.domains or DOMAINS, n_runs=args.runs,
        temperature=None if args.no_temperature else args.temperature,
        max_tokens=max_tokens, workers=args.workers,
        region=args.region, profile=args.profile, base_url=args.base_url,
        api_key=args.api_key, extra_body=extra, limit=args.limit,
        protocol=args.protocol)
    total = sum(counts.values())
    print(f"generation pass complete; {total} new calls made into {args.out}")
    return 0


def _cmd_judge(args) -> int:
    from .run import judge
    judge_model = args.judge_model or judge_model_id(args.protocol, args.api)
    print(f"protocol {args.protocol}: judge={judge_model}")
    counts = judge(
        model=args.model or "", out_dir=args.out, judge_model=judge_model,
        judge_api=args.api, domains=args.domains or DOMAINS,
        workers=args.workers, region=args.region, profile=args.profile,
        base_url=args.base_url, api_key=args.api_key, swapped=args.swapped,
        protocol=args.protocol)
    total = sum(counts.values())
    print(f"judging pass complete; {total} new judge calls made into {args.out}")
    return 0


def _cmd_score(args) -> int:
    from .score import format_scorecard, score_dir, write_scorecard
    sc = score_dir(args.out, model=args.model or "", ci_method=args.ci_method,
                   alpha=args.alpha, n_boot=args.bootstrap_iters,
                   seed=args.seed, judge_model=args.judge_model)
    if args.model_id:
        sc["model_id"] = args.model_id
    sc["source"] = args.source
    print(format_scorecard(sc))
    if args.json:
        path = write_scorecard(sc, args.json)
        print(f"\nwrote {path}")
    return 0


def _cmd_leaderboard(args) -> int:
    from .leaderboard import build
    root = Path(args.repo_root)
    statement = None
    if args.statement_file:
        statement = Path(args.statement_file).read_text(encoding="utf-8").strip()
    page = build(root, statement=statement)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}")
    return 0


def _add_protocol_arg(p: argparse.ArgumentParser):
    p.add_argument("--protocol", default=DEFAULT_PROTOCOL, choices=list(PROTOCOLS),
                   help="paper = as published (500 tokens, Claude Opus 4.8 "
                        "judge); latest = default for new evaluations (2000 "
                        "tokens, Claude Fable 5.2 judge). Use the same value "
                        "for `run` and `judge`.")


def _add_provider_args(p: argparse.ArgumentParser, judge: bool = False):
    p.add_argument("--api", default="bedrock",
                   choices=["bedrock", "openai", "anthropic"],
                   help="backend (default: bedrock, as used for the paper)")
    p.add_argument("--region", default=None,
                   help="Bedrock region where your account has model access "
                        "(default: $AWS_REGION, then your AWS profile's region)")
    p.add_argument("--profile", default=None, help="AWS profile for Bedrock")
    p.add_argument("--base-url", default=None, help="OpenAI/Anthropic API root")
    p.add_argument("--api-key", default=None, help="OpenAI/Anthropic API key")
    p.add_argument("--workers", type=int, default=8,
                   help="concurrent requests (default: 8)")
    p.add_argument("--domains", nargs="*", choices=DOMAINS, default=None,
                   help="restrict to these domains")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ndcb",
        description="Normative-Descriptive Calibration Benchmark: data, "
                    "evaluation harness and NDS scorer.")
    parser.add_argument("--version", action="version", version=f"ndcb {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="fetch or copy the benchmark data")
    p.add_argument("--dest", default=None, help=f"target dir (default: {cache_dir()})")
    p.add_argument("--from-dir", default=None,
                   help="copy from a local directory instead of downloading")
    p.add_argument("--base-url", default=None,
                   help="HTTP base URL hosting the data files "
                        "(default: $NDCB_DATA_URL, then the GitHub release)")
    p.add_argument("--no-verify", action="store_true", help="skip checksum check")
    p.add_argument("--check", action="store_true",
                   help="only verify the data already present")
    p.set_defaults(func=_cmd_download)

    p = sub.add_parser("info", help="show data location and composition")
    p.set_defaults(func=_cmd_info)

    p = sub.add_parser("run", help="generate responses from a model under test")
    p.add_argument("--model", required=True, help="provider model id")
    p.add_argument("--out", required=True, help="output directory for this run")
    p.add_argument("--runs", type=int, default=DEFAULT_N_RUNS,
                   help=f"runs per scenario per framing (default: {DEFAULT_N_RUNS})")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--no-temperature", action="store_true",
                   help="omit temperature entirely (reasoning models)")
    _add_protocol_arg(p)
    p.add_argument("--max-tokens", type=int, default=None,
                   help="generation token limit (default: the protocol's -- "
                        "paper 500, latest 2000). Overriding it makes the "
                        "score 'custom'.")
    p.add_argument("--extra-body", default=None,
                   help='provider passthrough as JSON, e.g. \'{"reasoning_effort":"medium"}\'')
    p.add_argument("--limit", type=int, default=None,
                   help="cap scenarios per domain (smoke tests; NOT paper-comparable)")
    _add_provider_args(p)
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("judge", help="label responses with the judge rubric")
    p.add_argument("--out", required=True, help="run directory from `ndcb run`")
    p.add_argument("--model", default=None, help="model under test (for records)")
    _add_protocol_arg(p)
    p.add_argument("--judge-model", default=None,
                   help="judge model id (default: the protocol's -- paper "
                        "Claude Opus 4.8, latest Claude Fable 5.2). Another "
                        "judge makes the score 'custom'.")
    p.add_argument("--swapped", action="store_true",
                   help="responses come from the order-reversed control")
    _add_provider_args(p, judge=True)
    p.set_defaults(func=_cmd_judge)

    p = sub.add_parser("score", help="compute NER / NLR / NDS with CIs")
    p.add_argument("--out", required=True, help="run directory to score")
    p.add_argument("--model", default=None, help="display name for the scorecard")
    p.add_argument("--model-id", default=None, help="provider model id to record")
    p.add_argument("--judge-model", default=None, help="override recorded judge")
    p.add_argument("--source", default="submission",
                   choices=["paper", "submission"],
                   help="which leaderboard section this belongs to")
    p.add_argument("--ci-method", default="bootstrap",
                   choices=["bootstrap", "wilson"],
                   help="bootstrap reproduces the paper (default)")
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--bootstrap-iters", type=int, default=DEFAULT_BOOTSTRAP_ITERS)
    p.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    p.add_argument("--json", default=None, help="also write the scorecard here")
    p.set_defaults(func=_cmd_score)

    p = sub.add_parser("leaderboard", help="re-render the leaderboard page")
    p.add_argument("--repo-root", default=".", help="repo root holding results/")
    p.add_argument("--output", default="docs/leaderboard.md")
    p.add_argument("--statement-file", default=None,
                   help="append the contents of this file as a footer")
    p.set_defaults(func=_cmd_leaderboard)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted; progress is saved and `ndcb run`/`ndcb judge` "
              "will resume where they stopped", file=sys.stderr)
        return 130
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
