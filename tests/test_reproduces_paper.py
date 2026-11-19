"""The reproducibility gate.

The frozen scorecards in `results/paper/` were produced by this package's
scorer from the paper's own run data. These tests assert they still match the
values printed in the publication, so a change to the scorer that would alter
a published number fails here instead of silently shipping.

The paper's Table (tab:leakage_metrics_ci) reports Avg. NDS to two decimals
with a 95% cluster-bootstrap CI.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ndcb.metrics import DOMAINS

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = REPO_ROOT / "results" / "paper"

# model -> (avg NDS, ci_lo, ci_hi, profile), exactly as published.
PUBLISHED = {
    "Claude Sonnet 4.6": (+0.55, +0.51, +0.59, "calibration"),
    "Claude Opus 4.6": (+0.52, +0.49, +0.56, "calibration"),
    "GPT-OSS 120B": (+0.43, +0.40, +0.46, "divergence"),
    "Mistral Large 3": (+0.42, +0.38, +0.46, "divergence"),
    "Qwen3-32B": (+0.06, +0.02, +0.10, "divergence"),
    "GPT-5.5": (-0.14, -0.17, -0.12, "convergence"),
}

PUBLISHED_JUDGE = "us.anthropic.claude-opus-4-8"


def _cards():
    cards = {}
    for path in PAPER_DIR.glob("*.json"):
        with open(path, encoding="utf-8") as f:
            card = json.load(f)
        cards[card["model"]] = card
    return cards


@pytest.fixture(scope="module")
def cards():
    found = _cards()
    if not found:
        pytest.skip(f"no frozen scorecards in {PAPER_DIR}")
    return found


def test_all_six_paper_models_present(cards):
    assert set(cards) == set(PUBLISHED)


@pytest.mark.parametrize("model", sorted(PUBLISHED))
def test_avg_nds_matches_publication(cards, model):
    if model not in cards:
        pytest.skip(f"{model} scorecard absent")
    expected, lo, hi, _profile = PUBLISHED[model]
    card = cards[model]
    assert round(card["avg_NDS"], 2) == pytest.approx(expected), (
        f"{model}: scorer gives {card['avg_NDS']:+.4f}, paper says {expected:+.2f}")
    got_lo, got_hi = card["avg_NDS_ci"]
    assert round(got_lo, 2) == pytest.approx(lo)
    assert round(got_hi, 2) == pytest.approx(hi)


@pytest.mark.parametrize("model", sorted(PUBLISHED))
def test_profile_matches_publication(cards, model):
    if model not in cards:
        pytest.skip(f"{model} scorecard absent")
    assert cards[model]["profile"] == PUBLISHED[model][3]


def test_published_ranking_order(cards):
    """The paper's ordering, Sonnet > Opus > GPT-OSS > Mistral > Qwen > GPT-5.5."""
    order = [m for m, _ in sorted(cards.items(),
                                  key=lambda kv: kv[1]["avg_NDS"], reverse=True)]
    assert order == ["Claude Sonnet 4.6", "Claude Opus 4.6", "GPT-OSS 120B",
                     "Mistral Large 3", "Qwen3-32B", "GPT-5.5"]


def test_top_two_intervals_overlap(cards):
    """The paper explicitly declines to separate ranks 1 and 2; keep that true."""
    a = cards["Claude Sonnet 4.6"]["avg_NDS_ci"]
    b = cards["Claude Opus 4.6"]["avg_NDS_ci"]
    assert a[0] <= b[1] and b[0] <= a[1]


@pytest.mark.parametrize("model", sorted(PUBLISHED))
def test_nds_identity_holds_in_frozen_cards(cards, model):
    if model not in cards:
        pytest.skip(f"{model} scorecard absent")
    card = cards[model]
    assert card["avg_NDS"] == pytest.approx(card["avg_NER"] - card["avg_NLR"])
    for metrics in card["per_domain"].values():
        assert metrics["NDS"] == pytest.approx(metrics["NER"] - metrics["NLR"])


@pytest.mark.parametrize("model", sorted(PUBLISHED))
def test_avg_is_unweighted_mean_over_domains(cards, model):
    """Not scenario-pooled. Domains have unequal n, so this distinction matters."""
    if model not in cards:
        pytest.skip(f"{model} scorecard absent")
    card = cards[model]
    per_domain = [card["per_domain"][d]["NDS"] for d in DOMAINS]
    assert card["avg_NDS"] == pytest.approx(sum(per_domain) / len(per_domain))


@pytest.mark.parametrize("model", sorted(PUBLISHED))
def test_paper_runs_are_complete(cards, model):
    if model not in cards:
        pytest.skip(f"{model} scorecard absent")
    completeness = cards[model]["completeness"]
    assert completeness["paper_comparable"] is True
    assert completeness["scenarios_scored"] == 697
    assert completeness["domains_scored"] == 6


@pytest.mark.parametrize("model", sorted(PUBLISHED))
def test_judge_and_ci_provenance_recorded(cards, model):
    if model not in cards:
        pytest.skip(f"{model} scorecard absent")
    card = cards[model]
    assert card["judge_model"] == PUBLISHED_JUDGE
    assert card["ci_method"] == "bootstrap"
    assert card["bootstrap_iters"] == 10000
    assert card["bootstrap_seed"] == 42
