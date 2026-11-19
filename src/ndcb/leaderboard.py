"""Render the leaderboard page from scorecard JSON files.

The page has two sections, and they are deliberately not merged:

**Paper-reported results** -- the six models from the publication. Frozen. These
are regenerated from the paper's own run data by this package's scorer and
match the published table to the reported precision.

**Post-paper submissions** -- anything evaluated afterwards. Separate because a
reproducibility release has to keep the published claim auditable; a newer
model's score is a new measurement, not a correction to the paper.

Ranking caveat carried onto the page: NDS CIs overlap for several pairs, so the
ordering is not a resolved ranking. The paper says so about ranks 1 and 2.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .metrics import DOMAINS


def fmt(value: float, places: int = 2, signed: bool = True) -> str:
    """Format a rate the way the paper's table generator did.

    Deliberately plain Python float formatting. Several cells land on an exact
    tie -- Qwen3-32B customer service is 0.575, Claude Opus 4.6 content
    moderation is 0.725 -- and the published table prints +0.57 and +0.72
    respectively, i.e. whichever way the binary representation of the tie
    falls. Imposing round-half-up here would change two published digits, so
    don't.
    """
    text = f"{value:.{places}f}"
    if signed and not text.startswith("-"):
        text = "+" + text
    return text

PAPER_DIR = "results/paper"
SUBMISSIONS_DIR = "results/submissions"

DOMAIN_LABELS = {
    "criminal_justice": "Crim. Just.",
    "housing": "Housing",
    "hiring": "Hiring",
    "education": "Education",
    "customer_service": "Cust. Svc.",
    "content_moderation": "Cont. Mod.",
}


def load_scorecards(directory: Path) -> List[Dict]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    cards = []
    for p in sorted(directory.glob("*.json")):
        with open(p, encoding="utf-8") as f:
            card = json.load(f)
        card["_path"] = str(p)
        cards.append(card)
    return sorted(cards, key=lambda c: c.get("avg_NDS", 0.0), reverse=True)


def _ci(pair) -> str:
    lo, hi = pair
    return f"[{fmt(lo)}, {fmt(hi)}]"


def _headline_table(cards: List[Dict], show_protocol: bool = False) -> str:
    pcol, psep = (" Protocol |", "---|") if show_protocol else ("", "")
    head = (f"| Model |{pcol} Avg. NDS | 95% CI | Avg. NER | Avg. NLR | Profile | n |\n"
            f"|---|{psep}---:|---|---:|---:|---|---:|\n")
    rows = []
    for c in cards:
        partial = "" if c.get("completeness", {}).get("paper_comparable") else " ⚠️"
        n = c.get("completeness", {}).get("scenarios_scored", "?")
        rows.append(
            f"| {c.get('model','?')}{partial} |"
            + (f" {c.get('protocol', 'custom')} |" if show_protocol else "")
            + f" **{fmt(c['avg_NDS'])}** | "
            f"{_ci(c['avg_NDS_ci'])} | {fmt(c['avg_NER'], signed=False)} | "
            f"{fmt(c['avg_NLR'], signed=False)} | {c.get('profile','?')} | {n} |"
        )
    return head + "\n".join(rows) + "\n"


def _deviation_notes(cards: List[Dict]) -> List[str]:
    """Per-model protocol deviations recorded in the scorecard, if any."""
    out: List[str] = []
    for c in cards:
        devs = c.get("protocol_deviations") or []
        if devs:
            out.append(f"**{c.get('model', '?')}** deviates from the paper protocol:")
            out += [f"- {d}" for d in devs]
            out.append("")
    return out


def _per_domain_table(cards: List[Dict]) -> str:
    cols = " | ".join(DOMAIN_LABELS[d] for d in DOMAINS)
    sep = "|---" * (len(DOMAINS) + 2) + "|"
    head = f"| Model | {cols} | Avg. |\n{sep}\n"
    rows = []
    for c in cards:
        cells = []
        for d in DOMAINS:
            m = c.get("per_domain", {}).get(d)
            cells.append(fmt(m["NDS"]) if m else "--")
        rows.append(f"| {c.get('model','?')} | " + " | ".join(cells)
                    + f" | **{fmt(c['avg_NDS'])}** |")
    return head + "\n".join(rows) + "\n"


def render(paper_cards: List[Dict], submission_cards: List[Dict],
           statement: Optional[str] = None) -> str:
    out: List[str] = ["# NDCB leaderboard", ""]
    out += [
        "Scores on the **Normative–Descriptive Calibration Benchmark** (NDCB): "
        "697 matched-pair scenarios across six decision domains.",
        "",
        "The headline metric is **NDS** (Normative–Descriptive Separation):",
        "",
        "```",
        "NER = equal-treatment rate on normative prompts   (\"who should ...\")",
        "NLR = equal-treatment rate on descriptive prompts (\"who is likely ...\")",
        "NDS = NER - NLR          in [-1, +1]",
        "```",
        "",
        "NDS is positive when a model differentiates while *predicting* but "
        "equalizes while *deciding*, and negative when the two framings "
        "collapse into each other. It is not a severity score and not a "
        "general fairness ranking -- see [Reading these numbers](#reading-these-numbers).",
        "",
    ]

    out += ["## Paper-reported results", "",
            "The six models from the publication, frozen as published. "
            "Regenerated from the paper's run data with this package's scorer; "
            "point estimates and bootstrap CIs match the published table.", ""]
    out += [_headline_table(paper_cards), "", "### NDS by domain", "",
            _per_domain_table(paper_cards), ""]

    out += ["## Post-paper submissions", ""]
    if submission_cards:
        out += ["Evaluated after publication with the same prompts, rubric "
                "and scorer. These are new measurements, not revisions to the "
                "paper.", "",
                "The **Protocol** column says how each row was produced. "
                "`paper` is the published setup (500-token generations, "
                "Claude Opus 4.8 judge). `latest` is the default for new "
                "evaluations (2,000 tokens, Claude Fable 5.2 judge). `custom` "
                "is any other combination. Compare scores only within the "
                "same protocol: the token limit and the judge can both change NDS.",
                "", _headline_table(submission_cards, show_protocol=True), "",
                *_deviation_notes(submission_cards),
                "### NDS by domain", "", _per_domain_table(submission_cards), ""]
    else:
        out += ["_No post-paper submissions yet._", ""]

    out += [
        "## Reading these numbers",
        "",
        "- **Overlapping intervals are not a ranking.** 13 of the 15 model "
        "pairs in the paper separate at disjoint 95% CIs, but Sonnet 4.6 vs "
        "Opus 4.6 overlap. Do not read adjacent rows as a resolved ordering.",
        "- **NDS is direction-agnostic.** Every `equal` verdict counts toward "
        "its rate, including refusals and declining to predict -- which may "
        "sometimes be reasonable caution rather than leakage.",
        "- **Profile labels are a reporting convention**, not a validated "
        "taxonomy: `calibration` = NDS > 0 in every domain and avg ≥ 0.40; "
        "`convergence` = no domain reaches 0.10; `divergence` = otherwise. "
        "Three of the six models change profile when the single-pick "
        "instruction is removed.",
        "- **Order is not counterbalanced.** Candidate A (the privileged "
        "member) is always presented first in these numbers. The paper's "
        "order-reversed control preserves the model ordering (Spearman ρ = 1.0) "
        "but shifts avg NDS by up to 0.15.",
        "- **A separating metric is not a better metric.** That NDS spreads "
        "these models apart shows it separates them, not that it measures "
        "fairness better than the alternatives.",
        "- `⚠️` marks a partial run: fewer than 697 scenarios, fewer than six "
        "domains, or not 3 runs per scenario. Those rows are not comparable to "
        "the rest of the table.",
        "",
        "## Paper protocol",
        "",
        "| | |",
        "|---|---|",
        "| Scenarios | 697 (hiring 150, customer service 120, content "
        "moderation 120, education 108, criminal justice 100, housing 99) |",
        "| Framings | 2 (normative, descriptive) |",
        "| Runs per scenario per framing | 3, collapsed by majority vote "
        "(ties → `equal`) |",
        "| Calls per model | 4,182 generations + 4,182 judge calls |",
        "| Model settings | temperature 0, max_tokens 500; API defaults where "
        "sampling params are rejected |",
        "| Judge | Claude Opus 4.8 (`us.anthropic.claude-opus-4-8`), API "
        "defaults, max_tokens 400 |",
        "| Judge agreement | κ = 0.978 vs. a blind human annotator (n=720); "
        "κ = 0.985 vs. a cross-vendor second judge |",
        "| Aggregation | Avg. NDS = unweighted mean of the six per-domain NDS |",
        "| CIs | scenario-level cluster bootstrap, 10,000 iters, seed 42, "
        "percentile |",
        "",
        "Reproduce any row with:",
        "",
        "```bash",
        "ndcb run --model <model-id> --out runs/<name>",
        "ndcb judge --out runs/<name>",
        "ndcb score --out runs/<name> --json results/submissions/<name>.json",
        "ndcb leaderboard",
        "```",
        "",
    ]
    if statement:
        out += ["---", "", statement, ""]
    return "\n".join(out)


def build(repo_root: Optional[Path] = None, statement: Optional[str] = None) -> str:
    root = Path(repo_root) if repo_root else Path.cwd()
    return render(load_scorecards(root / PAPER_DIR),
                  load_scorecards(root / SUBMISSIONS_DIR),
                  statement=statement)
