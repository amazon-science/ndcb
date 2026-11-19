# NDCB leaderboard

Scores on the **Normative–Descriptive Calibration Benchmark** (NDCB): 697 matched-pair scenarios across six decision domains.

The headline metric is **NDS** (Normative–Descriptive Separation):

```
NER = equal-treatment rate on normative prompts   ("who should ...")
NLR = equal-treatment rate on descriptive prompts ("who is likely ...")
NDS = NER - NLR          in [-1, +1]
```

NDS is positive when a model differentiates while *predicting* but equalizes while *deciding*, and negative when the two framings collapse into each other. It is not a severity score and not a general fairness ranking -- see [Reading these numbers](#reading-these-numbers).

## Paper-reported results

The six models from the publication, frozen as published. Regenerated from the paper's run data with this package's scorer; point estimates and bootstrap CIs match the published table.

| Model | Avg. NDS | 95% CI | Avg. NER | Avg. NLR | Profile | n |
|---|---:|---|---:|---:|---|---:|
| Claude Sonnet 4.6 | **+0.55** | [+0.51, +0.59] | 0.79 | 0.24 | calibration | 697 |
| Claude Opus 4.6 | **+0.52** | [+0.49, +0.56] | 0.71 | 0.18 | calibration | 697 |
| GPT-OSS 120B | **+0.43** | [+0.40, +0.46] | 0.66 | 0.22 | divergence | 697 |
| Mistral Large 3 | **+0.42** | [+0.38, +0.46] | 0.68 | 0.26 | divergence | 697 |
| Qwen3-32B | **+0.06** | [+0.02, +0.10] | 0.78 | 0.73 | divergence | 697 |
| GPT-5.5 | **-0.14** | [-0.17, -0.12] | 0.84 | 0.98 | convergence | 697 |


### NDS by domain

| Model | Crim. Just. | Housing | Hiring | Education | Cust. Svc. | Cont. Mod. | Avg. |
|---|---|---|---|---|---|---|---|
| Claude Sonnet 4.6 | +0.76 | +0.34 | +0.15 | +0.41 | +0.83 | +0.80 | **+0.55** |
| Claude Opus 4.6 | +0.95 | +0.22 | +0.12 | +0.14 | +0.98 | +0.72 | **+0.52** |
| GPT-OSS 120B | +0.96 | -0.12 | -0.07 | +0.35 | +0.76 | +0.73 | **+0.43** |
| Mistral Large 3 | +0.60 | -0.08 | +0.47 | +0.26 | +0.93 | +0.34 | **+0.42** |
| Qwen3-32B | +0.18 | -0.41 | -0.24 | +0.05 | +0.57 | +0.19 | **+0.06** |
| GPT-5.5 | +0.00 | -0.37 | -0.44 | -0.06 | +0.00 | +0.01 | **-0.14** |


## Post-paper submissions

Evaluated after publication with the same prompts, rubric and scorer. These are new measurements, not revisions to the paper.

The **Protocol** column says how each row was produced. `paper` is the published setup: 500 tokens, temperature 0 and thinking off (GPT-5.5: 3,000 tokens), with the Claude Opus 4.8 judge. `latest` is the default for new evaluations: 2,000 tokens, temperature 0 and thinking off for every model, with the Claude Fable 5.2 judge. `custom` is any other combination. Compare scores only within the same protocol: the token limit and the judge can both change NDS.

| Model | Protocol | Avg. NDS | 95% CI | Avg. NER | Avg. NLR | Profile | n |
|---|---|---:|---|---:|---:|---|---:|
| Claude Opus 4.6 (2,000 tokens) | custom | **+0.58** | [+0.55, +0.61] | 0.72 | 0.14 | calibration | 697 |
| Claude Sonnet 4.6 (2,000 tokens) | custom | **+0.54** | [+0.50, +0.58] | 0.78 | 0.24 | calibration | 697 |
| Claude Opus 5.5 | custom | **+0.05** | [+0.03, +0.06] | 1.00 | 0.95 | divergence | 697 |
| GPT-6 Astra (Bedrock, max_tokens=2000) ⚠️ | latest | **-0.34** | [-0.35, -0.34] | 0.66 | 1.00 | convergence | 697 |


**Claude Opus 4.6 (2,000 tokens)** deviates from the paper protocol:
- Generation limit 2,000 tokens instead of the paper's 500. Everything else matches the paper: temperature 0, thinking off, same prompts and three runs.
- Judged by the paper's Claude Opus 4.8, so the difference from the paper row comes from the token limit (and run-to-run variation).
- Paper score for comparison: +0.52 [+0.49, +0.56]; this run: +0.58 [+0.55, +0.61].

**Claude Sonnet 4.6 (2,000 tokens)** deviates from the paper protocol:
- Generation limit 2,000 tokens instead of the paper's 500. Everything else matches the paper: temperature 0, thinking off, same prompts and three runs.
- Judged by the paper's Claude Opus 4.8, so the difference from the paper row comes from the token limit (and run-to-run variation).
- Paper score for comparison: +0.55 [+0.51, +0.59]; this run: +0.54 [+0.50, +0.58].

**Claude Opus 5.5** deviates from the paper protocol:
- max_tokens 2000 instead of the paper's 500: at 500, 68% of pilot calls were truncated (paper models: 0-3%); at 2000, 0 of 4,182 were.
- No temperature: the model rejects the parameter, so it ran at API defaults (as the paper did for GPT-5.5 and the judge).

### NDS by domain

| Model | Crim. Just. | Housing | Hiring | Education | Cust. Svc. | Cont. Mod. | Avg. |
|---|---|---|---|---|---|---|---|
| Claude Opus 4.6 (2,000 tokens) | +0.98 | +0.33 | +0.14 | +0.25 | +1.00 | +0.79 | **+0.58** |
| Claude Sonnet 4.6 (2,000 tokens) | +0.71 | +0.32 | +0.10 | +0.38 | +0.89 | +0.83 | **+0.54** |
| Claude Opus 5.5 | +0.00 | +0.20 | +0.01 | +0.03 | +0.00 | +0.03 | **+0.05** |
| GPT-6 Astra (Bedrock, max_tokens=2000) | +0.00 | -1.00 | -1.00 | -0.06 | +0.00 | +0.00 | **-0.34** |


## Reading these numbers

- **Overlapping intervals are not a ranking.** 13 of the 15 model pairs in the paper separate at disjoint 95% CIs, but Sonnet 4.6 vs Opus 4.6 overlap. Do not read adjacent rows as a resolved ordering.
- **NDS is direction-agnostic.** Every `equal` verdict counts toward its rate, including refusals and declining to predict -- which may sometimes be reasonable caution rather than leakage.
- **Profile labels are a reporting convention**, not a validated taxonomy: `calibration` = NDS > 0 in every domain and avg ≥ 0.40; `convergence` = no domain reaches 0.10; `divergence` = otherwise. Three of the six models change profile when the single-pick instruction is removed.
- **Order is not counterbalanced.** Candidate A (the privileged member) is always presented first in these numbers. The paper's order-reversed control preserves the model ordering (Spearman ρ = 1.0) but shifts avg NDS by up to 0.15.
- **A separating metric is not a better metric.** That NDS spreads these models apart shows it separates them, not that it measures fairness better than the alternatives.
- `⚠️` marks a partial run: fewer than 697 scenarios, fewer than six domains, or not 3 runs per scenario. Those rows are not comparable to the rest of the table.

## Paper protocol

| | |
|---|---|
| Scenarios | 697 (hiring 150, customer service 120, content moderation 120, education 108, criminal justice 100, housing 99) |
| Framings | 2 (normative, descriptive) |
| Runs per scenario per framing | 3, collapsed by majority vote (ties → `equal`) |
| Calls per model | 4,182 generations + 4,182 judge calls |
| Model settings | temperature 0, max_tokens 500; API defaults where sampling params are rejected |
| Judge | Claude Opus 4.8 (`us.anthropic.claude-opus-4-8`), API defaults, max_tokens 400 |
| Judge agreement | κ = 0.978 vs. a blind human annotator (n=720); κ = 0.985 vs. a cross-vendor second judge |
| Aggregation | Avg. NDS = unweighted mean of the six per-domain NDS |
| CIs | scenario-level cluster bootstrap, 10,000 iters, seed 42, percentile |

Reproduce any row with:

```bash
ndcb run --model <model-id> --out runs/<name>
ndcb judge --out runs/<name>
ndcb score --out runs/<name> --json results/submissions/<name>.json
ndcb leaderboard
```
