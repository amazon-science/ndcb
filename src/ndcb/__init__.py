"""NDCB -- the Normative-Descriptive Calibration Benchmark.

697 matched-pair scenarios across six decision domains, plus the harness and
scorer for its headline metric, NDS:

    NER = equal-treatment rate on normative prompts   ("who should ...")
    NLR = equal-treatment rate on descriptive prompts ("who is likely ...")
    NDS = NER - NLR

Quick start::

    from ndcb import load_scenarios, compute_leakage_metrics

    scenarios = load_scenarios()

See `ndcb --help` for the evaluation pipeline.
"""
from __future__ import annotations

__version__ = "1.0.0"

from .dataset import load_prompts, load_scenarios  # noqa: E402
from .metrics import (DOMAINS, aggregate_over_domains, classify_profile,  # noqa: E402
                      collapse_to_scenarios, compute_leakage_metrics,
                      parse_direction)
from .prompts import get_bias_judge_prompt, is_joint_option_set  # noqa: E402

__all__ = [
    "__version__",
    "DOMAINS",
    "aggregate_over_domains",
    "classify_profile",
    "collapse_to_scenarios",
    "compute_leakage_metrics",
    "get_bias_judge_prompt",
    "is_joint_option_set",
    "load_prompts",
    "load_scenarios",
    "parse_direction",
]
