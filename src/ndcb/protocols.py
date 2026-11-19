"""The two evaluation protocols: the paper's, and the latest.

They share the prompts, the three runs per scenario, the judge rubric, the
400-token judge limit and the scorer. They differ in two settings:

=========  ==================  =====================
protocol   generation limit    judge
=========  ==================  =====================
paper      500 tokens          Claude Opus 4.8
latest     2,000 tokens        Claude Fable 5.2
=========  ==================  =====================

``paper`` reproduces the published numbers. ``latest`` is the default for new
evaluations. 500 tokens truncates recent reasoning models: Claude Opus 5.5 hit
the limit on 68% of pilot calls at 500, and on none of 4,182 calls at 2,000.

A score is only comparable with other scores from the same protocol, so every
generation and judge record carries the protocol name, and every scorecard
states which protocol produced it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional


@dataclass(frozen=True)
class Protocol:
    name: str
    max_tokens: int
    #: Provider-neutral model name; see `judge_model_id` for the per-API id.
    judge: str
    description: str


PROTOCOLS: Dict[str, Protocol] = {
    "paper": Protocol(
        "paper", 500, "claude-opus-4-8",
        "As published: 500-token generations, Claude Opus 4.8 judge."),
    "latest": Protocol(
        "latest", 2000, "claude-fable-5-2",
        "Default for new evaluations: 2,000-token generations, "
        "Claude Fable 5.2 judge."),
}
DEFAULT_PROTOCOL = "latest"
CUSTOM = "custom"

# Bedrock ids put a geography and the provider in front of the model name:
# "us.anthropic.claude-opus-4-8", "global.anthropic.claude-fable-5-2".
_BEDROCK_PREFIX = re.compile(r"^(?:[a-z]{2,6}\.)?anthropic\.")


def get(name: str) -> Protocol:
    try:
        return PROTOCOLS[name]
    except KeyError:
        raise ValueError(
            f"unknown protocol {name!r}; choose from {', '.join(PROTOCOLS)}") from None


def judge_model_id(protocol: str, api: str) -> str:
    """The protocol's judge as the given backend names it."""
    base = get(protocol).judge
    if api == "bedrock":
        # The US cross-region profile, as the paper used. Outside the US, pass
        # --judge-model with your geography's prefix, e.g. eu. or global.
        return f"us.anthropic.{base}"
    if api == "anthropic":
        return base
    raise ValueError(
        f"the {protocol!r} protocol's judge is a Claude model, which the "
        f"{api!r} backend cannot call. Judge with --api anthropic or "
        "--api bedrock, or pass --judge-model to use another judge (the score "
        "will then be labelled 'custom').")


def normalise_model(model_id: Optional[str]) -> str:
    """Provider-neutral form of a model id, for comparing judges."""
    return _BEDROCK_PREFIX.sub("", (model_id or "").strip())


def classify(max_tokens: Iterable[Optional[int]],
             judge_models: Iterable[Optional[str]]) -> str:
    """Which protocol a finished run followed: a protocol name or "custom".

    A run counts as a protocol only if every generation used its token limit
    and every verdict came from its judge. Anything mixed is "custom".
    """
    tokens = set(max_tokens)
    judges = {normalise_model(j) for j in judge_models}
    if len(tokens) != 1 or len(judges) != 1:
        return CUSTOM
    (tok,), (judge,) = tokens, judges
    for p in PROTOCOLS.values():
        if tok == p.max_tokens and judge == p.judge:
            return p.name
    return CUSTOM
