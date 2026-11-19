"""The two evaluation protocols: the paper's, and the latest.

They share the prompts, the three runs per scenario, the judge rubric, the
400-token judge limit and the scorer. They differ in generation settings and
the judge.

``latest`` (the default for new evaluations) uses one setting for every
model: 2,000 tokens, temperature 0, thinking off, judged by Claude Fable 5.2.

``paper`` reproduces how each model was run in the paper, judged by Claude
Opus 4.8. The paper did not use one setting for every model:

=====================================  ============  ===========  ==================
model                                  token limit   temperature  reasoning
=====================================  ============  ===========  ==================
Claude Opus 4.6, Claude Sonnet 4.6     2,500         1            thinking, 1,024 budget
GPT-5.5                                none          API default  effort "medium"
all other models                       500           0            off
=====================================  ============  ===========  ==================

``latest`` uses 2,000 tokens because 500 truncates recent reasoning models:
Claude Opus 5.5 hit the limit on 68% of pilot calls at 500, and on none of
4,182 calls at 2,000.

A score is only comparable with other scores from the same protocol, so every
generation and judge record carries the protocol name, and every scorecard
states which protocol produced it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class GenerationSettings:
    """How to call one model under test."""
    #: None means no limit is sent.
    max_tokens: Optional[int]
    #: None means the API default.
    temperature: Optional[float]
    thinking_budget: Optional[int] = None
    reasoning_effort: Optional[str] = None

    def extra_body(self, api: str) -> Optional[Dict]:
        """Provider-specific request fields for these settings, if any."""
        if self.thinking_budget is not None:
            thinking = {"type": "enabled", "budget_tokens": self.thinking_budget}
            if api == "bedrock":
                return {"additionalModelRequestFields": {"thinking": thinking}}
            if api == "anthropic":
                return {"thinking": thinking}
            raise ValueError("extended thinking needs --api bedrock or --api anthropic")
        if self.reasoning_effort is not None:
            if api == "bedrock":
                return {"additionalModelRequestFields":
                        {"reasoning_effort": self.reasoning_effort}}
            if api == "openai":
                return {"reasoning_effort": self.reasoning_effort}
            raise ValueError("reasoning effort needs --api bedrock or --api openai")
        return None


@dataclass(frozen=True)
class Protocol:
    name: str
    #: Settings for any model without its own entry in `per_model`.
    default: GenerationSettings
    #: Provider-neutral model name; see `judge_model_id` for the per-API id.
    judge: str
    description: str
    #: (substring of the model id, settings), checked in order.
    per_model: Tuple[Tuple[str, GenerationSettings], ...] = field(default=())

    @property
    def max_tokens(self) -> Optional[int]:
        """The default token limit (models with their own settings may differ)."""
        return self.default.max_tokens

    def generation(self, model_id: str) -> GenerationSettings:
        mid = (model_id or "").lower()
        for key, settings in self.per_model:
            if key in mid:
                return settings
        return self.default


_CLAUDE_46_PAPER = GenerationSettings(max_tokens=2500, temperature=1.0,
                                      thinking_budget=1024)

PROTOCOLS: Dict[str, Protocol] = {
    "paper": Protocol(
        "paper", GenerationSettings(max_tokens=500, temperature=0.0),
        "claude-opus-4-8",
        "As published: each model's paper settings, Claude Opus 4.8 judge.",
        per_model=(
            ("claude-opus-4-6", _CLAUDE_46_PAPER),
            ("claude-sonnet-4-6", _CLAUDE_46_PAPER),
            ("gpt-5.5", GenerationSettings(max_tokens=None, temperature=None,
                                           reasoning_effort="medium")),
        )),
    "latest": Protocol(
        "latest", GenerationSettings(max_tokens=2000, temperature=0.0),
        "claude-fable-5-2",
        "Default for new evaluations: 2,000 tokens, temperature 0, thinking "
        "off, Claude Fable 5.2 judge."),
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


def from_records(gen_protocols: Iterable[Optional[str]],
                 judge_protocols: Iterable[Optional[str]],
                 max_tokens: Iterable[Optional[int]],
                 judge_models: Iterable[Optional[str]]) -> str:
    """Which protocol a finished run followed: a protocol name or "custom".

    Records written by `ndcb run` / `ndcb judge` name their protocol, or
    "custom" when a setting was overridden. A run is a protocol only if every
    generation and every verdict names the same one. Records from before
    protocols existed fall back to `classify`.
    """
    gens, judges = set(gen_protocols), set(judge_protocols)
    if None in gens or None in judges or not gens or not judges:
        return classify(max_tokens, judge_models)
    both = gens | judges
    return both.pop() if len(both) == 1 else CUSTOM


def classify(max_tokens: Iterable[Optional[int]],
             judge_models: Iterable[Optional[str]]) -> str:
    """Fallback for records that don't name a protocol.

    Matches on the default token limit and the judge only, so it can't tell
    the paper's per-model settings apart; anything mixed is "custom".
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
