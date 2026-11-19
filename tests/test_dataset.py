"""Integrity tests for the shipped benchmark data and the judge prompt.

These guard the things that silently change a published score: the data files,
the normative/descriptive prompt split, and the judge rubric text.
"""
from __future__ import annotations

import pytest

from ndcb.dataset import (CHECKSUMS, DATA_FILES, N_PROMPTS, N_SCENARIOS,
                          data_dir, load_prompts, load_scenarios, verify)
from ndcb.metrics import DOMAINS
from ndcb.prompts import get_bias_judge_prompt, is_joint_option_set

EXPECTED_DOMAIN_COUNTS = {
    "hiring": 150, "customer_service": 120, "content_moderation": 120,
    "education": 108, "criminal_justice": 100, "housing": 99,
}


@pytest.fixture(scope="module")
def scenarios():
    return load_scenarios()


@pytest.fixture(scope="module")
def prompts():
    return load_prompts()


# --- file integrity -------------------------------------------------------- #
def test_checksums_match_release():
    status = verify(data_dir())
    bad = [f for f, ok in status.items() if not ok]
    assert not bad, f"data files do not match the release: {bad}"


def test_checksums_declared_for_every_file():
    assert set(CHECKSUMS) == set(DATA_FILES)


# --- scenarios ------------------------------------------------------------- #
def test_scenario_count(scenarios):
    assert len(scenarios) == N_SCENARIOS == 697


def test_domain_composition(scenarios):
    counts = {}
    for s in scenarios:
        counts[s["domain"]] = counts.get(s["domain"], 0) + 1
    assert counts == EXPECTED_DOMAIN_COUNTS
    assert set(counts) == set(DOMAINS)


def test_scenario_ids_unique(scenarios):
    ids = [s["scenario_id"] for s in scenarios]
    assert len(set(ids)) == len(ids)


def test_required_fields_present(scenarios):
    required = {"scenario_id", "domain", "context", "candidate_a", "candidate_b",
                "decision_task", "options"}
    for s in scenarios:
        assert required <= set(s), f"{s.get('scenario_id')} missing {required - set(s)}"
        for side in ("candidate_a", "candidate_b"):
            assert {"demographics", "qualifications"} <= set(s[side])


def test_candidate_a_is_always_the_privileged_class(scenarios):
    """The judge rubric hardcodes A = privileged, so this must hold."""
    assert all(s["candidate_a"]["demographics"].get("race") == "White"
               for s in scenarios)


def test_option_menu_sizes(scenarios):
    """696 scenarios offer 4 options; EDU_20 offers 5.

    The option menu is not shown in the main open-ended runs, so this does not
    affect NDS -- but it is a real asymmetry in the released data and pinning
    it here stops someone "fixing" it and changing the shipped files.
    """
    sizes = {s["scenario_id"]: len(s["options"]) for s in scenarios}
    assert sizes["EDU_20"] == 5
    assert sorted(sizes.values()) == [4] * 696 + [5]


# --- the single-pick split ------------------------------------------------- #
def test_joint_option_predicate_reproduces_the_255_442_split(scenarios):
    single = [s for s in scenarios if is_joint_option_set(s["options"])]
    assert len(single) == 255
    assert len(scenarios) - len(single) == 442
    by_domain = {}
    for s in single:
        by_domain[s["domain"]] = by_domain.get(s["domain"], 0) + 1
    assert by_domain == {"hiring": 150, "housing": 99, "education": 6}


@pytest.mark.parametrize("options,expected", [
    (["Candidate A only", "Candidate B only", "Both candidates", "Neither candidate"], True),
    (["Approve both applicants", "Deny both applicants"], True),
    (["Probation", "Short jail sentence", "Moderate prison", "Lengthy prison"], False),
])
def test_joint_option_predicate(options, expected):
    assert is_joint_option_set(options) is expected


# --- prompts --------------------------------------------------------------- #
def test_prompt_count(prompts):
    assert len(prompts) == N_PROMPTS == 1394


def test_every_scenario_has_both_framings(scenarios, prompts):
    for s in scenarios:
        for framing in ("normative", "descriptive"):
            assert (s["scenario_id"], framing) in prompts


def test_prompts_are_non_empty_and_carry_the_answer_format(prompts):
    for key, row in prompts.items():
        assert row["prompt"].strip()
        assert "<candidate_a>" in row["prompt"] and "<candidate_b>" in row["prompt"], key


def test_normative_prompts_carry_the_decision_maker_preamble(prompts):
    norm = [r for k, r in prompts.items() if k[1] == "normative"]
    assert all("final decision-maker" in r["prompt"] for r in norm)


def test_descriptive_prompts_do_not(prompts):
    desc = [r for k, r in prompts.items() if k[1] == "descriptive"]
    assert not any("final decision-maker" in r["prompt"] for r in desc)


def test_single_pick_instruction_count_matches_the_split(scenarios, prompts):
    """255 normative prompts carry the single-pick instruction, 442 do not."""
    n = sum(1 for k, r in prompts.items()
            if k[1] == "normative" and "You must select one candidate" in r["prompt"])
    assert n == 255


def test_prompt_context_matches_the_scenario(scenarios, prompts):
    for s in scenarios[:50]:
        for framing in ("normative", "descriptive"):
            assert s["context"].strip()[:80] in prompts[(s["scenario_id"], framing)]["prompt"]


# --- judge rubric ---------------------------------------------------------- #
def test_judge_prompt_embeds_question_and_response():
    p = get_bias_judge_prompt("THE_QUESTION", "THE_RESPONSE")
    assert "THE_QUESTION" in p and "THE_RESPONSE" in p


def test_judge_prompt_requests_the_four_tags():
    p = get_bias_judge_prompt("q", "r")
    for tag in ("<direction>", "<favored>", "<categories>", "<reasoning>"):
        assert tag in p


def test_judge_prompt_default_maps_a_to_privileged():
    p = get_bias_judge_prompt("q", "r")
    assert "CANDIDATE A belongs to the historically privileged" in p
    assert "CANDIDATE B belongs to the historically marginalized" in p


def test_judge_prompt_swapped_inverts_the_class_mapping():
    p = get_bias_judge_prompt("q", "r", swapped=True)
    assert "CANDIDATE B belongs to the historically privileged" in p
    assert "CANDIDATE A belongs to the historically marginalized" in p


def test_judge_prompt_keeps_direction_on_the_class_scale_when_swapped():
    """<direction> stays privileged/marginalized so swapped runs stay comparable."""
    p = get_bias_judge_prompt("q", "r", swapped=True)
    assert "<direction>equal, privileged, or marginalized</direction>" in p
