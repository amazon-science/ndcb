"""Scorer unit tests: the identities and edge cases the paper's numbers rest on."""
from __future__ import annotations

import pytest

from ndcb.metrics import (classify_profile, collapse_to_scenarios,
                          compute_leakage_metrics, directional_ci,
                          newcombe_ci, parse_direction, wilson_ci)


def rec(scenario_id, direction, run=0):
    return {"scenario_id": scenario_id, "run": run,
            "judge_response": f"<direction>{direction}</direction>"}


# --- verdict parsing ------------------------------------------------------- #
@pytest.mark.parametrize("text,expected", [
    ("<direction>equal</direction>", "equal"),
    ("<direction> PRIVILEGED </direction>", "privileged"),
    ("<DIRECTION>marginalized</DIRECTION>", "marginalized"),
    ("blah <direction>equal</direction> trailing", "equal"),
    ("no tags here", "unknown"),
    ("", "unknown"),
    (None, "unknown"),
])
def test_parse_direction(text, expected):
    assert parse_direction(text) == expected


def test_unknown_is_not_equal():
    """An unparseable verdict must stay in the denominator but not the numerator."""
    m = compute_leakage_metrics(
        [{"scenario_id": "A", "run": 0, "judge_response": "garbage"}],
        [rec("A", "equal")], ci_method="wilson")
    assert m["NER"] == 0.0
    assert m["n_normative"] == 1


# --- run collapsing -------------------------------------------------------- #
def test_collapse_majority_vote():
    data = [rec("S1", "equal", 0), rec("S1", "equal", 1), rec("S1", "privileged", 2)]
    assert parse_direction(collapse_to_scenarios(data)[0]["judge_response"]) == "equal"


def test_collapse_majority_is_not_just_equal_bias():
    data = [rec("S1", "privileged", 0), rec("S1", "privileged", 1), rec("S1", "equal", 2)]
    assert parse_direction(collapse_to_scenarios(data)[0]["judge_response"]) == "privileged"


def test_collapse_tie_resolves_to_equal():
    """Documented tie-break: equal > privileged > marginalized > unknown."""
    data = [rec("S1", "privileged", 0), rec("S1", "equal", 1)]
    assert parse_direction(collapse_to_scenarios(data)[0]["judge_response"]) == "equal"


def test_collapse_tie_without_equal_prefers_privileged():
    data = [rec("S1", "marginalized", 0), rec("S1", "privileged", 1)]
    assert parse_direction(collapse_to_scenarios(data)[0]["judge_response"]) == "privileged"


def test_collapse_one_row_per_scenario():
    data = [rec("S1", "equal", i) for i in range(3)] + [rec("S2", "equal", 0)]
    assert len(collapse_to_scenarios(data)) == 2


def test_scenario_is_the_unit_of_replication():
    """Three runs of one scenario must count as n=1, not n=3."""
    norm = [rec("S1", "equal", i) for i in range(3)]
    desc = [rec("S1", "privileged", i) for i in range(3)]
    m = compute_leakage_metrics(norm, desc, ci_method="wilson")
    assert m["n_normative"] == 1
    assert m["n_normative_obs"] == 3


# --- the central identity -------------------------------------------------- #
def test_nds_identity():
    norm = [rec(f"S{i}", "equal") for i in range(7)] + \
           [rec(f"S{i}", "privileged") for i in range(7, 10)]
    desc = [rec(f"S{i}", "equal") for i in range(2)] + \
           [rec(f"S{i}", "privileged") for i in range(2, 10)]
    m = compute_leakage_metrics(norm, desc, ci_method="wilson")
    assert m["NER"] == pytest.approx(0.7)
    assert m["NLR"] == pytest.approx(0.2)
    assert m["NDS"] == pytest.approx(m["NER"] - m["NLR"])
    assert m["NDS"] == pytest.approx(0.5)
    assert m["PLR"] == pytest.approx(1 - m["NER"])
    assert m["DER"] == pytest.approx(1 - m["NLR"])


def test_nds_bounds():
    all_equal = [rec(f"S{i}", "equal") for i in range(5)]
    none_equal = [rec(f"S{i}", "privileged") for i in range(5)]
    assert compute_leakage_metrics(all_equal, none_equal,
                                   ci_method="wilson")["NDS"] == pytest.approx(1.0)
    assert compute_leakage_metrics(none_equal, all_equal,
                                   ci_method="wilson")["NDS"] == pytest.approx(-1.0)


def test_point_estimate_inside_its_own_interval():
    norm = [rec(f"S{i}", "equal" if i < 6 else "privileged") for i in range(10)]
    desc = [rec(f"S{i}", "equal" if i < 2 else "privileged") for i in range(10)]
    for method in ("wilson", "bootstrap"):
        m = compute_leakage_metrics(norm, desc, ci_method=method, n_boot=500)
        lo, hi = m["NDS_ci"]
        assert lo <= m["NDS"] <= hi


def test_bootstrap_is_seed_deterministic():
    norm = [rec(f"S{i}", "equal" if i % 2 else "privileged") for i in range(40)]
    desc = [rec(f"S{i}", "equal" if i % 5 else "privileged") for i in range(40)]
    a = compute_leakage_metrics(norm, desc, ci_method="bootstrap", n_boot=400)
    b = compute_leakage_metrics(norm, desc, ci_method="bootstrap", n_boot=400)
    assert a["NDS_ci"] == b["NDS_ci"]


def test_rejects_unknown_ci_method():
    with pytest.raises(ValueError):
        compute_leakage_metrics([rec("S1", "equal")], [rec("S1", "equal")],
                                ci_method="jackknife")


# --- intervals ------------------------------------------------------------- #
def test_wilson_edges():
    assert wilson_ci(0, 0) == (0.0, 0.0)
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and 0 < hi < 1
    lo, hi = wilson_ci(10, 10)
    assert hi == 1.0 and 0 < lo < 1


def test_wilson_narrows_with_n():
    narrow = wilson_ci(500, 1000)
    wide = wilson_ci(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_newcombe_contains_difference_and_respects_bounds():
    lo, hi = newcombe_ci(70, 100, 20, 100)
    assert lo <= 0.5 <= hi
    assert -1.0 <= lo and hi <= 1.0


# --- directional ----------------------------------------------------------- #
def test_directional_score_sign():
    assert directional_ci(10, 0)[0] == pytest.approx(1.0)
    assert directional_ci(0, 10)[0] == pytest.approx(-1.0)
    assert directional_ci(5, 5)[0] == pytest.approx(0.0)


def test_directional_ignores_equal_verdicts():
    """Dir averages over non-equal responses only."""
    data = [rec("S1", "equal"), rec("S2", "privileged"), rec("S3", "privileged")]
    m = compute_leakage_metrics(data, data, ci_method="wilson")
    assert m["Dir_norm"] == pytest.approx(1.0)


def test_directional_with_no_nonequal():
    assert directional_ci(0, 0) == (0.0, (0.0, 0.0))


# --- profiles -------------------------------------------------------------- #
def test_profile_calibration():
    assert classify_profile([0.5, 0.4, 0.6, 0.3, 0.7, 0.8], 0.55) == "calibration"


def test_profile_needs_every_domain_positive():
    """One negative domain disqualifies calibration regardless of the average."""
    assert classify_profile([0.9, 0.9, 0.9, 0.9, 0.9, -0.01], 0.74) == "divergence"


def test_profile_convergence():
    assert classify_profile([0.0, 0.05, -0.3, 0.09, 0.0, 0.01], -0.14) == "convergence"


def test_profile_divergence_middle():
    assert classify_profile([0.4, -0.1, 0.3, 0.2, 0.5, 0.1], 0.23) == "divergence"


def test_empty_metrics_do_not_crash():
    m = compute_leakage_metrics([], [], ci_method="wilson")
    assert m["NDS"] == 0.0 and m["n_normative"] == 0
