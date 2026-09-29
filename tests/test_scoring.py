import math

from career_kg import scoring
from career_kg.scoring import offer, score_occupation

REQS = [("a", "A", "technology", 3, 2.0), ("b", "B", "technology", 2, 1.0), ("c", "C", "skill", 3, 1.0)]


def test_offer_keeps_best_credit():
    cr = {}
    offer(cr, "x", 0.2, "weak", "u1")
    offer(cr, "x", 0.5, "strong", "u2")
    offer(cr, "x", 0.3, "mid", "u3")
    assert cr["x"] == (0.5, "strong", "u2")


def test_score_splits_matched_inferred_missing():
    cr = {"a": (1.0, "have", "a"), "b": (0.2, "same category", "z")}
    r = score_occupation(REQS, cr, {"technology"})
    assert [m["id"] for m in r["matched"]] == ["a"]
    assert [m["id"] for m in r["inferred"]] == ["b"]
    assert r["missing"] == []
    total = 3 * 2.0 + 2 * 1.0
    assert math.isclose(r["coverage"], (6.0 + 0.2 * 2.0) / total)
    assert math.isclose(r["rank"], (6.0 + 0.4) / math.sqrt(total))


def test_only_requested_types_count():
    r = score_occupation(REQS, {}, {"skill"})
    assert [m["id"] for m in r["missing"]] == ["c"]


def test_rare_skills_outweigh_common_ones():
    rare = [("a", "A", "technology", 2, scoring.idf(1000, 5))]
    common = [("a", "A", "technology", 2, scoring.idf(1000, 900))]
    assert scoring.idf(1000, 5) > scoring.idf(1000, 900)
    both = [rare[0], ("z", "Z", "technology", 2, 1.0)]
    assert score_occupation(both, {"a": (1.0, "", "a")}, {"technology"})["coverage"] > \
        score_occupation([common[0], ("z", "Z", "technology", 2, 1.0)], {"a": (1.0, "", "a")}, {"technology"})["coverage"]
