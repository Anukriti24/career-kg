"""Integration tests against a loaded Neo4j (skipped when it is not running)."""
import math

from career_kg import scoring
from career_kg.dataset import compute_idf
from career_kg.scoring import score_occupation


def titles(res):
    return [r["title"] for r in res["results"]]


def test_ml_profile_finds_data_scientists(rec):
    res = rec.recommend(["Python", "TensorFlow", "Tableau"], k=5)
    assert "Data Scientists" in titles(res)[:3]


def test_inference_credits_similar_tool(rec):
    res = rec.recommend(["TensorFlow"], k=10, inference=True)
    why = {m["name"]: m for r in res["results"] for m in r["inferred"]}
    assert "PyTorch" in why and "TensorFlow" in why["PyTorch"]["why"]
    base = rec.recommend(["TensorFlow"], k=10, inference=False)
    assert all(not r["inferred"] for r in base["results"])


def test_alias_and_unknown_and_suggestions(rec):
    res = rec.recommend(["sql", "nonexistent-skill-xyz"], k=3)
    assert any(r["typed_as"] == "sql" for r in res["resolved"])
    assert [u["input"] for u in res["unknown"]] == ["nonexistent-skill-xyz"]
    assert rec.suggest("tabl")[0]["name"] == "Tableau"


def test_category_input_expands_transitively(rec):
    res = rec.recommend(["Business intelligence and data analysis software"], k=5)
    assert res["results"] and all(m["why"].startswith("part of") for r in res["results"] for m in r["inferred"])


def test_zone_filter(rec):
    res = rec.recommend(["Python"], k=20, zone_min=4, zone_max=5)
    assert all(r["zone"] is None or r["zone"] >= 4 for r in res["results"])


def test_no_known_skills_returns_empty(rec):
    assert rec.recommend(["zzz"], k=5)["results"] == []


def test_cypher_ranking_matches_in_memory_scoring(rec, ds):
    """The graph query and the pure-Python scorer used by the evaluation must agree."""
    names = ["Python", "Tableau", "Docker"]
    res = rec.recommend(names, k=5)
    skills, cats, _ = rec.resolve(names)
    credits = rec.credits(skills, cats)
    idf = compute_idf(ds)
    for r in res["results"]:
        reqs = [(s, ds.skills[s]["name"], ds.skills[s]["type"], v["weight"], idf[s])
                for (o, s), v in ds.requires.items() if o == r["code"] and v["core"]]
        expected = score_occupation(reqs, credits, {"technology"})
        assert math.isclose(expected["rank"], r["rank"], rel_tol=1e-6)
        assert math.isclose(expected["coverage"], r["coverage"], rel_tol=1e-6)


def test_rule_derived_and_learned_edges_present(rec):
    d = rec.occupation("15-2051.00")
    assert d["moves"] and all(m["hops"] in (1, 2) for m in d["moves"])
    assert d["predicted"] is not None
    res = rec.recommend(["Python", "Tableau"], k=3)
    assert all("areas" in r for r in res["results"])
