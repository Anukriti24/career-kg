from career_kg.dataset import compute_idf, general_weight
from career_kg.similarity import compute_similar


def test_general_weight_bands():
    assert [general_weight(x) for x in (3.4, 3.5, 4.0, 4.5)] == [0, 1, 2, 3]


def test_parse_is_consistent(ds):
    assert len(ds.occupations) > 900
    for (o, s), v in ds.requires.items():
        assert o in ds.occupations and s in ds.skills
        assert v["weight"] in (1, 2, 3)
    # every technology maps to a category, and categories chain up to a segment
    assert all(b in ds.categories for _, b in ds.subclass_of)
    assert {c["level"] for c in ds.categories.values()} == {"commodity", "class", "family", "segment"}
    hot = [v for (o, s), v in ds.requires.items() if s == "tech:Python"]
    assert hot and all(v["core"] for v in hot)


def test_idf_orders_by_rarity(ds):
    idf = compute_idf(ds)
    assert idf["tech:Python"] > idf["tech:Microsoft Excel"] > 0


def test_similarity_excludes_test_occupations_and_ubiquitous_skills(ds):
    sim = compute_similar(ds)
    assert "tech:Microsoft Excel" not in sim          # used by >20% of occupations
    assert all(0.85 <= s <= 1.0001 for ns in sim.values() for _, s in ns)
    names = {b for b, _ in sim.get("tech:TensorFlow", [])}
    assert "tech:PyTorch" in names
