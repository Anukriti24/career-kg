import numpy as np

from career_kg import kge


def test_triples_exclude_and_holdout(ds):
    some = set(list(ds.occupations)[:20])
    tr = kge.Triples(ds, exclude_occ=some, holdout_frac=0.2, seed=1)
    excluded_ids = {tr.eid[o] for o in some}
    assert not any(h in excluded_ids and r in (0, 1) for h, r, _ in tr.train + tr.test)
    assert tr.test and all(r == 0 for _, r, _ in tr.test)
    assert not set(tr.test) & set(tr.train)


def test_tiny_training_runs_and_scores_all_entities(ds):
    tr = kge.Triples(ds, holdout_frac=0.05)
    model = kge.train(tr, dim=8, epochs=1, log=lambda *_: None)
    logits = kge.score_tails(model, tr, [0, 1])
    assert logits.shape == (2, len(tr.ents)) and np.isfinite(logits).all()
    rows = kge.predict_missing(model, tr, ds, top=3)
    known = {(tr.ents[h], tr.ents[t]) for h, r, t in tr.train if r in (0, 1)}
    assert len(rows) == 3 * len(ds.occupations)
    assert all((o, s) not in known and s.startswith("tech:") for o, s, _ in rows)


def test_save_and_load_roundtrip(ds, tmp_path):
    tr = kge.Triples(ds)
    model = kge.train(tr, dim=8, epochs=1, log=lambda *_: None)
    kge.save(model, tr, tmp_path / "m.pt", meta={"x": 1})
    loaded, blob = kge.load(tmp_path / "m.pt")
    assert blob["entities"] == tr.ents and blob["meta"] == {"x": 1}
    assert np.allclose(kge.score_tails(model, tr, [0]), kge.score_tails(loaded, tr, [0]), atol=1e-5)


def test_rgcn_trains_and_scores_all_entities(ds):
    from career_kg import gnn
    tr = kge.Triples(ds, holdout_frac=0.05)
    model = gnn.train(tr, dim=8, epochs=1, log=lambda *_: None)
    logits = kge.score_tails(model, tr, [0, 1])
    assert logits.shape == (2, len(tr.ents)) and np.isfinite(logits).all()
