import argparse

from . import config


def show(res):
    if res["unknown"]:
        for u in res["unknown"]:
            hint = f"  (did you mean: {', '.join(u['suggestions'])}?)" if u["suggestions"] else ""
            print(f"  ? unknown skill '{u['input']}'{hint}")
    for i, r in enumerate(res["results"], 1):
        print(f"\n{i:>2}. {r['title']}  [{r['code']}, zone {r['zone']}]  {r['coverage']:.0%} coverage")
        if r["matched"]:
            print("      have    :", ", ".join(m["name"] for m in r["matched"]))
        for m in r["inferred"][:5]:
            print(f"      inferred: {m['name']} @ {m['credit']:.2f} ({m['why']})")
        if r["missing"]:
            print("      missing :", ", ".join(m["name"] for m in r["missing"][:6]))
    if res["learn"]:
        print("\nWorth learning next:", ", ".join(f"{x['name']}" for x in res["learn"]))


def main():
    ap = argparse.ArgumentParser(prog="career_kg")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="download O*NET, learn embeddings, load Neo4j")
    sub.add_parser("stats", help="node / relationship counts in Neo4j")
    r = sub.add_parser("recommend", help="recommend occupations for comma-separated skills")
    r.add_argument("skills")
    r.add_argument("-k", type=int, default=5)
    r.add_argument("--no-inference", action="store_true", help="exact matching only (baseline)")
    r.add_argument("--zones", default="1-5", help="job zone (education/experience level) range, e.g. 3-5")
    kp = sub.add_parser("kge", help="train ComplEx on the full graph; store PREDICTED_REQUIRES in Neo4j")
    kp.add_argument("--epochs", type=int, default=30)
    kp.add_argument("--dim", type=int, default=200)
    g = sub.add_parser("gnn", help="train a small R-GCN and compare its link prediction with the baselines")
    g.add_argument("--epochs", type=int, default=40)
    g.add_argument("--dim", type=int, default=64)
    e = sub.add_parser("eval", help="held-out evaluation vs. exact-match baseline")
    e.add_argument("-n", type=int, default=300)
    e.add_argument("--no-kge", action="store_true", help="skip the (slow) ComplEx training")
    e.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()

    if a.cmd == "ingest":
        from . import dataset, graph_store, reasoning, similarity
        ds = dataset.parse()
        print(f"parsed {len(ds.occupations)} occupations, {len(ds.skills)} skills, {len(ds.requires)} usage rows")
        sim = similarity.compute_similar(ds)
        print(f"learned SIMILAR edges for {len(sim)} technologies")
        facts = reasoning.derive(ds)
        print("rules derived:", {p: len(facts[p]) for p in ("in_cat", "has_area", "move")})
        driver = graph_store.connect()
        graph_store.load(driver, ds, sim, facts)
        print("loaded:", *graph_store.stats(driver), sep="\n  ")
    elif a.cmd == "stats":
        from . import graph_store
        print(*graph_store.stats(graph_store.connect()), sep="\n")
    elif a.cmd == "recommend":
        from . import graph_store
        from .recommender import Recommender
        lo, hi = (int(x) for x in a.zones.split("-"))
        res = Recommender(graph_store.connect()).recommend(
            [x for x in a.skills.split(",")], k=a.k, zone_min=lo, zone_max=hi, inference=not a.no_inference)
        print("understood:", ", ".join(x["name"] + (f" (from '{x['typed_as']}')" if x["typed_as"] else "") for x in res["resolved"]) or "-")
        show(res)
    elif a.cmd == "kge":
        from . import dataset, graph_store, kge
        ds = dataset.parse()
        tr = kge.Triples(ds)
        model = kge.train(tr, dim=a.dim, epochs=a.epochs)
        path = config.MODEL_DIR / "complex.pt"
        kge.save(model, tr, path, meta=dict(dim=a.dim, epochs=a.epochs, onet=config.ONET_VERSION))
        print(f"saved weights to {path} ({path.stat().st_size / 1e6:.1f} MB)")
        predicted = kge.predict_missing(model, tr, ds)
        graph_store.load_kge(graph_store.connect(), predicted)
        print(f"stored {len(predicted)} PREDICTED_REQUIRES edges")
    elif a.cmd == "gnn":
        from . import dataset, evaluate, gnn
        print(evaluate.report_lp(gnn.compare(dataset.parse(), epochs=a.epochs, dim=a.dim)))
    elif a.cmd == "eval":
        from . import dataset, evaluate
        prep = evaluate.prepare(dataset.parse(), n_test=a.n, use_kge=not a.no_kge, kge_epochs=a.epochs)
        if prep["lp"]:
            print(evaluate.report_lp(prep["lp"]), end="\n\n")
        for swap in (0.0, 0.5):
            print(evaluate.report(evaluate.run(prep, swap=swap)), end="\n\n")


if __name__ == "__main__":
    main()
