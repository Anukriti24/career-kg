"""Held-out evaluation, no database needed.

For each sampled occupation, pretend to be a job seeker who knows a random half of its core
technologies and check where the true occupation ranks. Everything learned from data (SVD and ComplEx
embeddings) is re-trained WITHOUT the test occupations, so nothing can leak the answer. Systems:
  exact      keyword-style overlap only
  kg         exact + shared-category credit
  kg+class   kg + rule-derived class-level credit (Datalog closure over the taxonomy)
  kg+svd     kg + SVD (LSA) SIMILAR edges
  kg+kge     kg + ComplEx SIMILAR_KGE edges
Scenario `swap`: that share of known skills is replaced by an equivalent tool of the same commodity."""
import random
from collections import defaultdict

from . import kge as kge_mod
from . import reasoning, scoring
from .dataset import compute_idf
from .scoring import offer, score_occupation
from .similarity import compute_similar

MODES = ("exact", "kg", "kg+class", "kg+svd", "kg+kge")


def prepare(ds, n_test=300, seed=0, use_kge=True, kge_epochs=30, kge_dim=200, log=print):
    """Everything that depends on the held-out split; reusable across scenarios."""
    rng = random.Random(seed)
    idf = compute_idf(ds)
    reqs = defaultdict(list)
    for (o, s), v in ds.requires.items():
        if v["core"]:
            sk = ds.skills[s]
            reqs[o].append((s, sk["name"], sk["type"], v["weight"], idf[s]))
    candidates = [o for o, r in reqs.items() if sum(1 for x in r if x[2] == "technology") >= 6]
    test = set(rng.sample(candidates, min(n_test, len(candidates))))
    by_cat, cat_of = defaultdict(set), defaultdict(set)
    for s, c in ds.is_a:
        by_cat[c].add(s)
        cat_of[s].add(c)
    facts = reasoning.derive(ds)
    classes = {c for c, v in ds.categories.items() if v["level"] == "class"}
    class_of, by_class = defaultdict(set), defaultdict(set)
    for s, c in facts["in_cat"]:
        if c in classes:
            class_of[s].add(c)
            by_class[c].add(s)
    prep = dict(ds=ds, reqs=reqs, test=test, by_cat=by_cat, cat_of=cat_of, class_of=class_of, by_class=by_class,
                svd=compute_similar(ds, exclude_occ=test), kge=None, lp=None)
    if use_kge:
        log("training ComplEx without the test occupations …")
        tr = kge_mod.Triples(ds, exclude_occ=test, holdout_frac=0.1)
        model = kge_mod.train(tr, dim=kge_dim, epochs=kge_epochs, log=log)
        prep["kge"] = kge_mod.skill_similar(model, tr, ds, exclude_occ=test)
        prep["lp"] = kge_mod.link_prediction(model, tr, ds)
    return prep


def run(prep, seed=0, keep=0.5, swap=0.0, ks=(1, 5, 10), modes=MODES):
    rng = random.Random(seed + 1)
    ds, reqs, test = prep["ds"], prep["reqs"], prep["test"]
    by_cat, cat_of, class_of, by_class = prep["by_cat"], prep["cat_of"], prep["class_of"], prep["by_class"]

    def credits_for(known, mode):
        cr = {}
        for u in known:
            offer(cr, u, 1.0, "have", u)
            if mode == "exact":
                continue
            for c in cat_of[u]:
                for t in by_cat[c]:
                    if t != u:
                        offer(cr, t, scoring.SAME_CATEGORY, "cat", u)
            if mode == "kg+class":
                for c in class_of[u]:
                    for t in by_class[c]:
                        if t != u:
                            offer(cr, t, scoring.SAME_CLASS, "class", u)
            for key, src in (("kg+svd", "svd"), ("kg+kge", "kge")):
                if mode == key and prep[src]:
                    for t, sim in prep[src].get(u, []):
                        offer(cr, t, scoring.SIM_SCALE * sim, "sim", u)
        return cr

    modes = [m for m in modes if m != "kg+kge" or prep["kge"]]
    hits = {m: {k: 0 for k in ks} for m in modes}
    mrr = {m: 0.0 for m in modes}
    for o in sorted(test):
        tech = [x[0] for x in reqs[o] if x[2] == "technology"]
        known = set(rng.sample(tech, max(1, int(len(tech) * keep))))
        if swap:
            own = {x[0] for x in reqs[o]}
            for u in sorted(known):
                subs = sorted({t for c in cat_of[u] for t in by_cat[c]} - own - known)
                if subs and rng.random() < swap:
                    known.remove(u)
                    known.add(rng.choice(subs))
        for mode in modes:
            cr = credits_for(known, mode)
            scores = sorted(((score_occupation(r, cr, {"technology"})["rank"], code) for code, r in reqs.items()),
                            reverse=True)
            rank = next(i for i, (_, code) in enumerate(scores, 1) if code == o)
            mrr[mode] += 1 / rank
            for k in ks:
                hits[mode][k] += rank <= k
    n = len(test)
    return dict(n=n, swap=swap, hits={m: {k: v / n for k, v in h.items()} for m, h in hits.items()},
                mrr={m: v / n for m, v in mrr.items()})


def report(res):
    ks = list(next(iter(res["hits"].values())))
    lines = [f"held-out occupations: {res['n']}, share of skills swapped for an equivalent tool: {res['swap']:.0%}",
             f"{'system':<10}" + "".join(f"  hit@{k:<3}" for k in ks) + "     MRR"]
    for m, h in res["hits"].items():
        lines.append(f"{m:<10}" + "".join(f"  {h[k]:6.1%}" for k in ks) + f"  {res['mrr'][m]:6.3f}")
    return "\n".join(lines)


def report_lp(lp):
    lines = ["link prediction on held-out occupation->skill edges (filtered ranking among all skills)",
             f"{'model':<20}  hits@1  hits@10     MRR"]
    for m, v in lp.items():
        lines.append(f"{m:<20}  {v['hits1']:5.1%}   {v['hits10']:5.1%}  {v['mrr']:6.3f}")
    return "\n".join(lines)
