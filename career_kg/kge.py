"""ComplEx knowledge graph embeddings trained on the career graph (PyTorch, CPU).

Entities: occupations, skills, categories.  Relations: requires, uses (non-core technologies), is_a,
subclass_of, related_to. Trained 1-N with reciprocal relations and BCE loss (Lacroix et al. 2018 style).
Used for (1) link prediction = KG completion of missing occupation->skill edges, and
(2) skill/skill similarity from the learned entity vectors."""
import random
from collections import defaultdict

import numpy as np
import torch
from torch import nn

REL = ["requires", "uses", "is_a", "subclass_of", "related_to"]


class Triples:
    def __init__(self, ds, exclude_occ=frozenset(), holdout_frac=0.0, seed=0):
        rng = random.Random(seed)
        self.ents = sorted(ds.occupations) + sorted(ds.skills) + sorted(ds.categories)
        self.eid = {e: i for i, e in enumerate(self.ents)}
        self.n_occ, self.n_skill = len(ds.occupations), len(ds.skills)
        self.skill_range = (self.n_occ, self.n_occ + self.n_skill)
        rid = {r: i for i, r in enumerate(REL)}
        train, test = [], []
        for (o, s), v in sorted(ds.requires.items()):
            if o in exclude_occ:
                continue
            t = (self.eid[o], rid["requires" if v["core"] else "uses"], self.eid[s])
            if v["core"] and holdout_frac and rng.random() < holdout_frac:
                test.append(t)
            else:
                train.append(t)
        train += [(self.eid[s], rid["is_a"], self.eid[c]) for s, c in ds.is_a if c in self.eid]
        train += [(self.eid[a], rid["subclass_of"], self.eid[b]) for a, b in ds.subclass_of]
        train += [(self.eid[a], rid["related_to"], self.eid[b]) for a, b, _, _ in ds.related]
        self.train, self.test = train, test


class ComplEx(nn.Module):
    def __init__(self, n_ent, n_rel, dim=200, dropout=0.3):
        super().__init__()
        self.dim = dim
        self.ent = nn.Embedding(n_ent, 2 * dim)
        self.rel = nn.Embedding(2 * n_rel, 2 * dim)      # second half = inverse relations
        nn.init.normal_(self.ent.weight, std=0.1)
        nn.init.normal_(self.rel.weight, std=0.1)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, r):
        """Scores of (h, r, t) for every entity t -> [B, n_ent] logits."""
        d = self.dim
        eh, er = self.drop(self.ent(h)), self.drop(self.rel(r))
        hr, hi, rr, ri = eh[:, :d], eh[:, d:], er[:, :d], er[:, d:]
        qr, qi = hr * rr - hi * ri, hr * ri + hi * rr
        E = self.ent.weight
        return qr @ E[:, :d].T + qi @ E[:, d:].T


def train(tr: Triples, dim=200, epochs=30, batch=512, lr=3e-3, smoothing=0.1, dropout=0.2, seed=0, log=print):
    torch.manual_seed(seed)
    n_rel = len(REL)
    groups = defaultdict(list)
    for h, r, t in tr.train:
        groups[(h, r)].append(t)
        groups[(t, r + n_rel)].append(h)
    keys = list(groups)
    model = ComplEx(len(tr.ents), n_rel, dim, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    n_ent = len(tr.ents)
    rng = random.Random(seed)
    for ep in range(1, epochs + 1):
        rng.shuffle(keys)
        model.train()
        total = 0.0
        for i in range(0, len(keys), batch):
            ks = keys[i:i + batch]
            h = torch.tensor([k[0] for k in ks])
            r = torch.tensor([k[1] for k in ks])
            y = torch.zeros(len(ks), n_ent)
            for j, k in enumerate(ks):
                y[j, groups[k]] = 1.0
            y = y * (1 - smoothing) + smoothing / n_ent
            loss = nn.functional.binary_cross_entropy_with_logits(model(h, r), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(ks)
        if ep % 25 == 0 or ep == 1:
            log(f"  epoch {ep:3d}  loss {total / len(keys):.5f}")
    model.eval()
    return model


@torch.no_grad()
def score_tails(model, tr: Triples, occ_idx, rel="requires"):
    """Logits of every entity as tail of (occupation, rel, ?) -> ndarray [len(occ_idx), n_ent]."""
    h = torch.tensor(occ_idx)
    r = torch.full_like(h, REL.index(rel))
    return model(h, r).numpy()


def link_prediction(model, tr: Triples, ds):
    """Filtered ranking of the true skill among all skills for held-out (occupation, requires, skill) triples,
    against two baselines: global popularity and 'what do related occupations require'."""
    lo, hi = tr.skill_range
    known = defaultdict(set)
    for h, r, t in tr.train + tr.test:
        if r in (0, 1):
            known[h].add(t)
    pop = np.zeros(len(tr.ents))
    nbr = defaultdict(list)
    for h, r, t in tr.train:
        if r == 0:
            pop[t] += 1
        if r == 4:
            nbr[h].append(t)
    req_of = defaultdict(list)
    for h, r, t in tr.train:
        if r == 0:
            req_of[h].append(t)
    res = {m: [] for m in ("complex", "popularity", "related-occupations")}
    by_h = defaultdict(list)
    for h, _, t in tr.test:
        by_h[h].append(t)
    for h, tails in by_h.items():
        s_kge = score_tails(model, tr, [h])[0]
        s_nbr = np.zeros(len(tr.ents))
        for o2 in nbr[h]:
            for t in req_of[o2]:
                s_nbr[t] += 1
        for name, sc in (("complex", s_kge), ("popularity", pop), ("related-occupations", s_nbr + 1e-6 * pop)):
            for t in tails:
                sc2 = sc[lo:hi].copy()
                for k in known[h]:
                    if k != t and lo <= k < hi:
                        sc2[k - lo] = -np.inf
                res[name].append(1 + int((sc2 > sc2[t - lo]).sum()))
    out = {}
    for name, ranks in res.items():
        r = np.array(ranks)
        out[name] = dict(mrr=float((1 / r).mean()), hits1=float((r <= 1).mean()), hits10=float((r <= 10).mean()), n=len(r))
    return out


def skill_similar(model, tr: Triples, ds, min_df=3, max_df_frac=0.2, top=8, threshold=0.8, exclude_occ=frozenset()):
    """{skill_id: [(skill_id, cosine)]} from ComplEx skill vectors; same filters as similarity.compute_similar."""
    df = defaultdict(int)
    for (o, s), v in ds.requires.items():
        if v["core"] and o not in exclude_occ and s.startswith("tech:"):
            df[s] += 1
    n = len(ds.occupations) - len(exclude_occ)
    skills = sorted(s for s, c in df.items() if min_df <= c <= max_df_frac * n)
    E = model.ent.weight.detach().numpy()[[tr.eid[s] for s in skills]]
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    sim = E @ E.T
    np.fill_diagonal(sim, -1)
    out = {}
    for j, s in enumerate(skills):
        nn_ = np.argpartition(-sim[j], top)[:top]
        pairs = [(skills[i], float(sim[j, i])) for i in nn_ if sim[j, i] >= threshold]
        if pairs:
            out[s] = sorted(pairs, key=lambda x: -x[1])
    return out


def predict_missing(model, tr: Triples, ds, top=10):
    """Top-N technologies per occupation that the model believes are needed but the graph does not list."""
    lo, hi = tr.skill_range
    known = defaultdict(set)
    for h, r, t in tr.train + tr.test:
        if r in (0, 1):
            known[h].add(t)
    tech_mask = np.array([ds.skills[tr.ents[i]]["type"] == "technology" for i in range(lo, hi)])
    rows = []
    occs = list(range(tr.n_occ))
    for i in range(0, len(occs), 128):
        chunk = occs[i:i + 128]
        logits = score_tails(model, tr, chunk)[:, lo:hi]
        probs = 1 / (1 + np.exp(-logits))
        for row, h in zip(probs, chunk):
            row = np.where(tech_mask, row, -1.0)
            for k in known[h]:
                if lo <= k < hi:
                    row[k - lo] = -1.0
            for j in np.argsort(-row)[:top]:
                rows.append((tr.ents[h], tr.ents[lo + j], float(row[j])))
    return rows


def save(model, tr: Triples, path, meta=None):
    """Weights plus everything needed to interpret them (entity order, relation names, hyper-parameters)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=model.state_dict(), entities=tr.ents, relations=REL, dim=model.dim,
                    n_train_triples=len(tr.train), meta=meta or {}), path)


def load(path):
    blob = torch.load(path, weights_only=False)
    model = ComplEx(len(blob["entities"]), len(blob["relations"]), blob["dim"])
    model.load_state_dict(blob["state"])
    model.eval()
    return model, blob
