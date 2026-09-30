"""Small relational graph convolutional network (R-GCN, Schlichtkrull et al. 2018) for link prediction.

Unlike ComplEx (one free vector per node, no neighbourhood), every node here is re-computed from its
typed neighbours: h' = relu(W_self h + sum_r  A_r h W_r), with A_r the degree-normalised adjacency of
relation r (inverse relations included). Two such layers give each occupation a view of the skills,
categories and related occupations up to two hops away. A DistMult decoder scores (occupation, requires, skill).

Training trick: each epoch half of the requires/uses edges carry messages, the other half are the
prediction targets, so the network cannot simply read the answer off its own neighbours.
Shares Triples / score_tails / link_prediction with kge.py so results are directly comparable to ComplEx."""
import random

import torch
from torch import nn

from . import kge
from .kge import REL, Triples


def _adjacency(edges, n_ent, n_rel):
    """One row-normalised sparse matrix per (relation, direction) from (h, r, t) triples."""
    e = torch.as_tensor(edges, dtype=torch.long).reshape(-1, 3)
    mats = []
    for direction in (0, 1):
        for r in range(n_rel):
            sel = e[e[:, 1] == r]
            if not len(sel):
                mats.append(None)
                continue
            src, dst = (sel[:, 2], sel[:, 0]) if direction == 0 else (sel[:, 0], sel[:, 2])
            deg = torch.bincount(dst, minlength=n_ent).clamp(min=1).float()
            mats.append(torch.sparse_coo_tensor(torch.stack([dst, src]), 1.0 / deg[dst], (n_ent, n_ent)).coalesce())
    return mats


class RGCN(nn.Module):
    def __init__(self, n_ent, n_rel, dim=64, layers=2, dropout=0.2):
        super().__init__()
        self.n_rel, self.dim = n_rel, dim
        self.emb = nn.Embedding(n_ent, dim)
        self.self_w = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(layers))
        self.rel_w = nn.ParameterList(nn.Parameter(torch.empty(2 * n_rel, dim, dim)) for _ in range(layers))
        self.dec = nn.Embedding(2 * n_rel, dim)               # DistMult relation vectors
        self.drop = nn.Dropout(dropout)
        for w in self.rel_w:
            nn.init.xavier_uniform_(w)
        nn.init.normal_(self.emb.weight, std=0.1)
        nn.init.normal_(self.dec.weight, std=0.1)
        self.adj, self.H = None, None

    def encode(self, adj):
        h = self.emb.weight
        for layer, (sw, rw) in enumerate(zip(self.self_w, self.rel_w)):
            out = sw(h)
            for k, a in enumerate(adj):
                if a is not None:
                    out = out + torch.sparse.mm(a, h @ rw[k])
            h = self.drop(torch.relu(out)) if layer < len(self.self_w) - 1 else out
        return h

    def decode(self, H, h, r):
        return (H[h] * self.dec(r)) @ H.T

    def forward(self, h, r):
        """Same contract as kge.ComplEx: logits of every entity as tail of (h, r, ?). Uses the cached encoding."""
        return self.decode(self.H, h, r)

    @torch.no_grad()
    def freeze(self, adj):
        self.eval()
        self.H = self.encode(adj)


def train(tr: Triples, dim=64, layers=2, epochs=40, lr=1e-2, dropout=0.2, negatives=50, seed=0, log=print):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    n_ent, n_rel = len(tr.ents), len(REL)
    model = RGCN(n_ent, n_rel, dim, layers, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    fixed = [t for t in tr.train if t[1] >= 2]                # taxonomy and related_to: always messages
    facts = [t for t in tr.train if t[1] < 2]                 # occupation->skill: split messages / targets
    for ep in range(1, epochs + 1):
        rng.shuffle(facts)
        half = len(facts) // 2
        targets, msgs = facts[:half], facts[half:]
        adj = _adjacency(fixed + msgs, n_ent, n_rel)
        model.train()
        H = model.encode(adj)
        tg = torch.as_tensor(targets, dtype=torch.long)
        q = H[tg[:, 0]] * model.dec(tg[:, 1])
        pos = (q * H[tg[:, 2]]).sum(1)
        neg_t = torch.randint(*tr.skill_range, (len(targets), negatives))   # sampled negative skills
        neg = (q.unsqueeze(1) * H[neg_t]).sum(2)
        loss = nn.functional.softplus(-pos).mean() + nn.functional.softplus(neg).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % 10 == 0 or ep == 1:
            log(f"  epoch {ep:3d}  loss {loss.item():.4f}")
    model.freeze(_adjacency(tr.train, n_ent, n_rel))          # inference: every training edge carries messages
    return model


def split(ds, n_test=300, seed=0):
    """The same held-out occupations and 10% edge hold-out that evaluate.prepare uses."""
    rng = random.Random(seed)
    tech = {}
    for (o, s), v in ds.requires.items():
        if v["core"] and ds.skills[s]["type"] == "technology":
            tech[o] = tech.get(o, 0) + 1
    # evaluate.py samples from dict insertion order of reqs (core rows of any type), keep that order
    order = dict.fromkeys(o for (o, _), v in ds.requires.items() if v["core"])
    candidates = [o for o in order if tech.get(o, 0) >= 6]
    test = set(rng.sample(candidates, min(n_test, len(candidates))))
    return Triples(ds, exclude_occ=test, holdout_frac=0.1)


def compare(ds, epochs=40, dim=64, log=print):
    """Train the R-GCN on the evaluation split and rank it against ComplEx's baselines (popularity, related)."""
    tr = split(ds)
    model = train(tr, dim=dim, epochs=epochs, log=log)
    res = kge.link_prediction(model, tr, ds)
    res["r-gcn"] = res.pop("complex")
    return res
