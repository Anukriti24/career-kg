"""Learned skill embeddings: truncated SVD (LSA) of the IDF-weighted occupation x technology matrix.

Skills that are used by similar sets of occupations end up close in vector space; the nearest
neighbours become SIMILAR edges in the knowledge graph (a lightweight, link-prediction style
completion of the graph). The 0.85 cosine threshold keeps only confident edges (TensorFlow ~ PyTorch);
lower thresholds added noisy edges that did not improve held-out ranking. Skills used by more than
20% of occupations (Excel, Word, ...) are left out: they are 'similar' to each other only because everyone uses them."""
import numpy as np


def compute_similar(ds, exclude_occ=frozenset(), dim=64, top=8, min_df=3, max_df_frac=0.2, threshold=0.85):
    """Return {skill_id: [(other_skill_id, cosine), ...]} using only occupations not in exclude_occ."""
    occs = [o for o in ds.occupations if o not in exclude_occ]
    oidx = {o: i for i, o in enumerate(occs)}
    df = {}
    for (o, s), _ in ds.requires.items():
        if o in oidx and s.startswith("tech:"):
            df[s] = df.get(s, 0) + 1
    skills = sorted(s for s, n in df.items() if min_df <= n <= max_df_frac * len(occs))
    sidx = {s: j for j, s in enumerate(skills)}
    if not skills:
        return {}
    X = np.zeros((len(occs), len(skills)), dtype=np.float32)
    for (o, s), v in ds.requires.items():
        if o in oidx and s in sidx:
            X[oidx[o], sidx[s]] = v["weight"]
    idf = np.log(len(occs) / np.array([df[s] for s in skills], dtype=np.float32))
    X *= idf
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    k = min(dim, len(S))
    E = Vt[:k].T * S[:k]
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    sim = E @ E.T
    np.fill_diagonal(sim, -1)
    out = {}
    for j, s in enumerate(skills):
        nn = np.argpartition(-sim[j], top)[:top]
        pairs = [(skills[i], float(sim[j, i])) for i in nn if sim[j, i] >= threshold]
        if pairs:
            out[s] = sorted(pairs, key=lambda x: -x[1])
    return out
