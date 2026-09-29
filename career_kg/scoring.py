"""Scoring model shared by the Cypher-backed recommender and the in-memory evaluation.

Every skill the user has earns *credit* (0..1) toward each skill an occupation requires:
  exact match                           1.0
  learned SIMILAR edge                  cosine * SIM_SCALE
  same UNSPSC commodity                 SAME_CATEGORY   (e.g. Tableau / Power BI)
  same UNSPSC class (rule-derived)      SAME_CLASS
  member of a category the user named   CATEGORY_MEMBER (transitive over IS_A / SUBCLASS_OF)
Requirement strength is weight * idf, so ubiquitous skills (Word, Excel) count for little.
  got      = sum(strength * credit)
  coverage = got / sum(strength)          shown to the user as "match"
  rank     = got / sqrt(sum(strength))    used for ordering; a length-normalised compromise between
                                          raw coverage (favours tiny occupations) and raw overlap
                                          (favours huge ones). Exponent 0.5 and the credits below were
                                          tuned on held-out occupations (see evaluate.py)."""
import math

SAME_CATEGORY = 0.2
SAME_CLASS = 0.08           # rule-derived: same UNSPSC class, different commodity
SIM_SCALE = 0.15
CATEGORY_MEMBER = 0.7


def idf(n_occupations, df):
    return math.log(n_occupations / df)


def offer(credits, target, credit, why, src):
    """Keep the best credit per target skill: credits[target] = (credit, why, source skill id)."""
    if credit > credits.get(target, (0.0,))[0]:
        credits[target] = (credit, why, src)


def score_occupation(requirements, credits, types):
    """requirements: iterable of (skill_id, name, type, weight, idf). Returns scores + explanation lists."""
    total = got = 0.0
    matched, inferred, missing = [], [], []
    for sid, name, typ, w, idf_ in sorted(requirements, key=lambda r: -r[3] * r[4]):
        if typ not in types:
            continue
        strength = w * idf_
        total += strength
        c = credits.get(sid)
        if c is None:
            missing.append(dict(id=sid, name=name, type=typ, weight=w, strength=strength))
            continue
        got += strength * c[0]
        item = dict(id=sid, name=name, type=typ, weight=w, credit=round(c[0], 2), why=c[1], source=c[2])
        (matched if c[0] >= 1.0 else inferred).append(item)
    return dict(rank=got / math.sqrt(total) if total else 0.0, coverage=got / total if total else 0.0,
                matched=matched, inferred=inferred, missing=missing)
