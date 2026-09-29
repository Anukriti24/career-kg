"""Run rules.dl over the parsed O*NET graph and return the derived facts."""
from pathlib import Path

from .datalog import Engine, parse_rules

RULES = Path(__file__).with_name("rules.dl")


def derive(ds, max_related_rank=5):
    eng = Engine()
    for (o, s), v in ds.requires.items():
        if v["core"]:
            eng.add("requires", (o, s))
    for s, c in ds.is_a:
        if c in ds.categories:
            eng.add("is_a", (s, c))
    for a, b in ds.subclass_of:
        eng.add("subclass_of", (a, b))
    for c in ds.categories.values():
        eng.add("level", (c["code"], c["level"]))
    for o in ds.occupations.values():
        if o["zone"]:
            eng.add("zone", (o["code"], o["zone"]))
    for a, b, _tier, rank in ds.related:
        if rank <= max_related_rank:
            eng.add("related", (a, b))
    eng.run(parse_rules(RULES.read_text()))
    return eng.facts
