"""Download and parse the O*NET database into plain Python structures (no DB needed)."""
import csv
import urllib.request
import zipfile
from dataclasses import dataclass, field

from . import scoring
from .config import ONET_DIR, ONET_URL, RAW_DIR

LEVELS = ("commodity", "class", "family", "segment")


@dataclass
class Dataset:
    occupations: dict = field(default_factory=dict)   # code -> {code,title,description,zone,alt_titles}
    skills: dict = field(default_factory=dict)        # id -> {id,name,key,type}
    categories: dict = field(default_factory=dict)    # code -> {code,name,key,level}
    subclass_of: set = field(default_factory=set)     # (child category code, parent category code)
    is_a: set = field(default_factory=set)            # (skill id, commodity code)
    requires: dict = field(default_factory=dict)      # (occ, skill id) -> {weight, core}
    related: list = field(default_factory=list)       # (occ, occ, tier, rank)


def download(force=False):
    if ONET_DIR.exists() and not force:
        return ONET_DIR
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zpath = RAW_DIR / f"{ONET_DIR.name}.zip"
    print(f"downloading {ONET_URL}")
    urllib.request.urlretrieve(ONET_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(RAW_DIR)
    return ONET_DIR


def _rows(name):
    with open(ONET_DIR / f"{name}.txt", newline="", encoding="utf-8-sig") as f:
        yield from csv.DictReader(f, delimiter="\t")


def tech_id(name):
    return f"tech:{name}"


def general_weight(importance):
    """O*NET importance (1-5) of a general skill/knowledge element -> requirement weight (0 = drop)."""
    return 3 if importance >= 4.5 else 2 if importance >= 4.0 else 1 if importance >= 3.5 else 0


def parse():
    download()
    ds = Dataset()
    zones = {r["O*NET-SOC Code"]: int(r["Job Zone"]) for r in _rows("Job Zones")}
    alt = {}
    for r in _rows("Alternate Titles"):
        alt.setdefault(r["O*NET-SOC Code"], []).append(r["Alternate Title"])
    for r in _rows("Occupation Data"):
        c = r["O*NET-SOC Code"]
        ds.occupations[c] = dict(code=c, title=r["Title"], description=r["Description"],
                                 zone=zones.get(c), alt_titles=alt.get(c, [])[:15])

    # Technology skills: example -> commodity, with hot / in-demand flags
    unspsc = {r["Commodity Code"]: r for r in _rows("UNSPSC Reference")}
    for r in _rows("Technology Skills"):
        occ, name, com = r["O*NET-SOC Code"], r["Example"], r["Commodity Code"]
        if occ not in ds.occupations:
            continue
        sid = tech_id(name)
        ds.skills.setdefault(sid, dict(id=sid, name=name, key=name.lower(), type="technology"))
        weight = 1 + (r["Hot Technology"] == "Y") + (r["In Demand"] == "Y")
        prev = ds.requires.get((occ, sid))
        if prev is None or weight > prev["weight"]:
            ds.requires[(occ, sid)] = dict(weight=weight, core=weight >= 2)
        ds.is_a.add((sid, com))
        u = unspsc.get(com)
        if u:
            chain = [("commodity", com, r["Commodity Title"]), ("class", u["Class Code"], u["Class Title"]),
                     ("family", u["Family Code"], u["Family Title"]), ("segment", u["Segment Code"], u["Segment Title"])]
            for (lvl, code, name_), parent in zip(chain, chain[1:] + [None]):
                ds.categories.setdefault(code, dict(code=code, name=name_, key=name_.lower(), level=lvl))
                if parent:
                    ds.subclass_of.add((code, parent[1]))
        else:
            ds.categories.setdefault(com, dict(code=com, name=r["Commodity Title"],
                                               key=r["Commodity Title"].lower(), level="commodity"))

    # General skills and knowledge areas rated for importance
    for table, kind in (("Skills", "skill"), ("Knowledge", "knowledge")):
        for r in _rows(table):
            if r["Scale ID"] != "IM" or r["O*NET-SOC Code"] not in ds.occupations:
                continue
            if r["Recommend Suppress"] == "Y" or r["Not Relevant"] == "Y":
                continue
            w = general_weight(float(r["Data Value"]))
            sid = f"{kind}:{r['Element ID']}"
            ds.skills.setdefault(sid, dict(id=sid, name=r["Element Name"], key=r["Element Name"].lower(), type=kind))
            if w:
                ds.requires[(r["O*NET-SOC Code"], sid)] = dict(weight=w, core=True)

    for r in _rows("Related Occupations"):
        a, b = r["O*NET-SOC Code"], r["Related O*NET-SOC Code"]
        if a in ds.occupations and b in ds.occupations:
            ds.related.append((a, b, r["Relatedness Tier"], int(r["Index"])))
    return ds


def compute_idf(ds):
    """skill id -> log(N / number of occupations that list it as a core requirement)."""
    df = {}
    for (_, sid), v in ds.requires.items():
        if v["core"]:
            df[sid] = df.get(sid, 0) + 1
    return {sid: scoring.idf(len(ds.occupations), n) for sid, n in df.items()}
