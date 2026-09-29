"""Graph-backed career recommender: reasoning happens in Cypher, explanations are assembled here."""
import difflib
import re

from . import config, scoring
from .scoring import offer, score_occupation

TYPE_TOTAL = {"technology": "total_technology", "skill": "total_skill", "knowledge": "total_knowledge"}


class Recommender:
    def __init__(self, driver):
        self.driver = driver
        rows = self._run("MATCH (s:Skill) WHERE s.n_occ > 0 RETURN s.name AS name, s.type AS type, s.n_occ AS n")
        cats = self._run("MATCH (c:Category) RETURN c.name AS name")
        self._catalog = sorted(({"name": r["name"], "type": r["type"], "n": r["n"]} for r in rows),
                               key=lambda x: -x["n"])
        self._catalog += [{"name": r["name"], "type": "category", "n": 0} for r in cats]
        self._names = {c["name"].lower(): c["name"] for c in self._catalog}
        self._tokens = [(c, set(re.findall(r"[a-z0-9+#.]+", c["name"].lower())))
                        for c in self._catalog if c["type"] != "category"]

    def _alias(self, raw):
        """'SQL' -> 'Structured query language SQL': the most widely used skill containing every word."""
        words = set(re.findall(r"[a-z0-9+#.]+", raw.lower()))
        hits = sorted((c for c, t in self._tokens if words and words <= t), key=lambda c: -c["n"])
        if hits and (len(hits) == 1 or hits[0]["n"] >= 1.4 * hits[1]["n"]):
            return hits[0]["name"]
        return None

    def _run(self, query, **params):
        with self.driver.session(database=config.NEO4J_DATABASE) as s:
            return [r.data() for r in s.run(query, **params)]

    # ---- input handling -------------------------------------------------
    def suggest(self, text, n=8):
        t = text.strip().lower()
        if not t:
            return []
        starts = [c for c in self._catalog if c["name"].lower().startswith(t)]
        contains = [c for c in self._catalog if t in c["name"].lower() and c not in starts]
        out = (starts + contains)[:n]
        if len(out) < n:
            close = difflib.get_close_matches(t, list(self._names), n=n - len(out), cutoff=0.7)
            out += [c for c in self._catalog if c["name"].lower() in close and c not in out]
        return out[:n]

    def resolve(self, names):
        keys, aliased = {}, {}
        for n in (n.strip() for n in names):
            if not n:
                continue
            canon = n if n.lower() in self._names else self._alias(n) or n
            if canon != n:
                aliased[canon.lower()] = n
            keys[canon.lower()] = canon
        skills = self._run("MATCH (s:Skill) WHERE s.key IN $k RETURN s.id AS id, s.name AS name, s.type AS type",
                           k=list(keys))
        cats = self._run("MATCH (c:Category) WHERE c.key IN $k RETURN c.code AS code, c.name AS name", k=list(keys))
        found = {s["name"].lower() for s in skills} | {c["name"].lower() for c in cats}
        unknown = [dict(input=raw, suggestions=[s["name"] for s in self.suggest(raw, 4)])
                   for key, raw in keys.items() if key not in found]
        for x in skills + cats:
            if x["name"].lower() in aliased:
                x["typed_as"] = aliased[x["name"].lower()]
        return skills, cats, unknown

    # ---- reasoning ------------------------------------------------------
    def credits(self, skills, cats, inference=True):
        credits = {}
        ids = [s["id"] for s in skills]
        for s in skills:
            offer(credits, s["id"], 1.0, "you have it", s["id"])
        if not inference:
            return credits
        for r in self._run(
                "MATCH (u:Skill)-[:IS_A]->(c:Category)<-[:IS_A]-(t:Skill) WHERE u.id IN $ids AND t <> u "
                "RETURN t.id AS t, u.id AS u, u.name AS uname, c.name AS cname", ids=ids):
            offer(credits, r["t"], scoring.SAME_CATEGORY, f"same category as {r['uname']}: {r['cname']}", r["u"])
        for r in self._run(
                "MATCH (u:Skill)-[r:SIMILAR]->(t:Skill) WHERE u.id IN $ids "
                "RETURN t.id AS t, u.id AS u, u.name AS uname, r.sim AS sim", ids=ids):
            offer(credits, r["t"], scoring.SIM_SCALE * r["sim"], f"similar to {r['uname']} ({r['sim']:.2f})", r["u"])
        for c in cats:
            for r in self._run(
                    "MATCH (t:Skill)-[:IS_A|SUBCLASS_OF*1..4]->(c:Category {code:$code}) RETURN DISTINCT t.id AS t",
                    code=c["code"]):
                offer(credits, r["t"], scoring.CATEGORY_MEMBER, f"part of {c['name']}", "cat:" + c["code"])
        return credits

    def recommend(self, names, k=10, zone_min=1, zone_max=5, inference=True):
        skills, cats, unknown = self.resolve(names)
        types = {s["type"] for s in skills} | ({"technology"} if cats else set())
        out = dict(resolved=[dict(name=s["name"], type=s["type"], typed_as=s.get("typed_as")) for s in skills]
                   + [dict(name=c["name"], type="category", typed_as=c.get("typed_as")) for c in cats],
                   unknown=unknown, results=[], learn=[])
        if not types:
            return out
        credits = self.credits(skills, cats, inference)
        total_expr = " + ".join(f"CASE WHEN '{t}' IN $types THEN o.{TYPE_TOTAL[t]} ELSE 0 END" for t in TYPE_TOTAL)
        ranked = self._run(
            "UNWIND $cr AS c MATCH (o:Occupation)-[r:REQUIRES]->(s:Skill {id:c.id}) "
            "WHERE s.type IN $types AND (o.zone IS NULL OR (o.zone >= $zmin AND o.zone <= $zmax)) "
            f"WITH o, sum(r.weight * s.idf * c.credit) AS got, ({total_expr}) AS total "
            "RETURN o.code AS code, o.title AS title, o.zone AS zone, o.description AS description, "
            "got / sqrt(total) AS rank, got / total AS coverage ORDER BY rank DESC LIMIT $k",
            cr=[dict(id=i, credit=c[0]) for i, c in credits.items()], types=sorted(types),
            zmin=zone_min, zmax=zone_max, k=k)
        reqs = {}
        for r in self._run("MATCH (o:Occupation)-[r:REQUIRES]->(s:Skill) WHERE o.code IN $codes "
                           "RETURN o.code AS code, s.id AS id, s.name AS name, s.type AS type, r.weight AS w, s.idf AS idf",
                           codes=[r["code"] for r in ranked]):
            reqs.setdefault(r["code"], []).append((r["id"], r["name"], r["type"], r["w"], r["idf"]))
        areas = {}
        for r in self._run(
                "MATCH (o:Occupation)-[:HAS_AREA]->(:CompetencyArea)-[:AREA_CAT]->(c:Category) WHERE o.code IN $codes "
                "MATCH (o)-[:REQUIRES]->(s:Skill)-[:IN_CAT]->(c) RETURN o.code AS code, c.name AS area, collect(s.id) AS skills",
                codes=[r["code"] for r in ranked]):
            areas.setdefault(r["code"], []).append(r)
        want = {}
        for r in ranked:
            detail = score_occupation(reqs.get(r["code"], []), credits, types)
            have = {i for i, c in credits.items() if c[0] >= 1.0}
            detail["areas"] = sorted(
                (dict(area=a["area"], covered=len(have & set(a["skills"])), total=len(a["skills"]))
                 for a in areas.get(r["code"], [])), key=lambda x: (-x["covered"], -x["total"]))[:5]
            out["results"].append({**r, **detail, "rank": r["rank"], "coverage": r["coverage"]})
            for m in detail["missing"]:
                w = want.setdefault(m["id"], dict(name=m["name"], type=m["type"], value=0.0, jobs=0))
                w["value"] += m["strength"] * r["coverage"]
                w["jobs"] += 1
        out["learn"] = sorted(want.values(), key=lambda x: -x["value"])[:8]
        return out

    # ---- detail views ---------------------------------------------------
    def occupation(self, code):
        rows = self._run("MATCH (o:Occupation {code:$c}) RETURN o.code AS code, o.title AS title, "
                         "o.description AS description, o.zone AS zone, o.alt_titles AS alt_titles", c=code)
        if not rows:
            return None
        occ = rows[0]
        occ["requirements"] = self._run(
            "MATCH (:Occupation {code:$c})-[r:REQUIRES]->(s:Skill) "
            "RETURN s.name AS name, s.type AS type, r.weight AS weight ORDER BY r.weight DESC, s.name", c=code)
        occ["moves"] = self._run(
            "MATCH (:Occupation {code:$c})-[m:CAN_MOVE_TO]->(o:Occupation) "
            "RETURN o.title AS title, o.zone AS zone, m.hops AS hops ORDER BY m.hops, o.zone DESC, o.title LIMIT 15", c=code)
        occ["predicted"] = self._run(
            "MATCH (:Occupation {code:$c})-[p:PREDICTED_REQUIRES]->(s:Skill) "
            "RETURN s.name AS skill, round(p.p * 1000) / 1000 AS probability ORDER BY p.p DESC LIMIT 10", c=code)
        occ["related"] = self._run(
            "MATCH (:Occupation {code:$c})-[r:RELATED_TO]->(o:Occupation) "
            "RETURN o.code AS code, o.title AS title, r.tier AS tier ORDER BY r.rank LIMIT 10", c=code)
        return occ

    def explain_graph(self, code, names, inference=True, max_req=30):
        """Nodes/edges showing how the user's skills connect to an occupation's requirements."""
        skills, cats, _ = self.resolve(names)
        credits = self.credits(skills, cats, inference)
        reqs = self._run("MATCH (:Occupation {code:$c})-[r:REQUIRES]->(s:Skill) "
                         "RETURN s.id AS id, s.name AS name, s.type AS type, r.weight AS w "
                         "ORDER BY r.weight DESC LIMIT $n", c=code, n=max_req)
        title = self._run("MATCH (o:Occupation {code:$c}) RETURN o.title AS t", c=code)[0]["t"]
        nodes = {"occ": dict(id="occ", label=title, kind="occupation")}
        edges = []
        for r in reqs:
            c = credits.get(r["id"])
            kind = "matched" if c and c[0] >= 1 else "inferred" if c else "missing"
            nodes[r["id"]] = dict(id=r["id"], label=r["name"], kind=kind)
            edges.append(dict(from_="occ", to=r["id"], label=f"w={r['w']}", kind="requires"))
            if c and c[0] < 1:
                src = c[2]
                if src not in nodes:
                    lbl = next((s["name"] for s in skills if s["id"] == src),
                               next((x["name"] for x in cats if "cat:" + x["code"] == src), src))
                    nodes[src] = dict(id=src, label=lbl, kind="user")
                edges.append(dict(from_=src, to=r["id"], label=f"{c[0]:.2f}", kind="infers", title=c[1]))
        return dict(nodes=list(nodes.values()), edges=edges)
