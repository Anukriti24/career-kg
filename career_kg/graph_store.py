"""Neo4j connection and bulk loading of the knowledge graph.

Graph schema
  (:Occupation {code, title, description, zone, alt_titles, total_technology, total_skill, total_knowledge})
  (:Skill {id, name, key, type: technology|skill|knowledge, idf, n_occ})
  (:Category {code, name, key, level: commodity|class|family|segment})
  (Occupation)-[:REQUIRES {weight}]->(Skill)
  (Skill)-[:IS_A]->(Category commodity)
  (Category)-[:SUBCLASS_OF]->(Category)          commodity -> class -> family -> segment
  (Skill)-[:SIMILAR {sim}]->(Skill)              learned from embeddings
  (Occupation)-[:RELATED_TO {tier, rank}]->(Occupation)   O*NET career transitions
Derived by the Datalog rules (rules.dl):
  (Skill)-[:IN_CAT]->(Category)                  membership in every ancestor category (recursive closure)
  (Occupation)-[:HAS_AREA]->(:CompetencyArea)-[:AREA_CAT]->(Category class)    created objects (existential rule)
  (Occupation)-[:CAN_MOVE_TO {hops}]->(Occupation)        <=2-hop moves that never drop a job zone
Learned by the ComplEx model (`python -m career_kg kge`):
  (Occupation)-[:PREDICTED_REQUIRES {p}]->(Skill)         link-prediction: likely-needed but unlisted
"""
from neo4j import GraphDatabase

from . import config
from .dataset import compute_idf

BATCH = 5000


def connect():
    if not config.NEO4J_URI or not config.NEO4J_PASSWORD:
        raise RuntimeError("NEO4J_URI / NEO4J_PASSWORD not set: copy .env.example to .env and fill it in")
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    driver.verify_connectivity()
    return driver


def _batched(session, query, rows):
    for i in range(0, len(rows), BATCH):
        session.run(query, rows=rows[i:i + BATCH]).consume()


def load(driver, ds, similar, facts=None):
    idf = compute_idf(ds)
    with driver.session(database=config.NEO4J_DATABASE) as s:
        s.run("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS").consume()
        for q in [
            "CREATE CONSTRAINT occ_code IF NOT EXISTS FOR (o:Occupation) REQUIRE o.code IS UNIQUE",
            "CREATE CONSTRAINT skill_id IF NOT EXISTS FOR (s:Skill) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT cat_code IF NOT EXISTS FOR (c:Category) REQUIRE c.code IS UNIQUE",
            "CREATE INDEX skill_key IF NOT EXISTS FOR (s:Skill) ON (s.key)",
            "CREATE INDEX cat_key IF NOT EXISTS FOR (c:Category) ON (c.key)",
        ]:
            s.run(q).consume()

        _batched(s, "UNWIND $rows AS r CREATE (:Occupation {code:r.code, title:r.title, description:r.description,"
                    " zone:r.zone, alt_titles:r.alt_titles, total_technology:0, total_skill:0, total_knowledge:0})",
                 list(ds.occupations.values()))
        _batched(s, "UNWIND $rows AS r CREATE (:Skill {id:r.id, name:r.name, key:r.key, type:r.type, idf:r.idf, n_occ:0})",
                 [{**k, "idf": idf.get(k["id"], 0.0)} for k in ds.skills.values()])
        _batched(s, "UNWIND $rows AS r CREATE (:Category {code:r.code, name:r.name, key:r.key, level:r.level})",
                 list(ds.categories.values()))
        _batched(s, "UNWIND $rows AS r MATCH (a:Skill {id:r.a}),(b:Category {code:r.b}) CREATE (a)-[:IS_A]->(b)",
                 [dict(a=a, b=b) for a, b in ds.is_a if b in ds.categories])
        _batched(s, "UNWIND $rows AS r MATCH (a:Category {code:r.a}),(b:Category {code:r.b}) CREATE (a)-[:SUBCLASS_OF]->(b)",
                 [dict(a=a, b=b) for a, b in ds.subclass_of])
        _batched(s, "UNWIND $rows AS r MATCH (o:Occupation {code:r.o}),(k:Skill {id:r.s}) "
                    "CREATE (o)-[:REQUIRES {weight:r.w}]->(k)",
                 [dict(o=o, s=k, w=v["weight"]) for (o, k), v in ds.requires.items() if v["core"]])
        _batched(s, "UNWIND $rows AS r MATCH (a:Skill {id:r.a}),(b:Skill {id:r.b}) CREATE (a)-[:SIMILAR {sim:r.sim}]->(b)",
                 [dict(a=a, b=b, sim=sim) for a, ns in similar.items() for b, sim in ns])
        _batched(s, "UNWIND $rows AS r MATCH (a:Occupation {code:r.a}),(b:Occupation {code:r.b}) "
                    "CREATE (a)-[:RELATED_TO {tier:r.tier, rank:r.rank}]->(b)",
                 [dict(a=a, b=b, tier=t, rank=k) for a, b, t, k in ds.related])

        if facts:
            _load_derived(s, ds, facts)
        for t in ("technology", "skill", "knowledge"):
            s.run(f"MATCH (o:Occupation)-[r:REQUIRES]->(k:Skill {{type:'{t}'}}) "
                  f"WITH o, sum(r.weight * k.idf) AS w SET o.total_{t} = w").consume()
        s.run("MATCH (k:Skill) OPTIONAL MATCH (k)<-[r:REQUIRES]-() WITH k, count(r) AS n SET k.n_occ = n").consume()


def _load_derived(s, ds, facts):
    commodity = {c for c, v in ds.categories.items() if v["level"] == "commodity"}
    _batched(s, "UNWIND $rows AS r MATCH (a:Skill {id:r.a}),(b:Category {code:r.b}) CREATE (a)-[:IN_CAT]->(b)",
             [dict(a=a, b=b) for a, b in facts["in_cat"] if b not in commodity])
    s.run("CREATE CONSTRAINT area_id IF NOT EXISTS FOR (a:CompetencyArea) REQUIRE a.id IS UNIQUE").consume()
    _batched(s, "UNWIND $rows AS r MATCH (o:Occupation {code:r.o}) CREATE (o)-[:HAS_AREA]->(:CompetencyArea {id:r.k})",
             [dict(o=o, k=k) for o, k in facts["has_area"]])
    _batched(s, "UNWIND $rows AS r MATCH (a:CompetencyArea {id:r.k}),(c:Category {code:r.c}) CREATE (a)-[:AREA_CAT]->(c)",
             [dict(k=k, c=c) for k, c in facts["area_cat"]])
    hops = {}
    for a, b, n in facts["move"]:
        hops[(a, b)] = min(n, hops.get((a, b), n))
    _batched(s, "UNWIND $rows AS r MATCH (a:Occupation {code:r.a}),(b:Occupation {code:r.b}) "
                "CREATE (a)-[:CAN_MOVE_TO {hops:r.n}]->(b)",
             [dict(a=a, b=b, n=n) for (a, b), n in hops.items()])


def load_kge(driver, predicted):
    with driver.session(database=config.NEO4J_DATABASE) as s:
        s.run("MATCH ()-[r:PREDICTED_REQUIRES]->() CALL (r) { DELETE r } IN TRANSACTIONS OF 10000 ROWS").consume()
        _batched(s, "UNWIND $rows AS r MATCH (o:Occupation {code:r.o}),(k:Skill {id:r.s}) "
                    "CREATE (o)-[:PREDICTED_REQUIRES {p:r.p}]->(k)",
                 [dict(o=o, s=k, p=p) for o, k, p in predicted])


def stats(driver):
    with driver.session(database=config.NEO4J_DATABASE) as s:
        nodes = {r["l"]: r["n"] for r in s.run("MATCH (n) RETURN labels(n)[0] AS l, count(*) AS n")}
        rels = {r["t"]: r["n"] for r in s.run("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS n")}
    return nodes, rels
