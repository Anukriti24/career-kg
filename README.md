# Career KG — knowledge-graph career recommendation from skills

Enter the technologies you know,  a Neo4j knowledge graph built from the real **O\*NET 30.0** database
(1,016 occupations, 8,843 skills) ranks matching occupations, explains every match, and suggests what to learn next.

## Quick start
```bash
make setup      # venv + dependencies
cp .env.example .env   # fill in your hosted Neo4j (e.g. Aura free tier) URI + password
make ingest     # download O*NET, learn skill embeddings, load the graph (~1 min)
make app        # Streamlit UI -> http://localhost:8501
make test       # 26 tests (integration tests skip if Neo4j is unreachable)
make eval       # held-out evaluation (trains ComplEx once, ~10 min)
.venv/bin/python -m career_kg kge   # after ingest: train ComplEx, save weights, store predictions
.venv/bin/python -m career_kg gnn   # train a small R-GCN and compare its link prediction with the baselines
```
CLI: `.venv/bin/python -m career_kg recommend "Python, TensorFlow, SQL" -k 5 [--no-inference] [--zones 3-5]`.
Neo4j is hosted, nothing runs locally. Credentials come from `.env` (git-ignored) or the `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` environment variables.

## Graph schema
```
(Occupation)-[:REQUIRES {weight 1-3}]->(Skill {type: technology|skill|knowledge, idf})
(Skill)-[:IS_A]->(Category commodity)-[:SUBCLASS_OF]->(class)->(family)->(segment)      UNSPSC taxonomy
(Skill)-[:SIMILAR {sim}]->(Skill)                       learned from co-usage embeddings
(Occupation)-[:RELATED_TO {tier, rank}]->(Occupation)   O*NET career transitions
+ derived: IN_CAT, HAS_AREA -> CompetencyArea -> AREA_CAT, CAN_MOVE_TO;  learned: PREDICTED_REQUIRES
```
~16k nodes, ~115k relationships (within Aura Free limits). Weights: technologies 1 + hot + in-demand (only ≥2 kept as requirements);
general skills / knowledge by O\*NET importance (≥3.5 of 5).

## How a recommendation is made (`career_kg/scoring.py`, `recommender.py`)
Each of your skills earns **credit** toward each skill an occupation requires, found by graph queries:

| Graph pattern | Credit |
|---|---|
| exact skill | 1.0 |
| `(you)-[:IS_A]->(commodity)<-[:IS_A]-(required)` — equivalent tool, e.g. Tableau / Power BI | 0.2 |
| `(you)-[:SIMILAR]->(required)` — e.g. TensorFlow / PyTorch (cosine ≥ 0.85) | 0.15 × cosine |
| you named a category; required skill is a member via `IS_A\|SUBCLASS_OF*1..4` | 0.7 |

Requirement strength = weight × IDF, so Word/Excel matter little. **Coverage** = Σ strength·credit / Σ strength is shown;
occupations are **ranked** by Σ strength·credit / √Σ strength, which balances tiny and huge requirement lists.
Input matching is case-insensitive with aliases ("sql" → *Structured query language SQL*) and suggestions for unknown names.

**Schema of the live graph**
```cypher
CALL db.schema.visualization()
```
![Graph schema](queries/query%201.png)

```cypher
MATCH (a)-[r]->(b)
RETURN DISTINCT labels(a)[0] AS from, type(r) AS relationship, labels(b)[0] AS to
ORDER BY from, relationship;
```
![Relationship patterns](queries/query%202.png)

**1. Resolve the user's input to skill nodes** (`resolve`)
```cypher
MATCH (s:Skill) WHERE s.key IN ['python', 'structured query language sql']
RETURN s.id AS id, s.name AS name, s.type AS type
```
![Resolve skills](queries/query%203.png)

**2. Credit for equivalent tools in the same category** (`credits`, 0.2)
```cypher
MATCH (u:Skill)-[:IS_A]->(c:Category)<-[:IS_A]-(t:Skill)
WHERE u.key IN ['tableau'] AND t <> u
RETURN t.name AS target, u.name AS uname, c.name AS cname
```
![Same category](queries/query%204.png)

**3. Credit for learned similar skills** (`credits`, 0.15 x cosine)
```cypher
MATCH (u:Skill)-[r:SIMILAR]->(t:Skill)
WHERE u.key IN ['tensorflow']
RETURN t.name AS target, u.name AS uname, r.sim AS sim
```
![Similar skills](queries/query%205.png)

**4. Ranking** (`recommend`; exact-match credit only, technologies only)
```cypher
MATCH (s0:Skill) WHERE s0.key IN ['python', 'structured query language sql', 'tensorflow']
WITH collect({id: s0.id, credit: 1.0}) AS cr
UNWIND cr AS c
MATCH (o:Occupation)-[r:REQUIRES]->(s:Skill {id: c.id})
WHERE s.type IN ['technology'] AND (o.zone IS NULL OR (o.zone >= 1 AND o.zone <= 5))
WITH o, sum(r.weight * s.idf * c.credit) AS got, (o.total_technology) AS total
RETURN o.code AS code, o.title AS title, o.zone AS zone,
       got / sqrt(total) AS rank, got / total AS coverage
ORDER BY rank DESC LIMIT 10
```
![Ranking query](queries/query%206.png)

With only three skills, exact matching ranks short tool lists first (see Limitations); the category and similarity credits above are what make results more sensible.

**5. Occupation detail** (`occupation`: requirements, career moves, predicted skills, related occupations)
```cypher
MATCH (o:Occupation {title:'Data Scientists'})-[r:REQUIRES]->(s:Skill)
RETURN s.name AS name, s.type AS type, r.weight AS weight ORDER BY r.weight DESC, s.name;

MATCH (:Occupation {title:'Data Scientists'})-[m:CAN_MOVE_TO]->(o:Occupation)
RETURN o.title AS title, o.zone AS zone, m.hops AS hops ORDER BY m.hops, o.zone DESC, o.title LIMIT 15;

MATCH (:Occupation {title:'Data Scientists'})-[p:PREDICTED_REQUIRES]->(s:Skill)
RETURN s.name AS skill, round(p.p * 1000) / 1000 AS probability ORDER BY p.p DESC LIMIT 10;

MATCH (:Occupation {title:'Data Scientists'})-[r:RELATED_TO]->(o:Occupation)
RETURN o.title AS title, r.tier AS tier ORDER BY r.rank LIMIT 10;
```
![Occupation detail](queries/query%207.png)

## Reasoning rules (`career_kg/rules.dl`, `datalog.py`)
A small semi-naive Datalog engine (recursion, multi-atom heads, existential head variables, arithmetic builtins) runs at ingest:
- **recursion**: transitive closure of the UNSPSC taxonomy -> `(Skill)-[:IN_CAT]->(ancestor Category)`
- **object creation (existential)**: one `CompetencyArea` object per (occupation, class) it draws skills from -> shown as "areas you cover"
- **recursion + arithmetic**: `CAN_MOVE_TO {hops<=2}` career moves that never lower the job zone

## Trained KG embeddings (`career_kg/kge.py`)
ComplEx (PyTorch, CPU, dim 200, ~2.5 min) on occupations, skills and categories with relations requires / uses / is_a / subclass_of / related_to.
`python -m career_kg kge` trains on the full graph, saves the weights to `data/models/complex.pt` (16 MB, git-ignored) and writes
`(Occupation)-[:PREDICTED_REQUIRES {p}]->(Skill)`: technologies the model expects but O*NET does not list (KG completion).
Reload with `kge.load(path)`.

## Graph neural network (`career_kg/gnn.py`)
A two-layer R-GCN (PyTorch, dim 64) recomputes every node from its typed neighbours (requires, uses, is_a, subclass_of,
related_to, plus inverse relations) and scores occupation->skill links with a DistMult decoder. Each epoch half of the
occupation->skill edges carry messages and the other half are the prediction targets, so the network cannot read the answer
off its own neighbours. `python -m career_kg gnn` trains it on the same held-out split as ComplEx (about 4 minutes on CPU).
It is a small experiment: it is not tuned and is not used by the recommender.

## Evaluation (`make eval`, ~10 min, 300 held-out occupations)
All learned components are re-trained *without* the test occupations. A simulated job seeker knows a random half of an occupation's core
technologies; where does the true occupation rank? Second scenario: half the skills are swapped for an equivalent tool of the same category.

| system | hit@1 / hit@10 (exact skills) | hit@1 / hit@10 (half swapped) |
|---|---|---|
| exact match only | 47% / 76% | 20% / 44% |
| + shared category (`kg`) | 48% / 75% | 23% / 50% |
| + rule-derived class credit | 43% / 74% | 18% / 37% |
| + SVD similarity | 48% / 75% | 23% / 49% |
| + ComplEx similarity | 49% / 75% | 23% / 48% |

Link prediction (held-out occupation->skill edges, filtered ranking among all skills):

| model | hits@1 | hits@10 | MRR |
|---|---|---|---|
| ComplEx | 31% | 72% | 0.45 |
| popularity | 27% | 52% | 0.36 |
| skills of related occupations | 37% | 76% | 0.50 |
| R-GCN (dim 64, 40 epochs, untuned, single run) | 17% | 48% | 0.28 |

Analysis:
- Category credit helps only when users name *equivalent* tools (+6 points hit@10); otherwise neutral.
- Rule-derived class-level credit **hurts**, so it is used for structure and explanation (areas, moves), not for scoring.
- Neither SVD nor ComplEx similarity beats plain category credit, so they are not used for ranking.
- ComplEx clearly beats popularity at predicting missing requirements but not the simple "ask the related occupations" baseline.
- A small untuned R-GCN scores below the popularity baseline on link prediction. The related-occupations baseline is already a one-hop neighbourhood aggregation, which is close to what a GNN layer computes, so no gain was found here; a tuned GNN was not tried.
- The swap scenario is derived from the same taxonomy the KG uses, and hyper-parameters were tuned on this split (seed 0): treat numbers as optimistic.

## Layout
```
career_kg/dataset.py      download + parse O*NET          career_kg/scoring.py     credit / score model
career_kg/similarity.py   SVD skill embeddings            career_kg/recommender.py Cypher reasoning + explanations
career_kg/datalog.py + rules.dl  rule engine + rules      career_kg/kge.py         ComplEx training, link prediction, save/load
career_kg/graph_store.py  Neo4j schema + bulk load        career_kg/evaluate.py    held-out evaluation
career_kg/gnn.py          R-GCN link prediction           career_kg/reasoning.py   runs rules.dl over the dataset
app.py                    Streamlit UI                    tests/                   unit + Neo4j integration tests
```

## Limitations
- Technology-centric: O*NET lists tools per occupation, not proficiency; general skills are only used if you enter them.
- Occupations with short tool lists can outrank obvious matches for very small profiles (1-3 skills); add more skills.
- The Datalog engine is deliberately small (no negation or aggregation); ComplEx is trained on the static graph only (no temporal model); the R-GCN is transductive (learned node embeddings) and cannot embed unseen occupations.
- English / US occupation taxonomy (O\*NET-SOC); data licensed CC BY 4.0 by the U.S. Department of Labor.

## Deploying the Streamlit app (Streamlit Community Cloud)


Live app: https://career-kg-ftcvszay9xz8lbk7cm7dfv.streamlit.app/
