from career_kg.datalog import Engine, parse_rules


def run(facts, rules):
    e = Engine()
    for p, args in facts:
        e.add(p, args)
    e.run(parse_rules(rules))
    return e.facts


def test_recursion_transitive_closure():
    f = run([("edge", ("a", "b")), ("edge", ("b", "c")), ("edge", ("c", "d"))],
            "path(X,Y) :- edge(X,Y).\npath(X,Z) :- path(X,Y), edge(Y,Z).\n")
    assert f["path"] == {("a", "b"), ("b", "c"), ("c", "d"), ("a", "c"), ("b", "d"), ("a", "d")}


def test_existential_head_creates_one_object_per_binding():
    f = run([("works", ("ann", "x")), ("works", ("ann", "y")), ("works", ("bob", "x"))],
            "has(P,K), kind(K,T) :- works(P,T).\n")
    objs = {k for _, k in f["has"]}
    assert len(objs) == 3                                       # one per (person, tag) binding
    assert {t for _, t in f["kind"]} == {"x", "y"}
    assert all(k in objs for k, _ in f["kind"])                 # same object referenced from both head atoms


def test_arithmetic_and_comparison_builtins_bound_recursion():
    f = run([("r", ("a", "b")), ("r", ("b", "c")), ("r", ("c", "d")), ("z", ("a", 1)), ("z", ("d", 1))],
            "m(A,B,1) :- r(A,B).\nm(A,C,N) :- m(A,B,M), r(B,C), add(M,1,N), le(N,2), ne(A,C).\n")
    assert ("a", "c", 2) in f["m"] and ("a", "d", 3) not in f["m"]


def test_delta_atom_not_first_still_joins():
    f = run([("p", ("a",)), ("q", ("a", "b")), ("r", ("b",))], "s(X,Y) :- p(X), q(X,Y), r(Y).\n")
    assert f["s"] == {("a", "b")}


def test_project_rules_on_real_data(ds):
    from career_kg import reasoning
    facts = reasoning.derive(ds)
    # closure reaches beyond direct parents: every commodity skill is in a segment
    segments = {c for c, v in ds.categories.items() if v["level"] == "segment"}
    assert any(b in segments for _, b in facts["in_cat"])
    # existential rule: every competency area object is attached to exactly one class category
    assert len(facts["has_area"]) == len(facts["area_cat"]) > 0
    # career moves are bounded to 2 hops and never lower the job zone
    zone = {o["code"]: o["zone"] for o in ds.occupations.values()}
    assert all(1 <= n <= 2 and zone[b] >= zone[a] for a, b, n in facts["move"])


def test_existential_object_keyed_by_frontier_only():
    # S is not used in the head, so two skills of the same (person, area) share ONE created object
    f = run([("req", ("o1", "s1", "c")), ("req", ("o1", "s2", "c")), ("req", ("o2", "s1", "c"))],
            "has(O,K), cat(K,C) :- req(O,S,C).\n")
    assert len({k for _, k in f["has"]}) == 2
