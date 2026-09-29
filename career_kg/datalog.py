"""A small semi-naive Datalog engine: recursion, multi-atom heads, existential head variables
(Skolem terms) and arithmetic builtins. Enough to express the rules in rules.dl; not a general system."""
import re
from collections import defaultdict

BUILTINS = {"lt": lambda a, b: a < b, "le": lambda a, b: a <= b, "ge": lambda a, b: a >= b,
            "gt": lambda a, b: a > b, "ne": lambda a, b: a != b}
ATOM = re.compile(r"\s*([a-z_][a-z0-9_]*)\s*\(([^)]*)\)\s*")


def is_var(t):
    return isinstance(t, str) and (t[0].isupper() or t[0] == "_")


def parse_term(tok):
    tok = tok.strip()
    return int(tok) if re.fullmatch(r"-?\d+", tok) else tok


def parse_atoms(text):
    atoms, pos = [], 0
    while pos < len(text):
        m = ATOM.match(text, pos)
        if not m:
            raise ValueError(f"cannot parse atom near: {text[pos:pos + 40]!r}")
        atoms.append((m.group(1), tuple(parse_term(t) for t in m.group(2).split(","))))
        pos = m.end()
        if pos < len(text) and text[pos] == ",":
            pos += 1
    return atoms


def parse_rules(text):
    text = "\n".join(l.split("%")[0] for l in text.splitlines())
    rules = []
    for i, stmt in enumerate(s.strip() for s in text.split(".\n") if s.strip()):
        stmt = stmt.rstrip(".")
        head, body = stmt.split(":-")
        head, body = parse_atoms(head), parse_atoms(body)
        bound = [t for _, args in body for t in args if is_var(t)]
        bound = list(dict.fromkeys(bound))
        head_vars = {t for _, args in head for t in args if is_var(t)}
        # existential objects are keyed by the frontier: body variables that are also used in the head
        rules.append(dict(id=i, head=head, body=body, bound=[v for v in bound if v in head_vars]))
    return rules


class Engine:
    def __init__(self):
        self.facts = defaultdict(set)
        self.index = defaultdict(lambda: defaultdict(list))   # (pred, pos) -> value -> [tuples]

    def add(self, pred, args):
        args = tuple(args)
        if args in self.facts[pred]:
            return False
        self.facts[pred].add(args)
        for pos, v in enumerate(args):
            self.index[(pred, pos)][v].append(args)
        return True

    def _match(self, atoms, i, env, delta_i, delta):
        if i == len(atoms):
            yield env
            return
        pred, args = atoms[i]
        if pred in BUILTINS:
            a, b = (env.get(t, t) if is_var(t) else t for t in args)
            if BUILTINS[pred](a, b):
                yield from self._match(atoms, i + 1, env, delta_i, delta)
            return
        if pred == "add":
            x, y, z = args
            out = (env[x] if is_var(x) else x) + (env[y] if is_var(y) else y)
            if is_var(z) and z not in env:
                yield from self._match(atoms, i + 1, {**env, z: out}, delta_i, delta)
            elif (env[z] if is_var(z) else z) == out:
                yield from self._match(atoms, i + 1, env, delta_i, delta)
            return
        if i == delta_i:
            cands = delta.get(pred, ())
        else:
            cands = None
            for pos, t in enumerate(args):
                v = env.get(t) if is_var(t) else t
                if v is not None:
                    cands = self.index[(pred, pos)].get(v, ())
                    break
            if cands is None:
                cands = self.facts[pred]
        for fact in cands:
            e = dict(env)
            ok = True
            for t, v in zip(args, fact):
                if is_var(t):
                    if e.setdefault(t, v) != v:
                        ok = False
                        break
                elif t != v:
                    ok = False
                    break
            if ok:
                yield from self._match(atoms, i + 1, e, delta_i, delta)

    def run(self, rules, max_iter=50):
        delta = {p: set(f) for p, f in self.facts.items()}
        for _ in range(max_iter):
            new = defaultdict(set)
            for r in rules:
                for i, (pred, _) in enumerate(r["body"]):
                    if pred in BUILTINS or pred == "add" or not delta.get(pred):
                        continue
                    body = r["body"]
                    order = [i] + [j for j, (p, _) in enumerate(body) if j != i and p not in BUILTINS and p != "add"] \
                        + [j for j, (p, _) in enumerate(body) if p in BUILTINS or p == "add"]
                    reordered = [body[j] for j in order]      # delta atom first so later atoms are index lookups
                    for env in self._match(reordered, 0, {}, 0, delta):
                        sk = "sk%d(%s)" % (r["id"], ",".join(map(str, (env[v] for v in r["bound"]))))
                        for hp, hargs in r["head"]:
                            fact = tuple(env.get(t, sk) if is_var(t) else t for t in hargs)
                            if fact not in self.facts[hp]:
                                new[hp].add(fact)
            added = {}
            for p, fs in new.items():
                fresh = {f for f in fs if self.add(p, f)}
                if fresh:
                    added[p] = fresh
            if not added:
                return
            delta = added
        raise RuntimeError("no fixpoint within max_iter")
