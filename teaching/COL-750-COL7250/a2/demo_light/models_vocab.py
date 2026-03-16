"""
models_vocab.py  -  Core data structures for DEMO_LIGHT.
Translated from ModelsVocab.hs by Jan van Eijck.

Key differences from DEMO_S5:
  - Relations are stored as flat (Agent, state, state) triples, not partitions.
  - Vocabulary is explicit: each EpistM carries a list of propositions.
  - Truth evaluation is 3-valued (True / False / None) because a proposition
    outside the vocabulary has no defined truth value.
  - bisim = convert(minimalModel(gsm(m))) computes the bisimulation-minimal
    model and relabels its states as integers 0, 1, 2, ...
"""

from __future__ import annotations
from functools import reduce


# ---------------------------------------------------------------------------
# Propositional atoms:  P(n), Q(n), R(n)
# ---------------------------------------------------------------------------

class _PrpAtom:
    """A propositional atom.  P(0), Q(0), R(0), P(1), ..."""
    def __init__(self, letter: str, n: int):
        self.letter = letter
        self.n = n

    def __eq__(self, other):
        return isinstance(other, _PrpAtom) and self.letter == other.letter and self.n == other.n

    def __hash__(self):
        return hash((self.letter, self.n))

    def __repr__(self):
        return self.letter if self.n == 0 else f"{self.letter}{self.n}"

    # Ordering: P < Q < R, and P(i) < P(j) iff i < j
    def _key(self):
        return (self.letter, self.n)

    def __lt__(self, other): return self._key() < other._key()
    def __le__(self, other): return self._key() <= other._key()


def P(n=0): return _PrpAtom('p', n)
def Q(n=0): return _PrpAtom('q', n)
def R(n=0): return _PrpAtom('r', n)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class Agent:
    _names = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}

    def __init__(self, n: int):
        self.n = n

    def __eq__(self, other):
        return isinstance(other, Agent) and self.n == other.n

    def __hash__(self):
        return hash(self.n)

    def __repr__(self):
        return self._names.get(self.n, f"a{self.n}")

    def __lt__(self, other):
        return self.n < other.n


a = alice = Agent(0)
b = bob   = Agent(1)
c = carol = Agent(2)
d = dave  = Agent(3)
e = ernie = Agent(4)


# ---------------------------------------------------------------------------
# Formulas
# ---------------------------------------------------------------------------

class Top:
    """Tautology - always true."""
    def __repr__(self): return "T"
    def _ord_key(self): return (0,)


class Prp:
    """
    Formula wrapping a propositional atom.
    Usage: Prp(P(0)), Prp(Q(1)), etc.
    Matches Haskell: data Form = ... | Prp Prp | ...
    """
    def __init__(self, atom: _PrpAtom):
        self.atom = atom

    def __eq__(self, other):
        return isinstance(other, Prp) and self.atom == other.atom

    def __hash__(self):
        return hash(('Prp', self.atom))

    def __repr__(self):
        return repr(self.atom)

    def _ord_key(self): return (1, self.atom._key())
    def __lt__(self, other): return self._ord_key() < other._ord_key()
    def __le__(self, other): return self._ord_key() <= other._ord_key()


class Neg:
    """Negation."""
    def __init__(self, f):
        self.f = f

    def __repr__(self): return f"-{self.f!r}"
    def _ord_key(self): return (2,)


class Conj:
    """Conjunction of a list of formulas."""
    def __init__(self, fs):
        self.fs = list(fs)

    def __repr__(self): return f"&{self.fs!r}"
    def _ord_key(self): return (3,)


class Disj:
    """Disjunction of a list of formulas."""
    def __init__(self, fs):
        self.fs = list(fs)

    def __repr__(self): return f"v{self.fs!r}"
    def _ord_key(self): return (4,)


class K:
    """Agent knowledge: K(agent, form)."""
    def __init__(self, agent: Agent, f):
        self.agent = agent
        self.f = f

    def __repr__(self): return f"[{self.agent!r}]{self.f!r}"
    def _ord_key(self): return (5,)


class CK:
    """Common knowledge: CK(agents, form)."""
    def __init__(self, agents, f):
        self.agents = list(agents)
        self.f = f

    def __repr__(self): return f"C{self.agents!r}{self.f!r}"
    def _ord_key(self): return (6,)


def impl(f1, f2):
    """Material implication f1 => f2."""
    return Disj([Neg(f1), f2])

def equiv(f1, f2):
    """Equivalence f1 <=> f2."""
    return Conj([impl(f1, f2), impl(f2, f1)])

def negation(f):
    """Smart negation: strips double negation."""
    return f.f if isinstance(f, Neg) else Neg(f)


# ---------------------------------------------------------------------------
# Epistemic Model
# ---------------------------------------------------------------------------

class EpistM:
    """
    An S5 epistemic model.

    states : list of worlds
    agents : list of Agent
    voc    : list of _PrpAtom  -  the vocabulary (propositions that exist)
    val    : dict { state -> [_PrpAtom] }  -  props true at each world
    rels   : list of (Agent, state, state) triples  -  accessibility relation
    actual : list of actual worlds
    """
    def __init__(self, states, agents, voc, val, rels, actual):
        self.states = list(states)
        self.agents = list(agents)
        self.voc    = list(voc)
        # Accept both dict and list-of-pairs for val
        self.val    = dict(val) if not isinstance(val, dict) else val
        self.rels   = list(rels)
        self.actual = list(actual)

    def __repr__(self):
        return (f"EpistM(\n  states={self.states},\n  agents={self.agents},"
                f"\n  voc={self.voc},\n  val={self.val},"
                f"\n  rels={self.rels},\n  actual={self.actual}\n)")

    def __eq__(self, other):
        return (isinstance(other, EpistM) and
                self.states == other.states and self.agents == other.agents and
                self.voc == other.voc and self.val == other.val and
                sorted(self.rels) == sorted(other.rels) and
                self.actual == other.actual)


# ---------------------------------------------------------------------------
# Relation helpers
# ---------------------------------------------------------------------------

def rel(ag: Agent, m: EpistM):
    """Extract the accessibility relation for agent ag as (state, state) pairs."""
    return [(x, y) for (agent, x, y) in m.rels if agent == ag]


def right_s(rel_pairs, x):
    """Successors of x in a relation given as a list of (state, state) pairs."""
    seen = []
    for (y, z) in rel_pairs:
        if y == x and z not in seen:
            seen.append(z)
    seen.sort(key=lambda s: (str(type(s)), s) if not isinstance(s, (int, float)) else (str(type(s)), s))
    return seen


def gen_k(rels, ags):
    """Union of relations for a group of agents."""
    return [(x, y) for (ag, x, y) in rels if ag in ags]


def gen_alts(rels, ags, s):
    """Successors of state s under the union of relations for agents ags."""
    return right_s(gen_k(rels, ags), s)


def _rel_compose(r, s):
    """Relational composition r ; s."""
    result = []
    for (x, y) in r:
        for (u, z) in s:
            if y == u and (x, z) not in result:
                result.append((x, z))
    return result


def _lfp(f, x):
    """Least fixed point: keep applying f until stable."""
    while True:
        fx = f(x)
        if fx == x:
            return x
        x = fx


def _rtc(xs, r):
    """Reflexive transitive closure of r over domain xs."""
    ri = list(set(r + [(x, x) for x in xs]))
    def step(s):
        comp = _rel_compose(s, s)
        combined = list(set(s + comp))
        combined.sort()
        return combined
    return _lfp(step, sorted(list(set(ri))))


def common_k(rels, ags, xs):
    """Reflexive-transitive closure of union of agent relations (for CK)."""
    return _rtc(xs, gen_k(rels, ags))


def common_alts(rels, ags, xs, s):
    """Worlds reachable from s under common knowledge relation for agents ags."""
    return right_s(common_k(rels, ags, xs), s)


# ---------------------------------------------------------------------------
# 3-Valued logic  (None = 'Nothing' in Haskell)
# ---------------------------------------------------------------------------

def maybe_not(v):
    return None if v is None else (not v)


def maybe_and(vs):
    """
    True if all True, False if any False, None if any None and no False.
    """
    result = True
    for v in vs:
        if v is False:
            return False
        if v is None:
            result = None
    return result


def maybe_or(vs):
    """
    True if any True, False if all False, None if any None and no True.
    """
    result = False
    for v in vs:
        if v is True:
            return True
        if v is None:
            result = None
    return result


# ---------------------------------------------------------------------------
# Semantic evaluation (3-valued)
# ---------------------------------------------------------------------------

def is_true_at(m: EpistM, w, f) -> bool | None:
    """
    3-valued truth of formula f at world w in model m.
    Returns True, False, or None (undefined - prop not in vocab).
    """
    if isinstance(f, Top):
        return True

    if isinstance(f, Prp):
        if f.atom not in m.voc:
            return None
        return f.atom in m.val.get(w, [])

    if isinstance(f, Neg):
        return maybe_not(is_true_at(m, w, f.f))

    if isinstance(f, Conj):
        return maybe_and([is_true_at(m, w, fi) for fi in f.fs])

    if isinstance(f, Disj):
        return maybe_or([is_true_at(m, w, fi) for fi in f.fs])

    if isinstance(f, K):
        succs = right_s(rel(f.agent, m), w)
        return maybe_and([is_true_at(m, v, f.f) for v in succs])

    if isinstance(f, CK):
        succs = common_alts(m.rels, f.agents, m.states, w)
        return maybe_and([is_true_at(m, v, f.f) for v in succs])

    raise TypeError(f"Unknown formula type: {type(f)}")


def is_true(m: EpistM, f) -> bool | None:
    """True iff f is true at ALL actual worlds of m."""
    result = maybe_and([is_true_at(m, w, f) for w in m.actual])
    return True if result == True else False


# ---------------------------------------------------------------------------
# Public announcement update (without bisimulation)
# ---------------------------------------------------------------------------

def upd_pa(m: EpistM, f) -> EpistM:
    """Restrict model to worlds where f is true; update relations accordingly."""
    new_states = [s for s in m.states if is_true_at(m, s, f) == True]
    new_val    = {s: ps for s, ps in m.val.items() if s in new_states}
    new_rels   = [(ag, x, y) for (ag, x, y) in m.rels
                  if x in new_states and y in new_states]
    new_actual = [s for s in m.actual if s in new_states]
    return EpistM(new_states, m.agents, m.voc, new_val, new_rels, new_actual)


# ---------------------------------------------------------------------------
# Model generation helpers
# ---------------------------------------------------------------------------

def _power_list(xs):
    """All subsets of xs, sorted by length then lexicographically."""
    if not xs:
        return [[]]
    x, *rest = xs
    sub = _power_list(rest)
    return sub + [[x] + s for s in sub]


def _sort_l(xss):
    """Sort list of lists by length, then lexicographically."""
    return sorted(xss, key=lambda xs: (len(xs), xs))


def initM(agents, props) -> EpistM:
    """
    Generate the blissful-ignorance model: every agent is uncertain about
    every proposition, and all worlds are actual.
    """
    k      = len(props)
    worlds = list(range(2 ** k))
    val    = dict(zip(worlds, _sort_l(_power_list(props))))
    rels   = [(ag, s1, s2) for ag in agents
              for s1 in worlds for s2 in worlds]
    return EpistM(worlds, agents, props, val, rels, worlds)


# ---------------------------------------------------------------------------
# Partition utilities (used by bisim)
# ---------------------------------------------------------------------------

def rel2partition(states, rel_pairs):
    """
    Build a partition of states from an equivalence relation given as pairs.
    Each state's class is: {state} ∪ {y in remaining | (state,y) in rel_pairs}.
    """
    remaining = list(states)
    result = []
    while remaining:
        x = remaining[0]
        cls = [x] + [y for y in remaining[1:] if (x, y) in rel_pairs]
        result.append(cls)
        remaining = [y for y in remaining if y not in cls]
    return result


def _cf2part(xs, r):
    """Partition xs by equivalence relation r (given as a function x,y -> bool)."""
    remaining = list(xs)
    result = []
    while remaining:
        x = remaining[0]
        block = [x] + [y for y in remaining[1:] if r(x, y)]
        rest  = [y for y in remaining[1:] if not r(x, y)]
        result.append(block)
        remaining = rest
    return result


def _bl(partition, x):
    """Find the block in partition containing x."""
    for block in partition:
        if x in block:
            return block
    raise ValueError(f"Element {x!r} not found in partition")


# ---------------------------------------------------------------------------
# Generated submodel (gsm): restrict to states reachable from actual worlds
# ---------------------------------------------------------------------------

def _alternatives(rels, ag, s):
    """Direct successors of state s for agent ag."""
    return [s2 for (a, s1, s2) in rels if a == ag and s1 == s]


def _expand(rels, agents, ys):
    """One-step expansion: all states reachable from ys via any agent."""
    result = list(ys)
    for ag in agents:
        for s in ys:
            for s2 in _alternatives(rels, ag, s):
                if s2 not in result:
                    result.append(s2)
    return result


def _closure(rels, agents, xs):
    """Fixed-point: all states reachable from xs via any sequence of agent steps."""
    return _lfp(lambda ys: _expand(rels, agents, ys), xs)


def gsm(m: EpistM) -> EpistM:
    """Generated submodel: restrict to states reachable from actual worlds."""
    reachable = _closure(m.rels, m.agents, m.actual)
    new_val   = {s: ps for s, ps in m.val.items() if s in reachable}
    new_rels  = [(ag, x, y) for (ag, x, y) in m.rels
                 if x in reachable and y in reachable]
    return EpistM(reachable, m.agents, m.voc, new_val, new_rels, m.actual)


# ---------------------------------------------------------------------------
# Bisimulation minimisation (partition refinement)
# ---------------------------------------------------------------------------

def _same_val(m: EpistM, w1, w2) -> bool:
    return m.val.get(w1, []) == m.val.get(w2, [])


def _acc_blocks(m: EpistM, partition, s, ag: Agent):
    """
    The set of partition-blocks accessible from s for agent ag.
    Returns a list of blocks (each block is a list).
    """
    successors = [y for (a, x, y) in m.rels if a == ag and x == s]
    blocks = []
    for y in successors:
        b = _bl(partition, y)
        if b not in blocks:
            blocks.append(b)
    return blocks


def _same_ab(m: EpistM, partition, s, t) -> bool:
    """True iff s and t have the same accessible blocks for every agent."""
    return all(_acc_blocks(m, partition, s, ag) == _acc_blocks(m, partition, t, ag)
               for ag in m.agents)


def _refine_step(m: EpistM, partition):
    """One partition-refinement step using accessible-block equivalence."""
    new_partition = []
    for block in partition:
        sub_blocks = _cf2part(block, lambda x, y: _same_ab(m, partition, x, y))
        new_partition.extend(sub_blocks)
    return new_partition


def _refine(m: EpistM, partition):
    """Iterate refinement until stable."""
    return _lfp(lambda p: _refine_step(m, p), partition)


def _minimal_model(m: EpistM) -> EpistM:
    """
    Compute the bisimulation-minimal model.
    States in the result are the equivalence classes (lists of original states).
    """
    init_part = _cf2part(m.states, lambda x, y: _same_val(m, x, y))
    final_part = _refine(m, init_part)          # list of blocks

    # f maps each original state to its equivalence class (its block)
    def f(s):
        return tuple(_bl(final_part, s))        # use tuple so it's hashable

    new_states = list(dict.fromkeys(f(s) for s in m.states))  # ordered, deduped
    new_val    = {f(s): ps for s, ps in m.val.items()}         # same val per class
    new_rels   = list(dict.fromkeys(
                   (ag, f(x), f(y)) for (ag, x, y) in m.rels))
    new_actual = list(dict.fromkeys(f(s) for s in m.actual))
    return EpistM(new_states, m.agents, m.voc, new_val, new_rels, new_actual)


def convert(m: EpistM) -> EpistM:
    """Relabel states of m as integers 0, 1, 2, ..."""
    mapping = {s: i for i, s in enumerate(m.states)}
    new_states = list(range(len(m.states)))
    new_val    = {mapping[s]: ps for s, ps in m.val.items()}
    new_rels   = [(ag, mapping[x], mapping[y]) for (ag, x, y) in m.rels]
    new_actual = [mapping[s] for s in m.actual]
    return EpistM(new_states, m.agents, m.voc, new_val, new_rels, new_actual)


def bisim(m: EpistM) -> EpistM:
    """
    Compute the bisimulation-minimal model with integer-labelled states.
    bisim = convert . minimalModel . gsm
    """
    return convert(_minimal_model(gsm(m)))


# ---------------------------------------------------------------------------
# Display utilities
# ---------------------------------------------------------------------------

def show_s5(m: EpistM):
    """Return a list of strings describing the model."""
    lines = [
        str(m.states),
        str(m.voc),
        str(m.val),
    ]
    for ag in m.agents:
        r = rel(ag, m)
        part = rel2partition(m.states, r)
        lines.append(f"  {ag}: {part}")
    lines.append(str(m.actual))
    return lines


def display_s5(m: EpistM):
    print('\n'.join(show_s5(m)))
