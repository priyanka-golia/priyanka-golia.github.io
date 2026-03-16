"""
demo_s5.py  –  Epistemic model checker for S5 knowledge logic.
Translated from DEMO_S5.hs by Jan van Eijck.

Key ideas
---------
* An epistemic model has a set of possible worlds, one accessibility
  relation per agent (represented as a partition), and a valuation that
  maps each world to the set of propositions true there.
* A formula is evaluated at a world; knowledge Kn(ag, f) is true at w
  when f is true at every world in ag's equivalence block containing w.
* A public announcement restricts the model to the worlds where the
  announced formula is true.
"""

from functools import reduce
from erel import bl, restrict, fct2erel, cfct2erel, restricted_prod


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class Agent:
    """An agent, identified by a non-negative integer."""
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


# Pre-built agents for convenience
a, b, c, d, e = Agent(0), Agent(1), Agent(2), Agent(3), Agent(4)


# ---------------------------------------------------------------------------
# Propositional atoms
# ---------------------------------------------------------------------------

class Prop:
    """A propositional atom such as p, p1, q, q2, …"""

    def __init__(self, letter: str, n: int = 0):
        self.letter = letter
        self.n = n

    def __eq__(self, other):
        return isinstance(other, Prop) and self.letter == other.letter and self.n == other.n

    def __hash__(self):
        return hash((self.letter, self.n))

    def __repr__(self):
        return self.letter if self.n == 0 else f"{self.letter}{self.n}"


# Constructors matching the Haskell names P, Q, R, S
def P(n=0): return Prop("p", n)
def Q(n=0): return Prop("q", n)
def R(n=0): return Prop("r", n)
def S(n=0): return Prop("s", n)


# ---------------------------------------------------------------------------
# Formulas
# ---------------------------------------------------------------------------

class Top:
    """Tautology – always true."""
    def __repr__(self): return "Top"


class Info:
    """True only at one specific world (used instead of ground propositions
    when the world itself encodes all relevant information)."""
    def __init__(self, world):
        self.world = world
    def __repr__(self): return f"Info({self.world!r})"


class Atom:
    """A propositional atom formula – holds a Prop object."""
    def __init__(self, prop: Prop):
        self.prop = prop
    def __repr__(self): return f"Atom({self.prop!r})"


class Ng:
    """Negation of a formula."""
    def __init__(self, f):
        self.f = f
    def __repr__(self): return f"Ng({self.f!r})"


class Conj:
    """Conjunction of a list of formulas."""
    def __init__(self, fs):
        self.fs = list(fs)
    def __repr__(self): return f"Conj({self.fs!r})"


class Disj:
    """Disjunction of a list of formulas."""
    def __init__(self, fs):
        self.fs = list(fs)
    def __repr__(self): return f"Disj({self.fs!r})"


class Kn:
    """Knowledge: agent `agent` knows formula `f`."""
    def __init__(self, agent: Agent, f):
        self.agent = agent
        self.f = f
    def __repr__(self): return f"Kn({self.agent!r}, {self.f!r})"


def impl(f1, f2):
    """Material implication f1 => f2, as Disj([Ng(f1), f2])."""
    return Disj([Ng(f1), f2])


# ---------------------------------------------------------------------------
# Epistemic model
# ---------------------------------------------------------------------------

class EpistM:
    """
    An S5 epistemic model.

    Parameters
    ----------
    states  : list of worlds (any hashable type, e.g. int or tuple)
    agents  : list of Agent
    val     : list of (world, [Prop]) pairs  – or a dict world -> [Prop]
    rels    : list of (Agent, partition) pairs – or a dict Agent -> partition
              A partition is a list of blocks, each block a list of worlds.
    actual  : list of actual (pointed) worlds
    """

    def __init__(self, states, agents, val, rels, actual):
        self.states = list(states)
        self.agents = list(agents)
        self.val    = dict(val)  if not isinstance(val,  dict) else val
        self.rels   = dict(rels) if not isinstance(rels, dict) else rels
        self.actual = list(actual)

    def __repr__(self):
        lines = [
            "EpistM(",
            f"  states = {self.states},",
            f"  agents = {self.agents},",
            f"  val    = {self.val},",
            f"  rels   = {self.rels},",
            f"  actual = {self.actual}",
            ")",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Semantic evaluation
# ---------------------------------------------------------------------------

def is_true_at(model: EpistM, world, formula) -> bool:
    """Return True iff `formula` holds at `world` in `model`."""
    if isinstance(formula, Top):
        return True

    if isinstance(formula, Info):
        return world == formula.world

    if isinstance(formula, Atom):
        return formula.prop in model.val.get(world, [])

    if isinstance(formula, Ng):
        return not is_true_at(model, world, formula.f)

    if isinstance(formula, Conj):
        return all(is_true_at(model, world, f) for f in formula.fs)

    if isinstance(formula, Disj):
        return any(is_true_at(model, world, f) for f in formula.fs)

    if isinstance(formula, Kn):
        partition = model.rels[formula.agent]
        block = bl(partition, world)
        return all(is_true_at(model, w, formula.f) for w in block)

    raise TypeError(f"Unknown formula type: {type(formula)}")


def is_true(model: EpistM, formula) -> bool:
    """Return True iff `formula` holds at ALL actual worlds of `model`."""
    return all(is_true_at(model, w, formula) for w in model.actual)


# ---------------------------------------------------------------------------
# Public announcement update
# ---------------------------------------------------------------------------

def upd_pa(model: EpistM, formula) -> EpistM:
    """
    Public announcement of `formula`:
    restrict the model to worlds where `formula` is true,
    and update the accessibility relations accordingly.
    """
    new_states = [s for s in model.states if is_true_at(model, s, formula)]
    new_val    = {s: ps for s, ps in model.val.items() if s in new_states}
    new_rels   = {ag: restrict(new_states, r) for ag, r in model.rels.items()}
    new_actual = [s for s in model.actual if s in new_states]
    return EpistM(new_states, model.agents, new_val, new_rels, new_actual)


def upds_pa(model: EpistM, formulas) -> EpistM:
    """Apply a sequence of public announcement updates one by one."""
    return reduce(upd_pa, formulas, model)
