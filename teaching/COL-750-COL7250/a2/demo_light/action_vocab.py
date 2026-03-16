"""
action_vocab.py  -  Action models (no vocabulary change).
Translated from ActionVocab.hs by Jan van Eijck.

An action model (AM) is like an epistemic model but with preconditions
instead of a valuation.  Product update (up/upd) takes an epistemic model
and an action model and returns the updated epistemic model.
"""

from models_vocab import (
    EpistM, Agent, bisim, is_true_at, convert,
    a, b, c, d, e, Top, Prp, P, Q, R
)


# ---------------------------------------------------------------------------
# Action Model
# ---------------------------------------------------------------------------

class AM:
    """
    An action model (without vocabulary change).

    states  : list of action states (events)
    agents  : list of Agent
    pre     : dict { event -> Form }  -  preconditions
    rels    : list of (Agent, event, event) triples
    actual  : list of actual events
    """
    def __init__(self, states, agents, pre, rels, actual):
        self.states = list(states)
        self.agents = list(agents)
        self.pre    = dict(pre) if not isinstance(pre, dict) else pre
        self.rels   = list(rels)
        self.actual = list(actual)

    def __repr__(self):
        return (f"AM(states={self.states}, pre={self.pre}, "
                f"rels={self.rels}, actual={self.actual})")


# A FAM is a function from a list of agents to an AM.
# In Python we represent this simply as a callable.


# ---------------------------------------------------------------------------
# Product update (raw - returns EpistM of pairs)
# ---------------------------------------------------------------------------

def up(m: EpistM, fam) -> EpistM:
    """
    Product update of epistemic model m with action model fam (unapplied).
    Returns EpistM whose states are (world, event) pairs.
    """
    am = fam(m.agents)

    # New worlds: pairs (w, s) where the precondition of event s is true at w
    new_states = [(w, s) for w in m.states for s in am.states
                  if is_true_at(m, w, am.pre[s]) == True]

    # Valuation: carry over the epistemic model's props (action doesn't change val)
    new_val = {(w, s): props
               for (w, props) in m.val.items()
               for s in am.states
               if (w, s) in new_states}

    # Relations: synchronise epistemic and action relations
    new_rels = [(ag1, (w1, s1), (w2, s2))
                for (ag1, w1, w2) in m.rels
                for (ag2, s1, s2) in am.rels
                if ag1 == ag2
                and (w1, s1) in new_states
                and (w2, s2) in new_states]

    # Actual worlds: pairs of actual epistemic world with actual action event
    new_actual = [(p, a_ev) for p in m.actual for a_ev in am.actual
                  if (p, a_ev) in new_states]

    return EpistM(new_states, m.agents, m.voc, new_val, new_rels, new_actual)


def upd(m: EpistM, fam) -> EpistM:
    """Product update followed by bisimulation minimisation."""
    return bisim(up(m, fam))


# ---------------------------------------------------------------------------
# Standard action model constructors
# ---------------------------------------------------------------------------

def public(form):
    """
    Public announcement of form as an FAM.
    Single event 0 with precondition form; all agents treat it the same.
    """
    def fam(agents):
        return AM(
            states = [0],
            agents = agents,
            pre    = {0: form},
            rels   = [(ag, 0, 0) for ag in agents],
            actual = [0]
        )
    return fam
