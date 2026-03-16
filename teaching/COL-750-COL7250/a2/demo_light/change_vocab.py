"""
change_vocab.py  -  Action models with vocabulary change (substitutions).
Translated from ChangeVocab.hs by Jan van Eijck.

An ACM (Action model with Change of vocabulary) extends AM with a
substitution: each event maps propositions to new formulas, allowing
factual change in the world.

The high-level constructors public, groupM, message, test, info are the
main interface for building common epistemic actions.
"""

from models_vocab import (
    EpistM, Agent, bisim, is_true_at, convert,
    a, b, c, d, e, Top, Neg, Prp, P, Q, R,
    negation, _sort_l
)


# ---------------------------------------------------------------------------
# ACM datatype
# ---------------------------------------------------------------------------

class ACM:
    """
    Action model with change of vocabulary.

    states  : list of action states (events)
    agents  : list of Agent
    pre     : dict { event -> Form }   preconditions
    subst   : dict { event -> [(PrpAtom, Form)] }  substitutions
    rels    : list of (Agent, event, event) triples
    actual  : list of actual events
    """
    def __init__(self, states, agents, pre, subst, rels, actual):
        self.states = list(states)
        self.agents = list(agents)
        self.pre    = dict(pre)   if not isinstance(pre,   dict) else pre
        self.subst  = dict(subst) if not isinstance(subst, dict) else subst
        self.rels   = list(rels)
        self.actual = list(actual)

    def __repr__(self):
        return (f"ACM(states={self.states}, pre={self.pre}, "
                f"subst={self.subst}, rels={self.rels}, actual={self.actual})")


# ---------------------------------------------------------------------------
# ACM helpers
# ---------------------------------------------------------------------------

def _sub(acm: ACM, s, p):
    """
    Apply the substitution for event s to proposition p.
    If p has no substitution in s, return Prp(p) (identity).
    """
    sb = acm.subst.get(s, [])
    for (prop, form) in sb:
        if prop == p:
            return form
    return Prp(p)


def _new_val(m: EpistM, acm: ACM, w, s):
    """
    Compute the new valuation at world (w, s) after applying the substitution.
    allprops = props currently true at w  UNION  props mentioned in subst[s]
    Keep p iff its substitution formula is true at w in m.
    """
    current_props = m.val.get(w, [])
    subst_props   = [p for (p, _) in acm.subst.get(s, [])]
    all_props     = list(dict.fromkeys(current_props + subst_props))  # ordered, deduped
    return [p for p in all_props if is_true_at(m, w, _sub(acm, s, p)) == True]


# ---------------------------------------------------------------------------
# Product update with vocabulary change (raw - returns EpistM of pairs)
# ---------------------------------------------------------------------------

def upc(m: EpistM, facm) -> EpistM:
    """
    Product update of epistemic model m with ACM facm (unapplied).
    Applies substitutions to compute the new valuation.
    Returns EpistM whose states are (world, event) pairs.
    """
    acm = facm(m.agents)

    # New worlds: pairs (w, s) where precondition of s is true at w
    new_states = [(w, s) for w in m.states for s in acm.states
                  if is_true_at(m, w, acm.pre[s]) == True]

    # New vocabulary: union of old vocab and props in substitution preconditions
    subst_voc = sorted(list(set(
        p for s in acm.states for (p, _) in acm.subst.get(s, [])
    )))
    new_voc = sorted(list(set(m.voc + subst_voc)), key=lambda x: x._key())

    # New valuation: apply substitutions
    new_val = {(w, s): _new_val(m, acm, w, s) for (w, s) in new_states}

    # Relations: synchronise epistemic and action relations
    new_rels = [(ag1, (w1, s1), (w2, s2))
                for (ag1, w1, w2) in m.rels
                for (ag2, s1, s2) in acm.rels
                if ag1 == ag2
                and (w1, s1) in new_states
                and (w2, s2) in new_states]

    # Actual worlds
    new_actual = [(p, a_ev) for p in m.actual for a_ev in acm.actual
                  if (p, a_ev) in new_states]

    return EpistM(new_states, m.agents, new_voc, new_val, new_rels, new_actual)


def upd(m: EpistM, facm) -> EpistM:
    """Product update with vocabulary change, followed by bisimulation."""
    return bisim(upc(m, facm))


def upds(m: EpistM, facm_list) -> EpistM:
    """Apply a sequence of ACM updates."""
    result = m
    for facm in facm_list:
        result = upd(result, facm)
    return result


# ---------------------------------------------------------------------------
# Standard ACM constructors
# ---------------------------------------------------------------------------

def public(form):
    """
    Public announcement of form.
    Single event 0, precondition=form, no substitution.
    All agents treat it the same way.
    """
    def facm(agents):
        return ACM(
            states = [0],
            agents = agents,
            pre    = {0: form},
            subst  = {0: []},
            rels   = [(ag, 0, 0) for ag in agents],
            actual = [0]
        )
    return facm


def p_change(substitution):
    """
    Public factual change: Top precondition, apply substitution to all agents.
    """
    def facm(agents):
        return ACM(
            states = [0],
            agents = agents,
            pre    = {0: Top()},
            subst  = {0: substitution},
            rels   = [(ag, 0, 0) for ag in agents],
            actual = [0]
        )
    return facm


def group_m(group, form):
    """
    Group announcement: `group` learns `form`; outsiders cannot distinguish
    whether form or Top was the precondition.

    If group == all agents: same as public(form).
    Otherwise: two events 0 (pre=form) and 1 (pre=Top); outsiders connect them.
    """
    def facm(agents):
        if sorted(group, key=lambda ag: ag.n) == sorted(agents, key=lambda ag: ag.n):
            return public(form)(agents)
        outsiders = [ag for ag in agents if ag not in group]
        return ACM(
            states = [0, 1],
            agents = agents,
            pre    = {0: form, 1: Top()},
            subst  = {0: [], 1: []},
            rels   = ([(ag, 0, 0) for ag in agents] +
                      [(ag, 0, 1) for ag in outsiders] +
                      [(ag, 1, 0) for ag in outsiders] +
                      [(ag, 1, 1) for ag in agents]),
            actual = [0]
        )
    return facm


def message(agent, form):
    """Private message to a single agent: only `agent` learns `form`."""
    return group_m([agent], form)


def test(form):
    """
    Silent test: restrict to worlds where form is true, but no agent
    observes the test (all agents connect the two events).
    Equivalent to group_m([], form).
    """
    return group_m([], form)


def info(group, form):
    """
    Everyone learns whether form is true or false.
    The `group` can distinguish which event occurred; outsiders cannot.

    Event 0: form is true.
    Event 1: negation(form) is true.
    Outsiders connect 0 and 1 in both directions.
    """
    def facm(agents):
        outsiders = [ag for ag in agents if ag not in group]
        return ACM(
            states = [0, 1],
            agents = agents,
            pre    = {0: form, 1: negation(form)},
            subst  = {0: [], 1: []},
            rels   = ([(ag, 0, 0) for ag in agents] +
                      [(ag, 1, 1) for ag in agents] +
                      [(ag, 0, 1) for ag in outsiders] +
                      [(ag, 1, 0) for ag in outsiders]),
            actual = [0, 1]
        )
    return facm
