"""
change_perception.py  -  Perception-change action model constructors.
Translated from ChangePerception.hs by Jan van Eijck.

These constructors model situations where an agent observes (or fails to
observe) a factual change, and where the group of observers is explicit.
"""

from models_vocab import (
    EpistM, Agent, Top, Neg, Prp, P,
    a, b, c, d, e
)
from change_vocab import ACM, upd, upc, negation


# ---------------------------------------------------------------------------
# Unobserved change
# ---------------------------------------------------------------------------

def unobserved_change(prp, form):
    """
    A factual change (prp := form) that nobody observes.
    
    Two events: 0 (the change happens) and 1 (nothing happens, Top).
    All agents connect 0 and 1, so nobody can distinguish the two events.
    Actual event: 0 (the change does happen).
    """
    def facm(agents):
        return ACM(
            states = [0, 1],
            agents = agents,
            pre    = {0: Top(), 1: Top()},
            subst  = {0: [(prp, form)], 1: []},
            rels   = [(ag, s, t) for ag in agents for s in [0, 1] for t in [0, 1]],
            actual = [0]
        )
    return facm


# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------

def perception(agent, form, group):
    """
    Agent agent perceives whether form is true.
    The group (which must contain agent) learns which event occurred.
    Outsiders (agents not in group) cannot distinguish any of the three events.

    Three events:
      0: form is true     (agent and group observe this)
      1: form is false    (agent and group observe this)
      2: Top - a 'dummy' event outsiders connect to both 0 and 1

    Accessibility:
      - All agents are reflexive on each event.
      - agent (the observer) distinguishes 0 and 1 but connects to 2.
        Wait - re-reading the Haskell: agent does NOT get (a,0,1) or (a,1,0),
        so agent distinguishes 0 from 1.  But agent DOES get (a,0,2) and
        (a,2,0) etc. from the outsiders block only if agent not in group.
        Since agent is in group by assumption, agent has no 0<->1 and no 0<->2.
      - Outsiders (not in group) get: 0<->1, 0<->2, 1<->2 (fully connected).
    
    Actual event: 0 (form is indeed true in the real world).
    """
    def facm(agents):
        outsiders = [ag for ag in agents if ag not in group]
        rels = ([(ag, s, s) for ag in agents for s in [0, 1, 2]] +
                [(ag, 0, 1) for ag in agents if ag not in [agent]] +
                [(ag, 1, 0) for ag in agents if ag not in [agent]] +
                [(ag, 0, 2) for ag in outsiders] +
                [(ag, 2, 0) for ag in outsiders] +
                [(ag, 1, 2) for ag in outsiders] +
                [(ag, 2, 1) for ag in outsiders])
        return ACM(
            states = [0, 1, 2],
            agents = agents,
            pre    = {0: form, 1: Neg(form), 2: Top()},
            subst  = {0: [], 1: [], 2: []},
            rels   = rels,
            actual = [0]
        )
    return facm


# ---------------------------------------------------------------------------
# Perceived change
# ---------------------------------------------------------------------------

def perceived_change(agent, prp, form, group):
    """
    Agent agent (and the group) observe a factual change (prp := form),
    and they also observe the truth value of prp before the change.

    Three events:
      0: prp is currently True  AND prp becomes form
      1: prp is currently False AND prp becomes form
      2: Top - dummy event for outsiders

    Outsiders cannot distinguish any of the three events.
    agent distinguishes 0 from 1 (knows the old value of prp).
    Non-agent members of group also know old value.
    """
    def facm(agents):
        outsiders = [ag for ag in agents if ag not in group]
        rels = ([(ag, s, s) for ag in agents for s in [0, 1, 2]] +
                [(ag, 0, 1) for ag in agents if ag not in [agent]] +
                [(ag, 1, 0) for ag in agents if ag not in [agent]] +
                [(ag, 0, 2) for ag in outsiders] +
                [(ag, 2, 0) for ag in outsiders] +
                [(ag, 1, 2) for ag in outsiders] +
                [(ag, 2, 1) for ag in outsiders])
        return ACM(
            states = [0, 1, 2],
            agents = agents,
            pre    = {0: Prp(prp), 1: Neg(Prp(prp)), 2: Top()},
            subst  = {0: [(prp, form)], 1: [(prp, form)], 2: []},
            rels   = rels,
            actual = [0]
        )
    return facm


# ---------------------------------------------------------------------------
# Positive / negative perceived change  (ppc / npc)
# ---------------------------------------------------------------------------

def ppc(agent, prp, form, group):
    """
    Positive perceived change: pre of event 0 is form, event 1 is Neg(form).
    Substitution: prp := form in both events 0 and 1.
    Actual event: 0 (form is true).
    """
    def facm(agents):
        outsiders = [ag for ag in agents if ag not in group]
        rels = ([(ag, s, s) for ag in agents for s in [0, 1, 2]] +
                [(ag, 0, 1) for ag in agents if ag not in [agent]] +
                [(ag, 1, 0) for ag in agents if ag not in [agent]] +
                [(ag, 0, 2) for ag in outsiders] +
                [(ag, 2, 0) for ag in outsiders] +
                [(ag, 1, 2) for ag in outsiders] +
                [(ag, 2, 1) for ag in outsiders])
        return ACM(
            states = [0, 1, 2],
            agents = agents,
            pre    = {0: form, 1: Neg(form), 2: Top()},
            subst  = {0: [(prp, form)], 1: [(prp, form)], 2: []},
            rels   = rels,
            actual = [0]
        )
    return facm


def npc(agent, prp, form, group):
    """
    Negative perceived change: same as ppc but with negation(form) for event 1.
    Actual event: 1 (form is false / negation holds).
    """
    def facm(agents):
        outsiders = [ag for ag in agents if ag not in group]
        rels = ([(ag, s, s) for ag in agents for s in [0, 1, 2]] +
                [(ag, 0, 1) for ag in agents if ag not in [agent]] +
                [(ag, 1, 0) for ag in agents if ag not in [agent]] +
                [(ag, 0, 2) for ag in outsiders] +
                [(ag, 2, 0) for ag in outsiders] +
                [(ag, 1, 2) for ag in outsiders] +
                [(ag, 2, 1) for ag in outsiders])
        return ACM(
            states = [0, 1, 2],
            agents = agents,
            pre    = {0: form, 1: negation(form), 2: Top()},
            subst  = {0: [(prp, form)], 1: [(prp, form)], 2: []},
            rels   = rels,
            actual = [1]    # actual event is 1 (the negation case)
        )
    return facm
