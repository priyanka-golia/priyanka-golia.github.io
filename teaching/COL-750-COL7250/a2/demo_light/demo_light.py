"""
demo_light.py  -  DEMO_LIGHT: Dynamic Epistemic Model Operations (Light).
Translated from DemoLight.hs by Jan van Eijck.

This module re-exports everything from the four sub-modules so that
example files need only import from demo_light.
"""

from models_vocab import (
    # Atoms
    P, Q, R,
    # Agents
    Agent, a, alice, b, bob, c, carol, d, dave, e, ernie,
    # Formulas
    Top, Prp, Neg, Conj, Disj, K, CK,
    impl, equiv, negation,
    # Model
    EpistM,
    # Model operations
    rel, right_s, is_true_at, is_true, upd_pa,
    initM, bisim, gsm, convert, display_s5, show_s5,
    rel2partition,
)

from change_vocab import (
    ACM,
    upc, upd, upds,
    public, p_change, group_m, message, test, info,
)

from change_perception import (
    unobserved_change,
    perception,
    perceived_change,
    ppc, npc,
)
