"""
cheryl.py  –  Cheryl's Birthday puzzle with PDF visualisation.

Run with:
    python cheryl.py

Four PDFs will be written to the tmp/ directory, one per stage of the puzzle.
Requires the Graphviz `dot` command (graphviz package).
"""

from demo_s5 import EpistM, Agent, a, b, upd_pa, Kn, Info, Disj, Conj, Ng
from erel import fct2erel
from kripkevis import VisModel, pdf_model


# ---------------------------------------------------------------------------
# All candidate dates
# ---------------------------------------------------------------------------

all_dates = [
    (15, "May"), (16, "May"), (19, "May"),
    (17, "June"), (18, "June"),
    (14, "July"), (16, "July"),
    (14, "August"), (15, "August"), (17, "August"),
]


# ---------------------------------------------------------------------------
# Initial model
# Albert knows the month -> partition by month.
# Bernard knows the day  -> partition by day.
# ---------------------------------------------------------------------------

init_cheryl = EpistM(
    states = all_dates,
    agents = [a, b],
    val    = {},
    rels   = {
        a: fct2erel(lambda d: d[1], all_dates),   # group by month
        b: fct2erel(lambda d: d[0], all_dates),   # group by day
    },
    actual = all_dates,
)


# ---------------------------------------------------------------------------
# Helper: "agent knows which date it is"
# ---------------------------------------------------------------------------

def knows_which(agent: Agent):
    return Disj([Kn(agent, Info(d)) for d in all_dates])


# ---------------------------------------------------------------------------
# The three announcements
# ---------------------------------------------------------------------------

ann1 = Conj([Ng(knows_which(a)), Kn(a, Ng(knows_which(b)))])
model2 = upd_pa(init_cheryl, ann1)

ann2 = knows_which(b)
model3 = upd_pa(model2, ann2)

ann3 = knows_which(a)
model4 = upd_pa(model3, ann3)


# ---------------------------------------------------------------------------
# Visualisation helpers
# Mirrors myPdfModel from CHERYL.hs exactly:
#   showState (n, month) = str(n) + month
#   showVal   _          = ""
#   showAg    ag         = "Albert" if ag==a else "Bernard"
# ---------------------------------------------------------------------------

def show_state(d):
    n, month = d
    return str(n) + month

def show_val(_):
    return ""

def show_ag(ag):
    return "Albert" if ag == a else "Bernard"


def my_pdf_model(model: EpistM, filename: str):
    """Build a VisModel from an EpistM and write it as a PDF."""
    vis = VisModel(
        states = model.states,
        rels   = list(model.rels.items()),          # dict -> list of (agent, partition) pairs
        val    = {s: 0 for s in model.states},      # dummy value – showVal ignores it
        actual = model.actual,
    )
    msg = pdf_model(show_state, show_ag, show_val, "", vis, filename)
    print(msg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Cheryl's Birthday ===\n")

    print(f"Initial candidates ({len(init_cheryl.states)} dates):")
    print(" ", init_cheryl.states)

    print(f"\nAfter Albert's statement ({len(model2.states)} dates):")
    print(" ", model2.states)

    print(f"\nAfter Bernard's statement ({len(model3.states)} dates):")
    print(" ", model3.states)

    print(f"\nAfter Albert's final statement ({len(model4.states)} dates):")
    print(" ", model4.states)

    if len(model4.states) == 1:
        day, month = model4.states[0]
        print(f"\nCheryl's birthday is {month} {day}.")

    print("\nGenerating PDFs...")
    my_pdf_model(init_cheryl, "cheryl_1_init")
    my_pdf_model(model2,      "cheryl_2_albert")
    my_pdf_model(model3,      "cheryl_3_bernard")
    my_pdf_model(model4,      "cheryl_4_solution")
    print("Done. PDFs are in the tmp/ directory.")
