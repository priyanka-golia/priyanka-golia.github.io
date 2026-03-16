"""
kripkevis.py  –  Visualise Kripke / epistemic models as PDF files.
Translated from KRIPKEVIS.hs by Malvin Gattinger (adapted for DEMO_S5).

Requires the Graphviz `dot` command to be installed.
"""

import os
import subprocess


class VisModel:
    """
    A model prepared for visualisation.

    states  : list of worlds (any type)
    rels    : list of (agent, partition) pairs  – partition = list of blocks
    val     : dict mapping each world to an arbitrary display value
    actual  : list of actual (pointed) worlds
    """
    def __init__(self, states, rels, val, actual):
        self.states = list(states)
        self.rels   = list(rels)          # kept as list of pairs, like Haskell
        self.val    = dict(val)
        self.actual = list(actual)


# ---------------------------------------------------------------------------
# DOT string construction  (corresponds to stringModel in Haskell)
# ---------------------------------------------------------------------------

def _node_id(show_state, s):
    """
    Build a safe Graphviz node identifier.
    We prepend 'w' (matching the Haskell) and strip characters that would
    break an unquoted DOT identifier.
    """
    raw = show_state(s)
    safe = (raw.replace(" ", "_").replace("(", "").replace(")", "")
               .replace(",", "_").replace("'", "").replace('"', ""))
    return "w" + safe


def _build_dot(show_state, show_agent, show_val, information, vis_model):
    """
    Return a Graphviz DOT string for the given VisModel.

    Edge deduplication: Haskell uses `x < y` under Ord.  In Python we use
    the index of each state in the states list to define the same ordering.
    """
    states = vis_model.states
    rels   = vis_model.rels
    val    = vis_model.val
    actual = vis_model.actual

    # Map each state to its position so we can impose x < y ordering
    idx = {s: i for i, s in enumerate(states)}

    # ---- node lines -------------------------------------------------------
    node_lines = []
    for s in states:
        label = show_state(s) + "\\n" + show_val(val.get(s))
        nid   = _node_id(show_state, s)
        # actual worlds get a double circle, others a plain circle
        shape = "doublecircle" if s in actual else "circle"
        node_lines.append(f'  node [shape={shape}, label="{label}"] {nid};\n')

    # ---- edge lines -------------------------------------------------------
    # For each agent and each block in that agent's partition, emit one
    # undirected edge per unordered pair {x, y}; nub via a seen set.
    seen   = set()
    edge_lines = []
    for (ag, partition) in rels:
        ag_label = show_agent(ag)
        for block in partition:
            for x in block:
                for y in block:
                    if idx[x] < idx[y]:               # x < y  (same as Haskell)
                        key = (ag_label, idx[x], idx[y])
                        if key not in seen:
                            seen.add(key)
                            nx = _node_id(show_state, x)
                            ny = _node_id(show_state, y)
                            edge_lines.append(
                                f'  {nx} -- {ny} [ label = "{ag_label}" ];\n'
                            )

    # ---- assemble ---------------------------------------------------------
    dot  = "graph G {\n"
    dot += '  edge [lblstyle=auto];\n'
    dot += "".join(node_lines)
    dot += '  rankdir=LR;\n  size="8,10";\n'
    dot += "".join(edge_lines)
    if information:
        dot += f'  label = "{information}";\n'
    dot += "  fontsize=12;\n"
    dot += "}\n"
    return dot


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dot_model(show_state, show_agent, show_val, information, vis_model, filename):
    """Write the DOT representation to `filename`. Returns a status string."""
    gstring = _build_dot(show_state, show_agent, show_val, information, vis_model)
    with open(filename, "w") as f:
        f.write(gstring)
    return f"The model was DOT'd to {filename}"


def pdf_model(show_state, show_agent, show_val, information, vis_model, filename):
    """
    Generate a PDF of the Kripke model.

    Steps (matching the Haskell original):
      1. Create tmp/ if it doesn't exist.
      2. Write tmp/<filename>.dot
      3. Run  dot -Tpdf  to produce tmp/<filename>.pdf
      4. Delete the temporary .dot file.

    Returns a status string.
    """
    os.makedirs("tmp", exist_ok=True)
    dot_path = os.path.join("tmp", filename + ".dot")
    pdf_path = os.path.join("tmp", filename + ".pdf")

    msg = dot_model(show_state, show_agent, show_val, information, vis_model, dot_path)
    print(msg)

    result = subprocess.run(
        ["dot", "-Tpdf", dot_path, "-o", pdf_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"`dot` command failed:\n{result.stderr}")

    os.remove(dot_path)
    return f"Model saved to {pdf_path}"
