"""
erel.py  –  Equivalence relations as partitions (lists of blocks).
Translated from EREL.hs by Jan van Eijck.

An Erel is a list of lists, e.g. [[1,2],[3,4,5]].
Each inner list is a block; together they partition the domain.
"""


# ---------------------------------------------------------------------------
# Block lookup
# ---------------------------------------------------------------------------

def bl(erel, x):
    """Return the block in erel that contains x."""
    for block in erel:
        if x in block:
            return block
    raise ValueError(f"Element {x!r} not found in any block of {erel!r}")


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

def dom_e(erel):
    """All elements that appear in any block, in order of first appearance."""
    seen = []
    for block in erel:
        for x in block:
            if x not in seen:
                seen.append(x)
    return seen


# ---------------------------------------------------------------------------
# Restrict a partition to a sub-domain
# ---------------------------------------------------------------------------

def restrict(domain, erel):
    """
    Keep only elements in `domain` from each block;
    drop blocks that become empty; drop duplicate blocks.
    """
    result = []
    seen = []
    for block in erel:
        filtered = [x for x in block if x in domain]
        if filtered and filtered not in seen:
            result.append(filtered)
            seen.append(filtered)
    return result


# ---------------------------------------------------------------------------
# Fusion: merge overlapping blocks until we have a true partition
# ---------------------------------------------------------------------------

def _overlap(xs, ys):
    """Do the two lists share at least one element?"""
    s = ys  # kept as list so it works for unhashable elements too
    return any(x in s for x in xs)


def _union(a, b):
    """Union of two lists, preserving order of first occurrence."""
    result = list(a)
    for x in b:
        if x not in result:
            result.append(x)
    return result


def fusion(blocks):
    """
    Given a cover (list of lists), fuse any blocks that share an element,
    repeating until no two blocks overlap.  Returns the resulting partition.

    This computes the partition of the reflexive-transitive-symmetric closure
    of the similarity defined by the cover.
    """
    if not blocks:
        return []
    b, *bs = blocks
    cs = [c for c in bs if _overlap(b, c)]          # blocks overlapping b
    xs = _union(b, [x for c in cs for x in c])      # merge them all into xs
    ds = [c for c in bs if _overlap(xs, c)]          # blocks overlapping xs
    if cs == ds:
        # no new overlaps introduced by merging – xs is a finished block
        remaining = [c for c in bs if c not in cs]
        return [xs] + fusion(remaining)
    else:
        # merging created new overlaps; retry with the larger xs
        return fusion([xs] + bs)


# ---------------------------------------------------------------------------
# Build partitions from functions
# ---------------------------------------------------------------------------

def fct2erel(f, domain):
    """
    Partition `domain` by grouping elements with the same f-value.
    e.g. fct2erel(lambda x: x % 3, [1..9])
         -> [[1,4,7],[2,5,8],[3,6,9]]
    """
    result = []
    remaining = list(domain)
    while remaining:
        x = remaining[0]
        block = [y for y in remaining if f(y) == f(x)]
        rest  = [y for y in remaining if f(y) != f(x)]
        result.append(block)
        remaining = rest
    return result


def cfct2erel(domain, f):
    """
    Build a cover from a symmetric characteristic function f(x, y) -> bool.
    Each element x is placed in a block with every y where f(x, y) is True.
    Used internally by the bisimulation refinement algorithm.
    """
    result = []
    remaining = list(domain)
    while remaining:
        x = remaining[0]
        block = [x] + [y for y in remaining[1:] if f(x, y)]
        rest  =       [y for y in remaining[1:] if not f(x, y)]
        result.append(block)
        remaining = rest
    return result


# ---------------------------------------------------------------------------
# Product of two partitions restricted to a domain of pairs
# ---------------------------------------------------------------------------

def restricted_prod(r, s, domain):
    """
    Cartesian product of blocks from r and s, keeping only pairs in `domain`.
    Used when updating with an action model.
    """
    result = []
    for b in r:
        for c in s:
            block = [(x, y) for x in b for y in c if (x, y) in domain]
            if block:
                result.append(block)
    return result
