"""
muddy.py  -  The Muddy Children puzzle using DEMO_LIGHT.
Translated from Muddy.hs by Jan van Eijck.

Four children (a, b, c, d) play outside.  Some get mud on their foreheads.
Each child can see the others' foreheads but not their own.

Setup: b, c, d are muddy; a is clean.

The puzzle unfolds through public announcements:
  1. Each child privately learns whether the others are muddy
     (via 'info' actions - each child sees the others).
  2. Father announces: "At least one of you is muddy."
  3. Father asks: "Does anyone know whether they are muddy?"
     - Round 1: Nobody knows.
     - Round 2: Nobody knows.
     - Round 3: b, c, d all know (they know they are muddy); a knows too.
"""

from demo_light import (
    a, b, c, d,
    P, Prp, Neg, Conj, Disj, K,
    initM, upd, upds, public, info, test, is_true, display_s5
)

# Propositions: P(i) means "child i is muddy"
# We use P(1)=Alice, P(2)=Bob, P(3)=Carol, P(4)=Dave
ma = Prp(P(1))   # Alice is muddy
mb = Prp(P(2))   # Bob is muddy
mc = Prp(P(3))   # Carol is muddy
md = Prp(P(4))   # Dave is muddy

# The actual situation: b, c, d are muddy; a is clean
bcd_dirty = Conj([Neg(ma), mb, mc, md])

# Each child privately learns about the others' muddiness.
# info([b,c,d], ma): b, c, d can distinguish whether a is muddy;
#   a cannot (a doesn't see their own forehead in this scenario).
awareness = [
    info([b, c, d], ma),   # b, c, d learn about a
    info([a, c, d], mb),   # a, c, d learn about b
    info([a, b, d], mc),   # a, b, d learn about c
    info([a, b, c], md),   # a, b, c learn about d
]

# "Does child X know whether they are muddy?"
aKn = Disj([K(a, ma), K(a, Neg(ma))])
bKn = Disj([K(b, mb), K(b, Neg(mb))])
cKn = Disj([K(c, mc), K(c, Neg(mc))])
dKn = Disj([K(d, md), K(d, Neg(md))])

# Step 0: Start from blissful ignorance, then silently set the actual situation.
#   test(bcd_dirty) restricts to worlds where bcd_dirty holds,
#   but nobody observes this restriction (it's just the setup).
mu0 = upd(initM([a, b, c, d], [P(1), P(2), P(3), P(4)]), test(bcd_dirty))

# Step 1: Apply the awareness actions - each child learns about the others.
mu1 = upds(mu0, awareness)

# Step 2: Father announces "At least one of you is muddy."
mu2 = upd(mu1, public(Disj([ma, mb, mc, md])))

# Step 3: Father asks "Does anyone know?" - Round 1: Nobody does.
mu3 = upd(mu2, public(Conj([Neg(aKn), Neg(bKn), Neg(cKn), Neg(dKn)])))

# Step 4: Father asks again - Round 2: Still nobody does.
mu4 = upd(mu3, public(Conj([Neg(aKn), Neg(bKn), Neg(cKn), Neg(dKn)])))

# Step 5: Father asks again - Round 3: b, c, d all know.
mu5 = upds(mu4, [public(Conj([bKn, cKn, dKn]))])


if __name__ == "__main__":
    print("=== Muddy Children ===\n")
    print(f"mu0 - after setup ({len(mu0.states)} states), actual: {mu0.actual}")
    print(f"mu1 - after awareness ({len(mu1.states)} states), actual: {mu1.actual}")
    print(f"mu2 - after father's announcement ({len(mu2.states)} states), actual: {mu2.actual}")
    print(f"mu3 - after round 1 nobody knows ({len(mu3.states)} states), actual: {mu3.actual}")
    print(f"mu4 - after round 2 nobody knows ({len(mu4.states)} states), actual: {mu4.actual}")
    print(f"mu5 - after round 3 b,c,d know ({len(mu5.states)} states), actual: {mu5.actual}")

    print("\nVerification at mu5:")
    print(f"  b knows whether b is muddy: {is_true(mu5, bKn)}")
    print(f"  c knows whether c is muddy: {is_true(mu5, cKn)}")
    print(f"  d knows whether d is muddy: {is_true(mu5, dKn)}")
    print(f"  a knows whether a is muddy: {is_true(mu5, aKn)}")
