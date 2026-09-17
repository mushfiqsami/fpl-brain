"""
The three votes: past, present, future - and the rule that a player must win
two of them, one of which has to be the present.

Every projection in this project is a single blended number, and a single
number cannot be argued with. It also cannot be corroborated: when the blend
says 5.54 there is no way to ask whether the past, the present and the outlook
actually agree, or whether one of them is carrying the other two. Thiago is the
case that forced this. The model projected him at 5.54 a game while he was
returning 1.2 - a strong archive record and a favourable outlook outvoting the
only source that had watched him play this season.

So the three are separated and each gets a vote:

  V1  past      the archive prior, built from the previous three seasons.
                Did he deliver, over a sample large enough to mean something?
  V2  present   this season's own record. Is he delivering NOW?
  V3  future    the model's projection over the horizon. Is he set up to
                deliver next?

A player needs TWO yeses, and V2 must be one of them. Written out, the whole
rule is: `v2 and (v1 or v3)`.

Why the present is mandatory rather than merely one vote of three: the failures
worth preventing were all players the present had already condemned. A pedigree
that is not currently showing up, and an outlook built partly on that same
pedigree, are not two independent pieces of evidence - they are one, counted
twice. Requiring the present breaks that.

Why two rather than one: a single view is weak on its own, and measurably so.
Ranking this season's players on points scored so far predicts what they do
next at a correlation of 0.151 - the fifteen hottest players after GW2 went on
to score 7.7 in GW3-4, exactly what everybody else scored. Form alone is close
to no information. Demanding that a second, differently-derived view agrees is
what turns it into evidence.

The bar for each vote is the median for that position among players who
actually play, so "yes" means "better than the typical starter in his role"
rather than a number typed in here that drifts as the season moves.

This gates selection. It does not touch any projection: a player who fails
still has exactly the expected points he had before, he is simply not offered.
"""
from __future__ import annotations
import statistics

# This season needs enough of a sample before the present can sensibly vote.
# Three full matches. Below it V2 abstains, which under the rule means the
# player does not qualify - deliberately, because "we cannot yet tell whether
# he is delivering" is not the same as "he is".
MIN_MINUTES_NOW = 270

# If almost nobody clears that bar the season is too young for the rule to mean
# anything, and applying it would empty the pool rather than filter it.
MIN_POOL = 40


def _median(values, fallback=0.0):
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else fallback


def _bars(elements, prior, ep_per_gw, reference):
    """Median of each vote's metric, per position, among players who play."""
    past, now, fut = {}, {}, {}
    for pos in (1, 2, 3, 4):
        members = [e for e in reference if e.get("element_type") == pos]
        past[pos] = _median([(prior.get(str(e.get("code"))) or {}).get("ppg")
                             for e in members
                             if prior.get(str(e.get("code")))])
        now[pos] = _median([float(e.get("points_per_game") or 0) for e in members])
        fut[pos] = _median([ep_per_gw.get(e["id"]) for e in members
                            if e["id"] in ep_per_gw])
    return past, now, fut


def assess(elements, player_prior, ep_per_gw, min_minutes=MIN_MINUTES_NOW):
    """
    Score every player against the three votes.

    ep_per_gw : {player_id: projected points per gameweek over the horizon}

    Returns (rows_by_id, meta). Each row carries the three booleans, the value
    and bar behind each, and `ok` - the rule's verdict.
    """
    prior = player_prior or {}
    reference = [e for e in elements if int(e.get("minutes") or 0) >= min_minutes]
    meta = dict(reference=len(reference), min_minutes=min_minutes,
                usable=len(reference) >= MIN_POOL)
    if not meta["usable"]:
        # Too early in the season to judge the present. Say so rather than
        # returning a verdict that would exclude everybody.
        return {}, meta

    bar_past, bar_now, bar_fut = _bars(elements, prior, ep_per_gw, reference)
    rows = {}
    for e in elements:
        pos = e.get("element_type")
        if pos not in (1, 2, 3, 4):
            continue
        minutes = int(e.get("minutes") or 0)
        p = prior.get(str(e.get("code"))) or {}

        past_val = p.get("ppg")
        v1 = past_val is not None and past_val >= bar_past[pos]

        now_val = float(e.get("points_per_game") or 0)
        # Not enough minutes is an abstention, not a pass: the present has to
        # have actually seen him to vote for him.
        v2 = minutes >= min_minutes and now_val >= bar_now[pos]

        fut_val = ep_per_gw.get(e["id"])
        v3 = fut_val is not None and fut_val >= bar_fut[pos]

        rows[e["id"]] = dict(
            id=e["id"], name=e.get("web_name", ""), pos=pos,
            price=round(int(e.get("now_cost") or 0) / 10.0, 1),
            v1=v1, v2=v2, v3=v3,
            past=(None if past_val is None else round(past_val, 2)),
            now=round(now_val, 2),
            future=(None if fut_val is None else round(fut_val, 2)),
            bar_past=round(bar_past[pos], 2), bar_now=round(bar_now[pos], 2),
            bar_future=round(bar_fut[pos], 2),
            minutes=minutes,
            ok=bool(v2 and (v1 or v3)),
        )
    meta["qualified"] = sum(1 for r in rows.values() if r["ok"])
    meta["bars"] = dict(past=bar_past, now=bar_now, future=bar_fut)
    return rows, meta


def reason(row):
    """One line saying why a player passed or failed, for display."""
    if row is None:
        return ""
    if row["ok"]:
        won = [n for n, v in (("past", row["v1"]), ("present", row["v2"]),
                              ("future", row["v3"])) if v]
        return "passes on " + " + ".join(won)
    if not row["v2"]:
        if row["minutes"] < MIN_MINUTES_NOW:
            return "not enough minutes this season to judge"
        return "not delivering now (%.1f vs %.1f needed)" % (row["now"], row["bar_now"])
    return "only the present says yes - needs the past or the outlook too"
