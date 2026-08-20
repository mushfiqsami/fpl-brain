"""
The second unit.

Everything else in this project is a calculator: it takes the data, does the
arithmetic honestly, and reports the best answer it can find. That is necessary
and it is not sufficient. A calculator has no opinion about whether the season is
going well, no memory of where the target is, and no way of noticing that it has
spent six gameweeks confidently recommending "hold" while the gap to the target
quietly widened. It answers the question it was asked. It does not ask whether
that was the right question.

This module holds the goal and judges everything against it.

Three things it does that the calculator cannot:

  1. Pace. Points banked, gameweeks left, and the rate now required. The target
     is a rate, and that rate MOVES: every week below it raises the bar for
     every week that follows. A season is lost gradually and then suddenly, and
     the only early warning is watching the required rate climb.

  2. Posture. How much variance to buy is not a constant - it is a function of
     the gap. A manager ahead of the rate should take the steady option; one
     behind it cannot afford to, because the safe choice locks in a deficit.
     `captain_value` has a fixed appetite for ceiling; this says what that
     appetite ought to be this week.

  3. An honest verdict. Including, when it is true, that the target is no longer
     reachable and pretending otherwise costs more than admitting it. A tool
     that only ever encourages is not giving advice.

Deliberately NOT a second model. It does no football projection of its own and
never overrides the maths - it reads the same numbers the dashboard shows and
decides what they mean for the goal. If the two ever disagree, the calculator is
right about football and this is right about strategy.
"""
from __future__ import annotations

SEASON_GWS = 38

# --- what this system actually delivers -------------------------------------
# MEASURED, by season_test.py, walk-forward on 2025/26: build the best legal
# £100m squad from what was knowable before each gameweek, then score it on what
# those players really got, blanks included. Sixteen gameweeks.
#
#   actual   61.8 per gameweek
#   expected 67.5 per gameweek   (the model's own projection for the same XI)
#
# The 5.7 shortfall is NOT treated as a correction. Its 95% interval is -18.0 to
# +6.6, so it cannot be distinguished from zero and baking it in would be
# inventing precision. What IS reliable is the actual figure, and that is what a
# season target has to be judged against - a projection is what the model hopes
# for, this is what the hoping was worth.
#
# The spread is the important half. 21.2 points of gameweek-to-gameweek standard
# deviation is enormous next to a 9-point gap to target, which is why a season is
# far less predictable than a weekly projection makes it look.
MEASURED_PER_GW = 61.8
MEASURED_SD_GW = 21.2
MEASURED_NOTE = "season_test.py, 16 walk-forward gameweeks of 2025/26"

# --- what a target is worth, in context -------------------------------------
# The 2025/26 competition was won with 2538. Reported by the owner, not derived
# here, and worth more than any of the arithmetic above: it is the only figure
# that says what a target actually MEANS. Ten million entrants, a whole season,
# and the best of them finished on 2538 - so 2700 is not an ambitious target, it
# is roughly 160 points beyond what won the game.
#
# This belongs in the model rather than in someone's head because a target with
# no benchmark quietly becomes a stick to measure a good season against and find
# it wanting. 66.8 a gameweek won last season. Sustaining that is the goal.
WINNER_LAST_SEASON = 2538
BENCHMARKS = (
    ("won the game", 2538),
    ("top 1k", 2400),
    ("top 10k", 2300),
    ("top 100k", 2200),
    ("average manager", 2050),
)


def benchmark(total):
    """What a season total is worth, against what people actually score."""
    for label, mark in BENCHMARKS:
        if total >= mark:
            return label, mark
    return "below average", BENCHMARKS[-1][1]


def _phi(z):
    """Standard normal CDF, via erf - no scipy in this project."""
    import math
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def outcome(target, points_so_far, gws_left, per_gw=None, sd_gw=MEASURED_SD_GW,
            chip_points=0.0):
    """
    The season as a distribution rather than a single number.

    Gameweek scores vary enormously and independently enough that the total's
    spread grows with the square root of the weeks left. Quoting one expected
    total hides that a season is a range hundreds of points wide, and it is the
    range - not the midpoint - that decides whether an ambitious target is
    merely unlikely or effectively impossible.
    """
    import math
    per_gw = MEASURED_PER_GW if per_gw is None else per_gw
    mean = points_so_far + per_gw * gws_left + chip_points
    sd = sd_gw * math.sqrt(max(0, gws_left))
    p_hit = 1.0 - _phi((target - mean) / sd) if sd > 0 else float(mean >= target)
    return dict(
        expected=round(mean), sd=round(sd),
        p10=round(mean - 1.2816 * sd), p50=round(mean),
        p90=round(mean + 1.2816 * sd),
        p_target=round(p_hit, 4),
        stretch=round(mean + 1.2816 * sd),   # a good season, not a miracle
    )

# How far behind the required rate you have to be before the answer changes.
#
# Under about a point a gameweek is noise: this model's own backtest puts its
# edge over a naive baseline at roughly 0.6 points per picked player over five
# gameweeks, so reacting to less than that is reacting to nothing. Beyond about
# three a gameweek the safe option is no longer safe, because holding a steady
# squad guarantees the deficit rather than risking it.
NOISE_PER_GW = 1.0
CHASE_PER_GW = 3.0

# Appetite for ceiling over mean, by posture. The middle value is what
# sim.captain_value uses by default; the others are what it should use when the
# situation is not neutral.
CEILING_APPETITE = {"protect": 0.20, "hold": 0.35, "press": 0.50, "chase": 0.70}


def pace(target, points_so_far, gws_played, projection):
    """
    Where the season actually stands.

    `projection` is what the model expects per gameweek from here. Everything
    else is history. The number that matters is `needed`, and its most important
    property is that it rises every time a gameweek comes in under it.
    """
    gws_left = max(0, SEASON_GWS - gws_played)
    remaining = target - points_so_far
    needed = (remaining / gws_left) if gws_left else 0.0
    achieved = (points_so_far / gws_played) if gws_played else 0.0
    return dict(
        target=target, points_so_far=points_so_far,
        gws_played=gws_played, gws_left=gws_left,
        needed=round(needed, 2),
        projection=round(projection, 2),
        gap=round(needed - projection, 2),
        achieved=round(achieved, 2),
        # where this rate lands if nothing changes
        pace_total=round(points_so_far + projection * gws_left),
        shortfall=round(target - (points_so_far + projection * gws_left)),
    )


def posture(gap, gws_left):
    """
    How much risk this week is worth, and why.

    The gap alone is not enough - the same deficit means different things in
    GW3 and GW33, because there is a different number of gameweeks left to
    recover it in. Late in a season a small gap is nearly fatal and a large one
    entirely so, which is exactly when a steady squad stops being the safe
    choice.
    """
    if gws_left <= 0:
        return "hold", "The season is over.", CEILING_APPETITE["hold"]

    urgency = gap * (10.0 / max(4, gws_left))     # same gap bites harder late on

    if gap <= -NOISE_PER_GW:
        return ("protect",
                f"You are {abs(gap):.1f} a gameweek ahead of the rate. The steady option "
                f"is the right one - variance can only cost you from here.",
                CEILING_APPETITE["protect"])
    if gap <= NOISE_PER_GW:
        return ("hold",
                f"You are within {abs(gap):.1f} a gameweek of the rate, which is inside "
                f"this model's own margin. Nothing about the situation argues for acting.",
                CEILING_APPETITE["hold"])
    if gap <= CHASE_PER_GW and urgency < CHASE_PER_GW:
        return ("press",
                f"You are {gap:.1f} a gameweek behind with {gws_left} to play. Not yet "
                f"serious, but stop taking the safe side of close calls - lean to the "
                f"higher ceiling where two options are level.",
                CEILING_APPETITE["press"])
    return ("chase",
            f"You are {gap:.1f} a gameweek behind with {gws_left} left. Holding a steady "
            f"squad now guarantees the deficit instead of risking it. Buy variance: "
            f"differentials, the fattest captain ceiling, chips on the earliest good week "
            f"rather than the perfect one.",
            CEILING_APPETITE["chase"])


def reachable(p, best_possible):
    """
    Is the target still on, honestly?

    `best_possible` is the strongest squad the optimiser can build right now.
    If the perfect team cannot reach the required rate, no advice about
    transfers closes the gap, and saying so is more useful than another
    suggestion. A tool that only ever encourages is not giving advice.
    """
    if p["gws_left"] <= 0:
        return dict(verdict="over", note="The season has finished.")
    headroom = round(best_possible - p["needed"], 2)
    if headroom >= 0:
        return dict(
            verdict="on", headroom=headroom,
            note=(f"The required {p['needed']:.1f} a gameweek is inside what a perfect "
                  f"£100m squad projects ({best_possible:.1f}), so the target is still "
                  f"reachable - by {headroom:.1f} a gameweek of room."))
    # The best team in the game cannot do it. Say so, and say what can.
    par = round(p["points_so_far"] + best_possible * p["gws_left"])
    # A projection is a mean, and before any football is played it is a mean the
    # model has never once been checked against. Treating that as a verdict
    # would be false precision - the honest reading is that the target needs the
    # model to be beaten, not that it is arithmetically impossible.
    untested = p["gws_played"] == 0
    note = (f"The required {p['needed']:.1f} a gameweek is above what the best possible "
            f"£100m squad projects ({best_possible:.1f}). No transfer closes that gap.")
    if untested:
        note += (" But nothing has been played yet, so that projection has never been "
                 "tested against a real gameweek - it rests entirely on last season, and "
                 "expected points are a mean rather than a ceiling. Read it as: the target "
                 "requires beating the model, not that it is unreachable. Calibration "
                 "starts correcting this from GW1.")
    else:
        note += (f" A finish around {par} is what the current projection supports. Chips "
                 f"and variance can beat a projection; they are not a plan.")
    return dict(verdict=("untested" if untested else "off"), headroom=headroom,
                realistic=par, note=note)


def levers(p, plan_gain=0.0, chip_gain=0.0, best_gain=0.0):
    """
    Where the missing points could actually come from, largest first.

    The point of ranking them is that they are not interchangeable. Transfers
    are available every week and each is worth little; chips are worth a lot and
    can be played once. A gap that transfers cannot close may still be closable,
    but only by something you get one shot at - and knowing which it is changes
    what you do this week.
    """
    need_total = round(p["gap"] * p["gws_left"], 1) if p["gap"] > 0 else 0.0
    out = []
    if best_gain > 0:
        out.append(dict(name="Rebuild toward the optimal squad",
                        gain=round(best_gain * p["gws_left"], 1), per_gw=round(best_gain, 2),
                        note="The full gap between your squad and the best available, "
                             "sustained. Rarely reachable in one move."))
    if plan_gain > 0:
        out.append(dict(name="Take this week's recommended transfers",
                        gain=round(plan_gain, 1), per_gw=round(plan_gain / max(1, 5), 2),
                        note="Already costed over the planning horizon."))
    if chip_gain > 0:
        out.append(dict(name="Chips, played on their best week",
                        gain=round(chip_gain, 1), per_gw=round(chip_gain / max(1, p["gws_left"]), 2),
                        note="One shot each. Worth far more on a double gameweek."))
    out.sort(key=lambda r: -r["gain"])
    covered = sum(r["gain"] for r in out)
    return dict(need=need_total, found=round(covered, 1),
                uncovered=round(max(0.0, need_total - covered), 1), items=out)


def assess(target, points_so_far, gws_played, projection, best_possible,
           plan_gain=0.0, chip_gain=0.0, floor=None, ceiling=None,
           measured_per_gw=None):
    """
    The whole judgement, in one object the UI can render.

    `projection` is what the model expects this week. It is deliberately NOT
    what the season forecast is built from - the model's expectation has been
    measured against reality and came in about six points a gameweek high, on a
    sample too noisy to correct for but far too suggestive to extrapolate from.
    The forecast uses measured delivery instead, and the two are shown side by
    side so the difference is visible rather than quietly resolved.
    """
    p = pace(target, points_so_far, gws_played, projection)
    per_gw = MEASURED_PER_GW if measured_per_gw is None else measured_per_gw
    # scale measured delivery by how this squad compares with the one that was
    # measured, so a genuinely better or worse team is not given the same forecast
    if projection > 0:
        per_gw = per_gw * (projection / 67.5)
    out = outcome(target, points_so_far, p["gws_left"], per_gw=per_gw,
                  chip_points=chip_gain)
    post, why, appetite = posture(p["gap"], p["gws_left"])
    reach = reachable(p, best_possible)
    lev = levers(p, plan_gain=plan_gain, chip_gain=chip_gain,
                 best_gain=max(0.0, best_possible - projection))

    # One instruction. If the target is gone, chasing it makes things worse, so
    # the posture is overridden rather than left to argue with the verdict.
    if reach["verdict"] == "untested":
        headline = "Target needs the model beaten, and the model is untested"
    elif reach["verdict"] == "off":
        headline = "The target is out of reach - play for the best finish available"
        post = "chase" if p["gap"] > CHASE_PER_GW else post
    elif post == "chase":
        headline = "Behind the rate - buy variance"
    elif post == "press":
        headline = "Slightly behind - take the higher ceiling on close calls"
    elif post == "protect":
        headline = "Ahead of the rate - protect it"
    else:
        headline = "On the rate - no change needed"

    # The honest headline is the probability, not a verdict. "Out of reach" and
    # "on track" are both wrong when the answer is a distribution: a season sits
    # hundreds of points wide, so an ambitious target is usually neither certain
    # nor impossible, just unlikely by a stateable amount.
    pct = out["p_target"] * 100

    # Chasing a target you cannot reach is worse than not chasing it.
    #
    # Posture is driven by the gap, so an ambitious target forces maximum
    # variance by construction. That logic only holds while the target is
    # actually reachable. Extra variance does raise the chance of clearing a
    # distant mark - but it lowers the median, and buying a rise from 0.5% to
    # perhaps 1% at the cost of roughly a point a gameweek is a bad trade unless
    # nothing except that number counts. Wanting to finish as high as possible
    # and wanting a 0.5% shot are different objectives with different answers,
    # and the tool should not silently pick the second.
    #
    # So below a threshold of plausibility the objective switches to the best
    # finish available, and says so rather than quietly chasing.
    CHASE_WORTH_IT = 0.05
    chasing_mirage = out["p_target"] < CHASE_WORTH_IT and post == "chase"
    if chasing_mirage:
        post, appetite = "press", CEILING_APPETITE["press"]
        why = (f"The gap says chase, but {target} is a {pct:.1f}% outcome - buying maximum "
               f"variance for that costs about a point a gameweek off the median and "
               f"barely moves the odds. Objective switched to the best finish available: "
               f"lean to the higher ceiling on close calls, without wrecking the expectation.")

    # Judge the target against what actually wins before judging the squad
    # against the target. A number nobody has ever reached is not a standard to
    # fall short of; it is the wrong number.
    beats_winner = target > WINNER_LAST_SEASON
    exp_label, _exp_mark = benchmark(out["expected"])
    stretch_label, _sm = benchmark(out["stretch"])
    context = dict(
        winner=WINNER_LAST_SEASON,
        target_vs_winner=target - WINNER_LAST_SEASON,
        expected_rank=exp_label, stretch_rank=stretch_label,
        note=(f"Last season was won with {WINNER_LAST_SEASON}. "
              + (f"{target} is {target - WINNER_LAST_SEASON} beyond that, so it is not a "
                 f"stretch target - it is better than anyone managed. "
                 if beats_winner else
                 f"{target} would have finished inside the top few thousand. ")
              + f"On current form this squad projects {out['expected']} ({exp_label}), "
                f"with a good season reaching about {out['stretch']} ({stretch_label})."))
    if beats_winner:
        headline = (f"{target} is past what won last season ({WINNER_LAST_SEASON}). "
                    f"Aim at {out['stretch']} - that is {stretch_label}")
    elif pct >= 45:
        headline = f"{target} is live - about a {pct:.0f}% chance on measured form"
    elif pct >= 15:
        headline = f"{target} is a stretch - about {pct:.0f}%, and it needs the variance"
    elif pct >= 2:
        headline = f"{target} is unlikely - about {pct:.0f}%. Play for the best finish"
    else:
        headline = (f"{target} is out of realistic reach ({pct:.1f}%). "
                    f"Aim at {out['stretch']}")

    return dict(pace=p, posture=post, posture_note=why, ceiling_appetite=appetite,
                reachable=reach, levers=lev, headline=headline,
                context=context, chasing_mirage=chasing_mirage,
                outcome=out, measured_per_gw=round(per_gw, 2),
                measured_note=MEASURED_NOTE,
                floor=floor, ceiling=ceiling)
