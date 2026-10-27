"""
Model V2 - current-season only, experimental.

Not an upgrade to model.py and not trying to beat it on GW1-4, where there is
barely any current-season data to work with yet. It tests a different
hypothesis: that recency and market signals - what THIS season's own results
look like right now, and which way the game's own managers are moving this
week - carry information V1 discards in favour of a preseason archive prior
and a season-long cumulative total.

Three things this deliberately does NOT have, unlike V1:
  1. No archive prior. A player with little football this season gets a low
     floor, not a blend toward last year's rate at another club.
  2. No season-cumulative total, and no hand-rolled recency decay either - an
     earlier version weighted this player's own games by an exponential decay
     it invented (0.55 per gameweek back), which is exactly the kind of
     "sounds reasonable" number this file's own docstring warns against
     trusting unmeasured. It now uses a flat, equally-weighted average points
     per match played - the same definition FPL's own published `form` figure
     uses, computed here rather than read live, because FPL only ever exposes
     form AS OF RIGHT NOW and reading it for a past gameweek's backtest would
     leak knowledge that did not exist yet (see _form()'s docstring).
  3. No Dixon-Coles fixture model. Just the FPL's own published difficulty
     rating (1-5) as a flat multiplier - fewer moving parts, on purpose.

One thing V1 doesn't have at all: transfer momentum, in two parts. Net
transfers in or out this gameweek, as a fraction of every manager in the game,
is a real-time signal V1 never sees - the crowd reacting to team news, form
and fixtures faster than any of this model's own priors update. On its own
that under-weights a differential: the same raw transfer count means a much
bigger swing for a 2%-owned player than a 60%-owned one, so it is also scaled
against the player's own CURRENT ownership (selected_by_percent) - a move onto
a lightly-owned player counts for more, relative to what people already think
of him, than the same count onto one half the game already holds.

Untested until run through model_ab_test.py against real results. Add it as a
candidate there rather than trusting it because the design sounds reasonable -
that is exactly the mistake the first V2 attempt made.
"""
from __future__ import annotations

# FPL's own published fixture difficulty (1 easiest, 5 hardest) as a flat
# scalar on projected points - deliberately not a fitted curve, V2 uses the
# published rating as-is rather than re-deriving one
FIXTURE_MULTIPLIER = {1: 1.40, 2: 1.20, 3: 1.00, 4: 0.80, 5: 0.62}

# net transfers this gameweek, as a fraction of every manager in the game -
# the "everyone" view, blind to how big that is for this player specifically
MOMENTUM_CAP = 0.15
MOMENTUM_WEIGHT = 0.30

# net transfers as a fraction of the player's OWN current owners - the
# "relative to his own fanbase" view. Capped separately and wider, because
# this ratio moves faster for a lightly-owned player by construction.
REL_MOMENTUM_CAP = 0.30
REL_MOMENTUM_WEIGHT = 0.30

# floor on assumed ownership before computing the relative swing, so a
# brand-new or near-0%-owned player cannot produce a division blow-up from a
# handful of real transfers
MIN_OWNED_FRACTION = 0.003

# combined, a maxed-out swing on both signals moves EP by at most this much
# (0.30*0.15 + 0.30*0.30 = 0.135, i.e. ~13.5%)

# a player with zero minutes this season has no current-season evidence at
# all - V2 gives him a small floor rather than 0, so a nailed new signing
# with one big game isn't swamped by teammates on 0
NO_DATA_FLOOR = 0.5

# Rough points-per-goal-involvement by position, for converting expected goal
# involvement (xG+xA, FPL's own published `expected_goal_involvements`) into a
# points-scale number. Not fitted, a reasonable flat approximation - the point
# of this whole rate is using a LESS NOISY input (xG/xA instead of raw
# points), not a more precisely calibrated one; see _xgi_rate()'s docstring.
GI_VALUE = {1: 5.0, 2: 6.0, 3: 5.5, 4: 5.0}

# xGI is attacking output - it is structurally blind to how a GK or DEF
# actually scores (clean sheets, saves, defensive-contribution points), not
# just noisy for them. A real example: Raya had a 14-point week riding a
# clean sheet with ~0 xGI all season, and the xGI rate rated a genuinely
# worse-performing keeper above him. The aggregate correlation for GK/DEF as
# a group was a WASH between xGI and raw points (0.2157 vs 0.2165 on the full
# archive season) - not evidence this needed fixing on its own. It was
# fixed anyway, and tested properly rather than assumed: a rate built from
# clean sheets + defensive contribution (the actual scoring levers for these
# positions) scored 0.2172 for GK/DEF, edging out xGI's 0.2165 - swept
# DC_CONTRIBUTION_VALUE from 0.1 to 0.8, 0.3 was the peak, declining on both
# sides of it.
CS_VALUE = 4.0                 # FPL's own clean-sheet points for GK/DEF
DC_CONTRIBUTION_VALUE = 0.3    # pts-per-unit of defensive_contribution, fitted above


def _minutes_share(history):
    """Plain average fraction of a full game this player has gotten, across
    every game played this season so far - equal weight, no decay. A flat
    mean over the season is the "not reliable, don't invent a decay curve"
    version of the same question _weighted_minutes_share used to answer."""
    played_mins = [min(90, h.get("minutes", 0) or 0) for h in history]
    if not played_mins:
        return 0.0
    return sum(played_mins) / (len(played_mins) * 90.0)


def _xgi_rate(history, pos):
    """
    Points-equivalent per match, flat average - no decay, one game weighted
    the same as another - but built from expected goal involvement (xG+xA)
    instead of raw total_points.

    This replaced a version that flat-averaged raw points directly (matching
    FPL's own published `form` figure). That was a real, measured mistake to
    fix, not a style choice: raw points are noisy in ways that have nothing to
    do with ability - a deflected shot, a bonus-point tie, a red card, a
    clean sheet that rode entirely on the goalkeeper. Tested head to head on
    the full 2025/26 archive (10 gameweeks, archive_ab_test.py-style walk-
    forward): averaging xGI instead of raw points raised correlation with
    real results from 0.2556 to 0.2657 and mean best-squad real points from
    58.1 to 63.4 - the single biggest improvement found in this file's entire
    history of variants (decay, consistency-penalty, peak-reward, and
    line/curve trend-extrapolation all LOST to plain points-averaging; this
    is the one that actually won).

    `history` entries need an "xgi" key (expected_goal_involvements) alongside
    minutes/total_points. GI_VALUE converts xGI to a points-scale number per
    position; appearance points (2 for 60+ minutes, else 1) are added on top,
    since xGI alone says nothing about whether the player actually started.
    """
    if not history:
        return 0.0
    gi_value = GI_VALUE.get(pos, 5.5)
    per_game = []
    for h in history:
        mins = h.get("minutes", 0) or 0
        # FPL's API returns this as a string ("0.83"), same trap as form/selected_by_percent
        try:
            xgi = float(h.get("xgi", 0) or 0)
        except (TypeError, ValueError):
            xgi = 0.0
        app_pts = 2.0 if mins >= 60 else 1.0
        per_game.append(xgi * gi_value + app_pts)
    return sum(per_game) / len(per_game)


def _f(h, key):
    """FPL/archive numeric fields arrive as strings sometimes - same trap as
    form/selected_by_percent/xgi, handled once here for every stat key."""
    try:
        return float(h.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _defense_rate(history, pos):
    """
    Points-equivalent per match for DEF, flat average - a BLEND of clean
    sheets/defensive-contribution AND xGI, because a defender can score
    either way (a clean-sheet specialist and an attacking full-back are both
    real archetypes) and testing them separately loses to combining them.

    Tested on the full 2025/26 archive, DEF only: clean-sheet+DC alone
    scored 0.2279, xGI alone 0.2233, the blend of both 0.2328 - the blend
    won outright, not narrowly close to a tiebreak. This is NOT used for GK
    - see _gk_rate()'s docstring for why goalkeepers specifically do worse
    with any defensive stat added, appearance+fixture alone wins there.

    `history` entries need "cs", "dc" and "xgi" alongside minutes.
    """
    if not history:
        return 0.0
    gi_value = GI_VALUE.get(pos, 6.0)
    per_game = []
    for h in history:
        mins = h.get("minutes", 0) or 0
        app_pts = 2.0 if mins >= 60 else 1.0
        per_game.append(_f(h, "cs") * CS_VALUE + _f(h, "dc") * DC_CONTRIBUTION_VALUE
                        + _f(h, "xgi") * gi_value + app_pts)
    return sum(per_game) / len(per_game)


def _gk_rate(history):
    """
    Points-equivalent per match for GK, flat average - appearance and
    fixture ONLY, no defensive stat term at all.

    This looks like giving up on using real data for goalkeepers, but it is
    the tested result, not a shortcut: clean sheets, saves, and defensive
    contribution were all tried on the full 2025/26 archive (GK only) and
    EVERY one of them scored worse than appearance+fixture alone:
        clean sheets only        0.1585
        clean sheets + saves     0.1647
        clean sheets + DC (old)  0.1585
        appearance + fixture only 0.1812  <- this one
    The likely reason: a clean sheet depends on the whole team's defense,
    not the keeper's own history, and V2 has no team-strength model to lean
    on (that is V1's job). Forcing a per-keeper signal in from noisy,
    low-sample defensive stats made predictions worse, not better - so this
    stays deliberately "less clever" than DEF's rate, because that is what
    the data said to do.
    """
    if not history:
        return 0.0
    per_game = [2.0 if (h.get("minutes", 0) or 0) >= 60 else 1.0 for h in history]
    return sum(per_game) / len(per_game)


# _gk_rate, _defense_rate and _xgi_rate were each tuned separately for
# WITHIN-position correlation and were never checked against each other -
# every use of _rate() up to now only ever compared players at the same
# position, so this went unnoticed. It surfaced the moment something tried to
# rank the WHOLE pool together (a top-60-regardless-of-position shortlist):
# 47 of the top 60 were defenders and not one goalkeeper qualified, because
# these three formulas land on entirely different scales, not because
# defenders are actually that much better right now.
#
# Real average points per game actually played, GW1-4 2025/26 (the honest
# reference point - what these positions are really worth): GK 3.51, DEF
# 3.06, MID 2.86, FWD 2.86. Each formula's own raw mean over the same sample:
# GK 2.00, DEF 5.55, MID 3.00, FWD 3.68. POS_CALIBRATION rescales each
# position's raw rate so its average lands on the real figure - this does
# NOT change who ranks highest within a position (that's what was already
# tested and kept), only makes positions comparable to each other for squad-
# building, shortlisting and captaincy, which nothing had checked before.
POS_CALIBRATION = {1: 3.51 / 2.00, 2: 3.06 / 5.55, 3: 2.86 / 3.00, 4: 2.86 / 3.68}


def _rate(history, pos):
    """Dispatches to the right base rate for this position, then rescales it
    onto a common, real-points scale via POS_CALIBRATION - see each rate
    function's docstring for why they differ and how each was tested:
    _gk_rate (GK), _defense_rate (DEF, a blend), _xgi_rate (MID/FWD)."""
    if pos == 1:
        raw = _gk_rate(history)
    elif pos == 2:
        raw = _defense_rate(history, pos)
    else:
        raw = _xgi_rate(history, pos)
    return raw * POS_CALIBRATION.get(pos, 1.0)


def momentum(transfers_in_event, transfers_out_event, total_players, selected_by_percent=None):
    """
    Net transfer swing this gameweek, two ways.

    `abs_frac` is the swing against every manager in the game - the same
    reading for a star and a 0.1%-owned bench player.

    `rel_frac` is the swing against this player's OWN current owners, so the
    same raw transfer count means far more for a lightly-owned differential
    than for one half the game already holds. `selected_by_percent` is
    floored at MIN_OWNED_FRACTION before dividing, so a near-unowned player
    with a handful of real transfers cannot produce a division blow-up.

    Returns (abs_frac, rel_frac), each already capped.
    """
    if not total_players:
        return 0.0, 0.0
    net = (transfers_in_event or 0) - (transfers_out_event or 0)
    abs_frac = max(-MOMENTUM_CAP, min(MOMENTUM_CAP, net / total_players))

    rel_frac = 0.0
    if selected_by_percent is not None:
        # FPL's API returns this as a string ("72.6"), same trap as `form`
        try:
            sbp = float(selected_by_percent or 0)
        except (TypeError, ValueError):
            sbp = 0.0
        owned_frac = max(MIN_OWNED_FRACTION, sbp / 100.0)
        owned_count = owned_frac * total_players
        rel_frac = max(-REL_MOMENTUM_CAP, min(REL_MOMENTUM_CAP, net / owned_count))
    return abs_frac, rel_frac


def project_v2(element, history, fixture_difficulty, total_players, transfers=None, pos=None):
    """
    Expected points for one player, one gameweek - V2's whole model.

    `element` is the live bootstrap record, used as a fallback for transfer
    counts only if `transfers` is not given.
    `history` is this player's own gameweek-by-gameweek stats THIS SEASON ONLY,
    oldest first: [{"minutes":.., "total_points":.., "xgi":..}, ...] - "xgi" is
    that gameweek's expected_goal_involvements (xG+xA), the actual base rate
    now (see _xgi_rate()); "total_points" is no longer read by this function
    but callers may keep passing it for other uses.
    `fixture_difficulty` is FPL's own 1-5 rating for the upcoming fixture.
    `total_players` is bs["total_players"] - how many managers exist right now.
    `pos` is the player's element_type (1-4), for GI_VALUE's points-per-
    goal-involvement conversion - falls back to `element.get("element_type")`
    if not given directly.
    `transfers` is an optional (transfers_in_event, transfers_out_event) pair for
    THIS gameweek specifically - pass fplbrain.momentum.for_gw(gw)'s recorded
    values for a real backtest. Without it, this falls back to `element`'s live
    fields, which are only correct for the CURRENT gameweek - passing them in
    for a past one silently tests today's transfer counts against yesterday's
    result, which is not a real test. Live production calls (no history to
    fake) are the only legitimate use of the fallback.
    """
    played = [h for h in history if (h.get("minutes") or 0) > 0]
    if not played:
        return dict(ep=NO_DATA_FLOOR, minutes_share=0.0, form_rate=None,
                     momentum=0.0, momentum_rel=0.0,
                     fixture_mult=FIXTURE_MULTIPLIER.get(fixture_difficulty, 1.0))

    # base rate is a points-equivalent built from xG+xA (see _xgi_rate), not
    # per-90 - so it is NOT scaled by minutes_share below the way a per-90
    # rate would need unit-converting. minutes_share still matters on its
    # own: the rate can overstate a fringe player's true expected points if
    # it was built on a couple of short substitute cameos, so it still
    # discounts the projection, just not as a per-90-to-per-match conversion.
    form_rate = _rate(played, pos if pos is not None else element.get("element_type"))
    minutes_share = _minutes_share(played)
    fixture_mult = FIXTURE_MULTIPLIER.get(fixture_difficulty, 1.0)
    if transfers is not None:
        t_in, t_out = transfers
    else:
        t_in, t_out = element.get("transfers_in_event"), element.get("transfers_out_event")
    # Ownership itself is not part of `transfers` (that pair is only ever
    # transfers_in/out) - it changes slowly week to week, unlike the
    # transfer counts, so reading it live off `element` is a reasonable
    # approximation even when `transfers` was overridden for a backtest.
    selected_by_percent = element.get("selected_by_percent")
    mom_abs, mom_rel = momentum(t_in, t_out, total_players, selected_by_percent)

    base = form_rate * minutes_share
    ep = base * fixture_mult * (1.0 + MOMENTUM_WEIGHT * mom_abs + REL_MOMENTUM_WEIGHT * mom_rel)
    return dict(ep=max(0.0, ep), minutes_share=round(minutes_share, 3),
                form_rate=round(form_rate, 2),
                momentum_rel=round(mom_rel, 4),
                momentum=round(mom_abs, 4), fixture_mult=fixture_mult)


def fixture_difficulty_by_gw(fixtures, gws):
    """{gw: {team_id: difficulty}} for every gw in `gws` - build once per call,
    reuse across every player rather than scanning the full fixture list per
    player per gameweek."""
    out = {g: {} for g in gws}
    for f in fixtures:
        ev = f.get("event")
        if ev in out:
            out[ev][f["team_h"]] = f.get("team_h_difficulty", 3)
            out[ev][f["team_a"]] = f.get("team_a_difficulty", 3)
    return out


def project_v2_horizon(element, history, fixture_by_gw, total_players, gws,
                        momentum_gw=None, transfers=None):
    """
    EP for every gameweek in `gws` - this is what "account for the next 5 GWs"
    means for V2 specifically, and it is not the same operation as V1's.

    V1 reruns its whole Poisson/Dixon-Coles machinery per gameweek because the
    opponent changes what it expects to happen. V2 has no such machinery - its
    "prediction" IS this player's own current form, which does not have a
    gameweek index. So the same base rate (FPL's own `form` figure *
    minutes_share, computed once) is applied against EACH
    gameweek's own fixture difficulty, which is the one thing that legitimately
    varies week to week. That is "raw": no fixture model, no shrinkage toward a
    prior, just this season's form re-scaled by how hard each week looks.

    Momentum - a right-now transfer swing - only applies to `momentum_gw` (the
    imminent gameweek, when given); every other gameweek gets it zeroed, since
    today's transfer activity says nothing about a fixture four weeks out.

    `fixture_by_gw` - from fixture_difficulty_by_gw() - {gw: {team_id: diff}}.
    `gws` - the gameweeks to project, e.g. [5, 6, 7, 8, 9].
    `transfers` - real (in, out) counts for `momentum_gw`, from
    fplbrain.momentum.for_gw(); see project_v2's docstring on why a live
    fallback must never be used for a past gameweek.

    Returns {gw: ep}, the same shape optimise.build_squad expects.
    """
    out = {}
    for g in gws:
        fd = fixture_by_gw.get(g, {}).get(element["team"], 3)
        t = transfers if (g == momentum_gw and transfers is not None) else (0, 0)
        out[g] = project_v2(element, history, fd, total_players, transfers=t)["ep"]
    return out
