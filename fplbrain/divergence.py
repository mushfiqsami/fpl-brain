"""
Past against present: where the two disagree about a player.

Every projection here fuses two sources - the archive prior built from previous
seasons (seed.py) and this season's own record from the live feed - into a
single blended rate. The blend is right, and it is also where information goes
to hide: once the two numbers are averaged, nothing downstream can tell a
player both sources agree about from one they flatly contradict.

That is not hypothetical. Isak was projected for 61 minutes against a fixture
he was fully fit for, having started every game of the season, because a
disrupted campaign at his previous club still carried half the weight. The
fused number, 0.73, looked unremarkable. The two numbers behind it - 1.00 from
this season, 0.52 from the archive - are obviously in conflict the moment you
see them side by side, and nothing in the app ever showed them side by side.

Measured across the league, 35% of players with evidence on both sides disagree
by 0.30 or more on start rate alone. So this is a large, routine blind spot
rather than a rare edge case.

Two directions, and they fail in opposite ways:

  rising   this season says he starts, the archive says he did not. The
           projection is dragged DOWN by a role he has already outgrown - a
           new signing, a promoted youngster, someone who won a place in
           pre-season. These are under-rated.
  fading   the archive says he starts, this season says he does not. The
           projection is propped UP by a place he has already lost. These are
           over-rated, and they are the more expensive mistake, because the
           model will happily keep recommending a man who is now a substitute.

This module is deliberately NOT wired into the projection. It changes no
number; it only reports. The blend's handover is already correct in the general
case - this season's weight rises with games played and the archive fades on
its own - and overriding it on a few games of evidence would reintroduce
exactly the small-sample whipsaw the shrinkage exists to prevent. What a human
can do that the formula cannot is tell WHY the two disagree: a transfer, a
manager change, a recovered injury. So the disagreement is surfaced for a
person to judge, and the maths is left alone.
"""
from __future__ import annotations

from .model import archive_start_rate, observed_start_rate, prior_blend_weight

# How far apart the two sources must be before it is worth a person's attention.
# 0.30 of a start rate is roughly "one source thinks he plays a third more often
# than the other" - below that the gap is ordinary blend, not a contradiction.
NOTABLE_START = 0.30

# This season needs enough of a sample to be worth comparing against. Two full
# matches of minutes is the floor for the start-rate comparison; per-90 rates
# are far noisier and get a higher bar of their own.
MIN_CURRENT_MINUTES = 180
MIN_RATE_MINUTES = 270


def _rate90(element, key, minutes):
    if minutes <= 0:
        return 0.0
    try:
        return float(element.get(key) or 0.0) / minutes * 90.0
    except (TypeError, ValueError):
        return 0.0


def compare(elements, player_prior, games_by_team, short_names=None,
            notable=NOTABLE_START, limit=12):
    """
    Where the archive and this season disagree about who plays.

    elements      : live bootstrap elements
    player_prior  : seed.load()[0], keyed by FPL `code`
    games_by_team : TeamStrength.games - how many games each club has played
    short_names   : {team_id: "ARS"}, for display only

    Returns {rising, fading, checked, notable, threshold}. `rising` and `fading`
    are sorted by how big the disagreement is, largest first.
    """
    short_names = short_names or {}
    rows = []
    checked = 0
    for e in elements:
        minutes = int(e.get("minutes") or 0)
        if minutes < MIN_CURRENT_MINUTES:
            continue
        played = games_by_team.get(e["team"], 0)
        if played < 2:
            continue
        prior = player_prior.get(str(e.get("code")))
        if not prior or not prior.get("minutes"):
            continue

        checked += 1
        past = archive_start_rate(prior)
        now = observed_start_rate(int(e.get("starts") or 0), played)
        gap = now - past
        if abs(gap) < notable:
            continue

        # Rates are compared only when this season has enough minutes to mean
        # something; below that the columns are left empty rather than filled
        # with a number nobody should act on.
        if minutes >= MIN_RATE_MINUTES:
            past_rate = float(prior.get("xg90", 0.0)) + float(prior.get("xa90", 0.0))
            now_rate = (_rate90(e, "expected_goals", minutes)
                        + _rate90(e, "expected_assists", minutes))
        else:
            past_rate = now_rate = None

        rows.append(dict(
            id=e["id"], name=e.get("web_name", ""),
            club=short_names.get(e["team"], ""), pos=e.get("element_type"),
            price=round(int(e.get("now_cost") or 0) / 10.0, 1),
            past=round(past, 3), now=round(now, 3), gap=round(gap, 3),
            # How much of the blended rate this season is currently winning. A
            # big gap matters far more when the archive still holds most of the
            # weight, because that is when the stale half is actually steering.
            weight_now=round(prior_blend_weight(played), 3),
            past_rate=(None if past_rate is None else round(past_rate, 3)),
            now_rate=(None if now_rate is None else round(now_rate, 3)),
            minutes=minutes, starts=int(e.get("starts") or 0),
            status=e.get("status"),
        ))

    rising = sorted((r for r in rows if r["gap"] > 0), key=lambda r: -r["gap"])
    fading = sorted((r for r in rows if r["gap"] < 0), key=lambda r: r["gap"])
    return dict(rising=rising[:limit], fading=fading[:limit],
                checked=checked, notable=len(rows), threshold=notable)
