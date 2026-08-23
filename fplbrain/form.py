"""
Note-taking: what actually happened, against what was expected.

Everything else in this project looks forward. This looks at the gameweek being
played or just finished and says how it went - not "who should I buy", but "who
did what I thought they would, and who did not".

That distinction matters because it is the only honest thing to do with a
part-played gameweek. Between a deadline and the results being confirmed the
model must not rebuild plans on one match's worth of noise (see season_phase in
dashboard.py), but the matches themselves are still information, and throwing
them away is its own kind of blindness. So they are recorded rather than acted
on.

It is deliberately NOT a projection. Nothing here feeds back into expected
points; calibrate.py owns that loop and does it slowly and with a minimum sample,
precisely so one loud week cannot move the model. This module only reports.

Four questions, in the order they are worth asking:

  shining      beat what the model expected, by enough to be worth noticing
  struggling   fell short by the same margin
  comeback     unavailable or doubtful now, but due back inside the horizon
  peaking      fixtures turning good NOW rather than later in the window

`history.py` answers a different question and the two are easy to confuse: it
diffs successive runs to catch the model changing its mind about a player. This
compares the model against reality. A player can be rising in history.py while
struggling here - that is the model catching up to a bad week, not a
contradiction.
"""
from __future__ import annotations

# How far from the projection is worth calling out. Expected points for a single
# gameweek are a mean over a very wide distribution - a 5.0 EP forward returning
# 2 or 9 is entirely ordinary - so a small gap is not a signal, it is the shape
# of the thing. Three points is roughly where a return (a goal, an assist, a
# clean sheet plus bonus) separates from noise.
NOTABLE = 3.0

# A haul is worth flagging even when it was half expected of a premium.
HAUL = 10


def _band(delta, actual, played):
    if not played:
        return "unused"
    if delta >= NOTABLE or actual >= HAUL:
        return "shining"
    if delta <= -NOTABLE:
        return "struggling"
    return "as expected"


def note_players(rows):
    """
    rows: [{id, name, club, pos, price, owned, projected, actual, minutes, started}]

    Returns the same rows with `delta` and `band`, sorted by how far reality
    landed from the projection - biggest overperformance first, biggest
    disappointment last, which is the order the list is read in.

    `played` is taken from minutes rather than from points, because zero points
    and zero involvement are different facts. A striker who played ninety and
    returned nothing is a genuine miss; one who never came on tells you about
    selection, not form, and lands in `unused` so it cannot be mistaken for the
    former.
    """
    out = []
    for r in rows:
        played = int(r.get("minutes") or 0) > 0
        proj = float(r.get("projected") or 0.0)
        act = float(r.get("actual") or 0.0)
        delta = act - proj
        out.append(dict(r, projected=round(proj, 2), actual=round(act, 1),
                        delta=round(delta, 2), played=played,
                        band=_band(delta, act, played)))
    out.sort(key=lambda r: -r["delta"])
    return out


def comebacks(elements, availability, horizon_len=5):
    """
    Players not available now who the model expects back inside the window.

    `availability` is PlayerModel.availability, passed in rather than imported so
    this module keeps no opinion about how availability is worked out. A player
    counts if he is short of full fitness today but materially better by the end
    of the horizon - that gap IS the comeback, and it is exactly what the
    FLAG_RECOVERY curve encodes.
    """
    out = []
    for e in elements:
        now = availability(e, 0)
        later = availability(e, horizon_len - 1)
        if now >= 0.99 or later - now < 0.2:
            continue
        out.append(dict(
            id=e["id"], name=e.get("web_name", ""), status=e.get("status", "a"),
            chance=e.get("chance_of_playing_next_round"),
            news=(e.get("news") or "").strip()[:120],
            now=round(now, 2), later=round(later, 2),
            gain=round(later - now, 2)))
    out.sort(key=lambda r: -r["gain"])
    return out


def peaking(per_gw, near=2):
    """
    Is this player's good run NOW, or later in the window?

    per_gw: {player_id: [ep_gw1, ep_gw2, ...]} over the horizon.

    Returns {player_id: swing}, where swing is how much better the next `near`
    gameweeks are than the rest of the window, per gameweek. Positive means buy
    him now; negative means his fixtures improve after the window and there is no
    hurry. Ranking on total points over the horizon cannot see this at all - two
    players with the same five-week total can want opposite timing.
    """
    out = {}
    for pid, series in per_gw.items():
        if len(series) < near + 1:
            continue
        head = sum(series[:near]) / near
        tail = sum(series[near:]) / (len(series) - near)
        out[pid] = round(head - tail, 3)
    return out
