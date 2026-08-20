"""
Cold-start seeding.

Problem this solves: at the start of a season every player in the FPL API has
minutes = 0, expected_goals = 0, starts = 0. With nothing to shrink toward except
a positional average, the model rates Haaland exactly the same as a bench forward
and gives everyone the same chance of starting. It is not wrong so much as blind.

Fix: carry last season forward as a player-level prior, exactly the way team
strength already carries a prior. As this season's minutes accumulate, the prior
fades on its own — same empirical-Bayes blend, same automatic handover.

Players are matched on FPL's `code` field, which is a permanent per-player ID that
survives transfers, price changes and season rollovers. Matching on names would be
guesswork; matching on code is exact.

Run once:   python run.py seed
"""
from __future__ import annotations
import csv, io, json, os, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED_FILE = os.path.join(HERE, "data", "player_prior.json")
ARCHIVE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
UA = {"User-Agent": "Mozilla/5.0 (fplbrain)"}

MIN_MINUTES = 400          # below this, a season tells us too little to trust

# Seasons to carry forward, newest first, with how much each counts.
#
# Two seasons beat one for the same reason 3,000 minutes beat 400: a per-90 rate
# built on one campaign is a small sample, and one good or unlucky year distorts
# it. Blending the previous two roughly doubles the evidence behind every player
# and pulls the flukes toward the truth.
#
# The older season is discounted rather than counted equally - football changes,
# players move clubs and age, and a rate from two years ago describes someone who
# no longer quite exists. 0.55 says the older campaign is worth a bit over half
# of the recent one per minute played. That is a judgement, not a fitted value.
SEASONS = (("2025-26", 1.00), ("2024-25", 0.55))

# Rates that are averaged across seasons. Everything else is taken from the most
# recent season the player appears in, because it describes his current role
# rather than his historical output.
BLEND_KEYS = ("xg90", "xa90", "dc90", "bps90", "saves90")


def _num(row, key):
    try:
        return float(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0


def _fetch(season):
    """One season's final player table, reduced to per-90 rates."""
    url = f"{ARCHIVE}/{season}/players_raw.csv"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        rows = list(csv.DictReader(io.StringIO(r.read().decode("utf-8-sig"))))
    return _reduce(rows)


def build(season=None, seasons=SEASONS):
    """
    Build the player prior from several seasons and cache it.

    Written once to data/player_prior.json and embedded into the single-file
    build, so nothing is downloaded on a refresh or a redeploy.
    """
    if season and seasons is SEASONS:          # explicit single season still works
        seasons = ((season, 1.0),)

    per_season, meta_seasons, failed = [], [], []
    for name, weight in seasons:
        try:
            got, skipped = _fetch(name)
        except Exception as exc:
            failed.append(f"{name}: {exc}")
            continue
        per_season.append((got, weight))
        meta_seasons.append(dict(season=name, weight=weight,
                                 players=len(got), skipped=skipped))
    if not per_season:
        raise RuntimeError("no seasons could be downloaded: " + "; ".join(failed))

    prior = _blend(per_season)
    meta = dict(season=" + ".join(s["season"] for s in meta_seasons),
                seasons=meta_seasons, players=len(prior),
                min_minutes=MIN_MINUTES, failed=failed)
    os.makedirs(os.path.dirname(SEED_FILE), exist_ok=True)
    with open(SEED_FILE, "w", encoding="utf-8") as f:
        json.dump(dict(meta=meta, prior=prior), f)
    return meta


def _blend(per_season):
    """
    Combine seasons per player, weighting by minutes played and recency.

    Minutes are the weight because they are the sample size: a rate off 3,000
    minutes deserves more say than one off 500. Recency multiplies that, so a
    heavy older season still counts for less than a heavy recent one.

    Two different minute counts come out of this and conflating them inflates
    everything:

      minutes        typical minutes in ONE season, recency-weighted. This is
                     what answers "does he play", and it must stay per-season -
                     summing two campaigns halves the bar for looking nailed and
                     promotes rotation players into guaranteed starters.
      minutes_total  the sum, which is genuinely the evidence behind the rates
                     and is what the shrinkage should use.
    """
    out = {}
    for idx, (season_rows, recency) in enumerate(per_season):
        for code, rec in season_rows.items():
            slot = out.setdefault(code, {"_w": {}, "_tot": 0.0, "_mins": 0.0,
                                         "_seasons": 0, "_first": None})
            w = rec["minutes"] * recency
            for k in BLEND_KEYS:
                slot["_w"][k] = slot["_w"].get(k, 0.0) + rec[k] * w
            slot["_tot"] += w
            slot["_mins"] += rec["minutes"]
            slot["_mw"] = slot.get("_mw", 0.0) + rec["minutes"] * recency
            slot["_rw"] = slot.get("_rw", 0.0) + recency
            slot["_seasons"] += 1
            if slot["_first"] is None:          # newest season wins for role fields
                slot["_first"] = rec

    prior = {}
    for code, slot in out.items():
        base = slot["_first"]
        tot = slot["_tot"] or 1.0
        rec = dict(base)
        for k in BLEND_KEYS:
            rec[k] = slot["_w"].get(k, 0.0) / tot
        # per-season, for "does he play"; total, for "how sure are we"
        rec["minutes"] = slot["_mw"] / (slot["_rw"] or 1.0)
        rec["minutes_total"] = slot["_mins"]
        rec["seasons"] = slot["_seasons"]
        prior[code] = rec
    return prior


def _reduce(rows):
    prior, skipped = {}, 0
    for row in rows:
        code = row.get("code")
        mins = _num(row, "minutes")
        if not code or mins < MIN_MINUTES:
            skipped += 1
            continue
        starts = _num(row, "starts")
        # team games is unknown per player, but starts/appearances is a decent
        # proxy for "does this player start when available"
        appearances = max(1.0, starts + max(0.0, _num(row, "starts") * 0))
        dc = (_num(row, "clearances_blocks_interceptions") + _num(row, "tackles")
              + _num(row, "recoveries"))
        prior[code] = dict(
            name=row.get("web_name", ""),
            minutes=mins,
            starts=starts,
            xg90=_num(row, "expected_goals") / mins * 90,
            xa90=_num(row, "expected_assists") / mins * 90,
            dc90=dc / mins * 90 if dc else 0.0,
            bps90=_num(row, "bps") / mins * 90,
            saves90=_num(row, "saves") / mins * 90,
            # minutes/starts counts substitute minutes against starts, so a
            # fringe player with 900 minutes off the bench and 3 starts came out
            # at 300 "minutes per start". The model clamps to 90, which turned
            # exactly the players who rarely start into guaranteed full-90 men.
            # Capped, and only trusted once there are enough starts to mean
            # anything - below that the caller's own default is more honest.
            mins_per_start=(min(90.0, mins / starts) if starts >= 5 else 0.0),
            ppg=_num(row, "points_per_game"),
        )
    return prior, skipped


# Filled in by build_standalone.py: the whole prior, compressed, carried inside
# the bundle.
#
# Without it the hosted app re-downloads two seasons from GitHub every time the
# disk is wiped - which is every redeploy - making a cold start slow and putting
# the single most important input behind someone else's uptime. A failed fetch
# does not error; it silently drops every player to a positional average, which
# is the failure mode this file exists to prevent.
EMBEDDED_PRIOR = ""


def load():
    """
    Returns ({code: rates}, meta).

    Prefers a locally seeded file, because `python run.py seed` is how you
    refresh; falls back to whatever was built into the bundle; returns empty
    only if there is genuinely nothing.
    """
    if os.path.exists(SEED_FILE):
        try:
            with open(SEED_FILE, encoding="utf-8") as f:
                d = json.load(f)
            if d.get("prior"):
                return d["prior"], d.get("meta", {})
        except Exception:
            pass
    if EMBEDDED_PRIOR:
        try:
            import base64, zlib
            d = json.loads(zlib.decompress(base64.b64decode(EMBEDDED_PRIOR)).decode("utf-8"))
            meta = dict(d.get("meta", {}))
            meta["embedded"] = True
            return d.get("prior", {}), meta
        except Exception:
            pass
    return {}, {}
