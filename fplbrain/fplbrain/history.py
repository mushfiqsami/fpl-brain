"""
Player trend tracking.

The point of this file is to catch a player *while* he is becoming good, not
after everyone has noticed. A single run of the model is a snapshot: it says who
is worth owning today and nothing at all about who is on the way up. Two
snapshots compared say considerably more.

So every run writes what it believed about every player, and the next run diffs
against it. What comes out is movement - projection rising, ceiling rising,
price rising, ownership rising - which is the earliest signal available that the
model has changed its mind about someone.

Two numbers are reported per player, matching how the owner asked for them:

  aps      Average Performance Score. Points per gameweek the model expects,
           averaged over the horizon it can see. A level, not a change.

  balance  GW Balance. How much that level moved. Positive means the model
           thinks more of him than it did last time.

A note on the arithmetic, because it is easy to get wrong: `balance` is a
*difference between levels*, not something that accumulates into `aps`. Adding
each week's balance onto the score would compound - a player on a good run would
climb without limit and never come back down. The level is recomputed from the
model each run and the balance simply reports how far it moved, which keeps the
two consistent no matter how many gameweeks pass.

Storage is a small ring of recent snapshots in data/, so this survives restarts
locally. On an ephemeral host it rebuilds itself from whatever runs it has seen.
"""
from __future__ import annotations
import json, os, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST_FILE = os.path.join(HERE, "data", "player_history.json")
KEEP = 40                      # snapshots retained; ~a season of daily runs


def _read():
    try:
        with open(HIST_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"snapshots": []}


def _write(obj):
    os.makedirs(os.path.dirname(HIST_FILE), exist_ok=True)
    tmp = HIST_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, HIST_FILE)


def snapshot(gw, rows):
    """
    Record what the model believes right now.

    rows: {player_id: {aps, ceiling, price, owned, status, name, club, pos}}
    Stored under the gameweek it was taken for, so a diff never compares two
    different gameweeks' expectations and calls the difference "movement".
    """
    d = _read()
    d["snapshots"].append({"t": int(time.time()), "gw": gw, "players": rows})
    d["snapshots"] = d["snapshots"][-KEEP:]
    _write(d)
    return d


def previous(gw):
    """
    The last snapshot stored for this gameweek, if any.

    Callers diff *before* recording the current run, so the newest stored
    snapshot is the previous run - taking [-2] here instead lagged every reading
    by a full cycle, reporting last time's movement as though it were this
    time's and showing a genuine change as zero.
    """
    same = [s for s in _read()["snapshots"] if s.get("gw") == gw]
    return same[-1] if same else None


def movement(gw, rows):
    """
    Compare the current beliefs with the last run's.

    Returns a list sorted by how much each player's expected output moved -
    biggest risers first, biggest fallers last, which is the order the list is
    actually read in. Players with no earlier reading are reported as new rather
    than as a change of zero, so a fresh install does not look like a flat week.
    """
    prev = previous(gw)
    old = (prev or {}).get("players") or {}
    out = []
    for pid, r in rows.items():
        o = old.get(str(pid)) or old.get(pid)
        if o:
            bal = round(r["aps"] - o.get("aps", r["aps"]), 3)
            d_ceiling = round(r.get("ceiling", 0) - o.get("ceiling", 0), 2)
            d_price = round(r.get("price", 0) - o.get("price", 0), 1)
            d_owned = round(r.get("owned", 0) - o.get("owned", 0), 1)
            fresh = False
        else:
            bal = d_ceiling = d_price = d_owned = 0.0
            fresh = True
        out.append(dict(id=int(pid), name=r["name"], club=r["club"], pos=r["pos"],
                        price=r.get("price", 0), owned=r.get("owned", 0),
                        status=r.get("status", "a"), news=r.get("news", ""),
                        aps=round(r["aps"], 2), balance=bal,
                        ceiling=r.get("ceiling", 0), d_ceiling=d_ceiling,
                        d_price=d_price, d_owned=d_owned, fresh=fresh))
    out.sort(key=lambda r: -r["balance"])
    return out, (prev or {}).get("t")
