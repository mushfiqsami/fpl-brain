"""
Weekly transfer-momentum snapshots for model_v2.

FPL's bootstrap-static only ever exposes CURRENT transfers_in_event /
transfers_out_event - there is no historical endpoint. Without this file,
testing model_v2's momentum term against a past gameweek silently used
whatever today's transfer counts happened to be, which is not what they were
at the time and made that part of the backtest meaningless.

record() is called once per real gameweek from run_job() (dashboard.py), right
after bootstrap-static is fetched and the current gameweek id is known, so the
live production app builds this history automatically as the season goes on.
Nothing reads a snapshot before it exists - a gameweek with no recorded
snapshot has no real momentum to test with, and model_ab_test.py should treat
that as "unavailable", not silently substitute today's numbers.

Lives under data/, so it is gitignored and ephemeral like everything else
there (see CLAUDE.md) - it rebuilds week by week rather than needing to exist.
"""
from __future__ import annotations
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "..", "data", "momentum_history.json")


def _load_raw():
    if not os.path.exists(STORE):
        return {}
    try:
        with open(STORE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def record(bs, gw):
    """Snapshot every player's current transfers_in_event/out_event under this gw.

    Safe to call more than once for the same gameweek (e.g. a forced refresh) -
    it overwrites that gameweek's entry rather than appending, so the snapshot
    always reflects the most recent call for that gw, not the first.
    """
    data = _load_raw()
    data[str(gw)] = {
        str(e["id"]): [int(e.get("transfers_in_event") or 0),
                       int(e.get("transfers_out_event") or 0)]
        for e in bs["elements"]
    }
    os.makedirs(os.path.dirname(STORE), exist_ok=True)
    tmp = STORE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, STORE)


def for_gw(gw):
    """{element_id: (transfers_in_event, transfers_out_event)} for one real gameweek,
    or {} if that gameweek was never snapshotted - callers must treat that as
    unavailable, not fall back to today's live numbers."""
    data = _load_raw()
    raw = data.get(str(gw))
    if not raw:
        return {}
    return {int(eid): tuple(v) for eid, v in raw.items()}
