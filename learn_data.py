#!/usr/bin/env python3
"""
Build the walk-forward dataset for the learned correction layer.

One row per player per gameweek of 2025/26. Every feature is something that was
knowable BEFORE that gameweek's deadline: the model's own projection built from
the state after the previous gameweek, the transfer market in the run-up to the
deadline, ownership, price, venue, fixture difficulty and recent minutes/points.
The target is the points actually scored.

    python learn_data.py        -> learn_data.npz
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fplbrain.api import ArchiveClient
from fplbrain.model import TeamStrength, FixtureModel, PlayerModel
from backtest import build_state, f

FEATURES = ["ep", "p_appear", "exp_min", "market", "own", "price", "home", "fdr",
            "min_l1", "min_l3", "pts_l3", "start_l5", "pos_gk", "pos_def", "pos_mid", "pos_fwd"]


def main(season="2025-26", first=6, last=38, out="learn_data.npz"):
    ac = ArchiveClient(season)
    teams = ac.teams(); tbn = {t["name"]: int(t["id"]) for t in teams}
    boot = [dict(id=int(t["id"]), name=t["name"], short_name=t["short_name"],
                 strength_attack_home=int(t["strength_attack_home"]),
                 strength_attack_away=int(t["strength_attack_away"]),
                 strength_defence_home=int(t["strength_defence_home"]),
                 strength_defence_away=int(t["strength_defence_away"])) for t in teams]
    fixtures, fdr = [], {}
    for r in ac.fixtures():
        try:
            fx = dict(id=int(r["id"]), event=int(r["event"]) if r["event"] else None,
                      team_h=int(r["team_h"]), team_a=int(r["team_a"]),
                      finished=(str(r["finished"]).lower() == "true"),
                      team_h_score=float(r["team_h_score"]) if r["team_h_score"] else None,
                      team_a_score=float(r["team_a_score"]) if r["team_a_score"] else None)
            fixtures.append(fx)
            fdr[(fx["id"], fx["team_h"])] = int(r["team_h_difficulty"])
            fdr[(fx["id"], fx["team_a"])] = int(r["team_a_difficulty"])
        except Exception:
            pass
    cache = {}
    for g in range(1, last + 1):
        cache[g] = ac.gw(g)
    # per-gameweek per-player aggregates (double gameweeks summed)
    agg = {}
    for g, rows in cache.items():
        d = {}
        for r in rows:
            eid = int(r["element"])
            a = d.setdefault(eid, dict(pts=0.0, mins=0.0, starts=0.0, rows=[]))
            a["pts"] += f(r, "total_points"); a["mins"] += f(r, "minutes")
            a["starts"] += f(r, "starts"); a["rows"].append(r)
        agg[g] = d
    X, Y, G, E = [], [], [], []
    for gw in range(first, last + 1):
        state = build_state(ac, gw - 1, tbn, cache)
        known = []
        for x in fixtures:
            y = dict(x)
            if y["event"] is not None and y["event"] >= gw:
                y["finished"] = False; y["team_h_score"] = y["team_a_score"] = None
            known.append(y)
        ts = TeamStrength.build(dict(teams=boot, elements=list(state.values()), events=[]),
                                known, prior_weight_games=6.0)
        fm = FixtureModel(ts, 1.10, 0.90); pm = PlayerModel(ts, fm)
        pm.start_rates = {eid: np.mean([min(1.0, agg[g].get(eid, {}).get("starts", 0)) for g in range(max(1, gw - 5), gw)])
                          for eid in state}
        pm.calibrate_depth(list(state.values()))
        views = fm.team_view(known, gw)
        total_sel = sum(f(r, "selected") for r in cache[gw]) / 15.0 or 1.0
        for eid, e in state.items():
            now = agg[gw].get(eid)
            if not now:
                continue
            r0 = now["rows"][0]
            pr = pm.project(e, views.get(e["team"], []))
            sel = f(r0, "selected")
            prev = [agg[g].get(eid, {}) for g in range(max(1, gw - 3), gw)]
            prev5 = [agg[g].get(eid, {}) for g in range(max(1, gw - 5), gw)]
            pos = e["element_type"]
            X.append([pr["ep"], pr["p_appear"], pr["exp_min"],
                      (f(r0, "transfers_balance") / sel) if sel > 0 else 0.0,
                      sel / total_sel, f(r0, "value") / 10.0,
                      1.0 if str(r0.get("was_home")).lower() == "true" else 0.0,
                      float(fdr.get((int(r0["fixture"]), e["team"]), 3)),
                      agg[gw - 1].get(eid, {}).get("mins", 0.0) if gw > 1 else 0.0,
                      np.mean([p.get("mins", 0.0) for p in prev]) if prev else 0.0,
                      np.mean([p.get("pts", 0.0) for p in prev]) if prev else 0.0,
                      np.mean([min(1.0, p.get("starts", 0.0)) for p in prev5]) if prev5 else 0.0,
                      pos == 1, pos == 2, pos == 3, pos == 4])
            Y.append(now["pts"]); G.append(gw); E.append(eid)
        print("GW%d rows so far %d" % (gw, len(Y)), flush=True)
    np.savez(out, X=np.array(X, dtype=float), Y=np.array(Y), G=np.array(G), E=np.array(E),
             features=np.array(FEATURES))
    print("saved", out, np.array(X).shape)


if __name__ == "__main__":
    main()
