#!/usr/bin/env python3
"""
Fit and validate the transfer-market adjustment.

Walk-forward on 2025/26. For each gameweek the model projects every player from
what was known before it; the transfer market is the net change in that
player's ownership in the run-up to the same deadline, which is also known
before the ball is kicked. The adjustment is fitted on odd gameweeks and judged
on even ones, so the headline numbers are out of sample.

    python market_fit.py
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fplbrain.api import ArchiveClient
from fplbrain.model import TeamStrength, FixtureModel, PlayerModel
from backtest import build_state, f


def collect(season="2025-26", gws=range(6, 38)):
    ac = ArchiveClient(season)
    teams = ac.teams(); tbn = {t["name"]: int(t["id"]) for t in teams}
    boot = [dict(id=int(t["id"]), name=t["name"], short_name=t["short_name"],
                 strength_attack_home=int(t["strength_attack_home"]),
                 strength_attack_away=int(t["strength_attack_away"]),
                 strength_defence_home=int(t["strength_defence_home"]),
                 strength_defence_away=int(t["strength_defence_away"])) for t in teams]
    fixtures = []
    for r in ac.fixtures():
        try:
            fixtures.append(dict(id=int(r["id"]), event=int(r["event"]) if r["event"] else None,
                team_h=int(r["team_h"]), team_a=int(r["team_a"]),
                finished=(str(r["finished"]).lower() == "true"),
                team_h_score=float(r["team_h_score"]) if r["team_h_score"] else None,
                team_a_score=float(r["team_a_score"]) if r["team_a_score"] else None))
        except Exception:
            pass
    cache, out = {}, []
    for gw in gws:
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
        pm.calibrate_depth(list(state.values()))
        views = fm.team_view(known, gw)
        if gw not in cache:
            cache[gw] = ac.gw(gw)
        rows = {int(r["element"]): r for r in cache[gw]}
        for eid, e in state.items():
            r = rows.get(eid)
            if not r:
                continue
            ep = pm.project(e, views.get(e["team"], []))["ep"]
            sel = f(r, "selected")
            if ep <= 1.0 or sel < 1000:
                continue
            out.append((gw, ep, f(r, "transfers_balance") / sel, f(r, "total_points")))
    return np.array(out)


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    d = collect()
    gw, ep, net, pts = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    net = np.clip(net, -0.5, 0.5)
    train, test = (gw % 2 == 1), (gw % 2 == 0)
    print("rows: %d  (train %d, test %d)\n" % (len(d), train.sum(), test.sum()))

    def design(mask, shape):
        e, n = ep[mask], net[mask]
        if shape == "linear":
            return np.column_stack([e, n])
        if shape == "split":
            return np.column_stack([e, np.minimum(n, 0), np.maximum(n, 0)])
        return np.column_stack([e])

    for shape in ("none", "linear", "split"):
        A = design(train, shape)
        b, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(A)), A]), pts[train], rcond=None)
        At = design(test, shape)
        pred = np.column_stack([np.ones(len(At)), At]) @ b
        print("%-7s out-of-sample R2 %.4f   coefs %s" % (shape, r2(pts[test], pred),
              " ".join("%+.3f" % x for x in b)))

    # The form the app will apply: keep the model's own EP untouched and ADD a
    # market term. Fit only that term, on training weeks, with EP fixed at 1.
    resid = pts[train] - ep[train]
    M = np.column_stack([np.minimum(net[train], 0), np.maximum(net[train], 0)])
    k, *_ = np.linalg.lstsq(M, resid - resid.mean(), rcond=None)
    adj = lambda n: k[0] * np.minimum(n, 0) + k[1] * np.maximum(n, 0)
    base = r2(pts[test], ep[test])
    withm = r2(pts[test], ep[test] + adj(net[test]))
    print("\nadditive form, fitted on odd GWs, judged on even GWs:")
    print("   dumping coefficient  %+.3f per unit share lost" % k[0])
    print("   buying  coefficient  %+.3f per unit share gained" % k[1])
    print("   out-of-sample R2: EP alone %.4f -> EP + market %.4f  (%+.4f)" % (base, withm, withm - base))
    for lo, hi, lab in ((-1, -0.10, "losing >10% of owners"), (-0.10, -0.03, "losing 3-10%"),
                        (-0.03, 0.03, "roughly stable"), (0.03, 0.10, "gaining 3-10%"),
                        (0.10, 9, "gaining >10%")):
        m = test & (net >= lo) & (net < hi)
        if m.sum() < 30:
            continue
        print("   %-24s n=%4d  model %.2f  model+market %.2f  actual %.2f" % (
            lab, m.sum(), ep[m].mean(), (ep[m] + adj(net[m])).mean(), pts[m].mean()))
    return k


if __name__ == "__main__":
    main()
