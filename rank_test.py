#!/usr/bin/env python3
"""
Scoring points and CLIMBING are not the same problem.

Everything else here maximises expected points. But rank is decided by your
score minus everyone else's, and a player owned by half the field contributes
to both sides of that subtraction. Captain the template and a haul lifts you
and five million others together - the points arrive, the rank does not move.

So this measures captaincy the way rank actually works: points gained against
the field, where the field is weighted by who actually owns each player, taken
from the real ownership in the gameweek files.

    python rank_test.py
"""
from __future__ import annotations
import statistics, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fplbrain.api import ArchiveClient
from fplbrain.model import TeamStrength, FixtureModel, PlayerModel
from fplbrain import optimise
from backtest import build_state, f


def main(test_gws=tuple(range(6, 38)), season="2025-26"):
    ac = ArchiveClient(season)
    teams = ac.teams()
    tbn = {t["name"]: int(t["id"]) for t in teams}
    boot = [dict(id=int(t["id"]), name=t["name"], short_name=t["short_name"],
                 strength_attack_home=int(t["strength_attack_home"]),
                 strength_attack_away=int(t["strength_attack_away"]),
                 strength_defence_home=int(t["strength_defence_home"]),
                 strength_defence_away=int(t["strength_defence_away"])) for t in teams]
    fixtures = []
    for r in ac.fixtures():
        try:
            fixtures.append(dict(id=int(r["id"]),
                event=int(r["event"]) if r["event"] else None,
                team_h=int(r["team_h"]), team_a=int(r["team_a"]),
                finished=(str(r["finished"]).lower() == "true"),
                team_h_score=float(r["team_h_score"]) if r["team_h_score"] else None,
                team_a_score=float(r["team_a_score"]) if r["team_a_score"] else None))
        except Exception:
            continue
    cache = {}
    tmpl, diff, best = [], [], []
    RAW_T, RAW_D, OWN_T, OWN_D = [], [], [], []

    for gw in test_gws:
        state = build_state(ac, gw - 1, tbn, cache)
        if not state:
            continue
        known = []
        for x in fixtures:
            y = dict(x)
            if y["event"] is not None and y["event"] >= gw:
                y["finished"] = False
                y["team_h_score"] = y["team_a_score"] = None
            known.append(y)
        bs = dict(teams=boot, elements=list(state.values()), events=[])
        ts = TeamStrength.build(bs, known, prior_weight_games=6.0)
        fm = FixtureModel(ts, 1.10, 0.90)
        pm = PlayerModel(ts, fm)
        if hasattr(pm, "calibrate_depth"):
            pm.calibrate_depth(list(state.values()))
        views = fm.team_view(known, gw)
        if gw not in cache:
            cache[gw] = ac.gw(gw)
        actual, owned = {}, {}
        tot_sel = 0.0
        for r in cache[gw]:
            eid = int(r["element"])
            actual[eid] = f(r, "total_points")
            owned[eid] = f(r, "selected")
            tot_sel += owned[eid]
        if tot_sel <= 0:
            continue
        # share of managers holding each player, normalised to 15 picks a squad
        share = {i: (owned[i] / tot_sel) * 15.0 for i in owned}

        pool, ep = [], {}
        for eid, e in state.items():
            proj = pm.project(e, views.get(e["team"], []))
            if proj["ep"] <= 0:
                continue
            ep[eid] = {gw: proj["ep"]}
            pool.append(dict(id=eid, name=e.get("web_name", str(eid)), club_id=e["team"],
                             pos=e["element_type"], price=e["now_cost"] / 10.0))
        if len(pool) < 150:
            continue
        by_id = {p["id"]: p for p in pool}
        built = optimise.build_squad(pool, ep, 100.0, [gw], decay=1.0)
        if built["status"] != "Optimal":
            continue
        ids = [p["id"] for p in built["squad"]]
        xi, _c, _v, _b, _d = optimise.rank_xi(ids, ep, gw, by_id)

        # what the armband gains you OVER THE FIELD: the doubled score minus the
        # share of the field that also doubled him.
        def edge(pid):
            return actual.get(pid, 0.0) * (1.0 - min(1.0, share.get(pid, 0.0)))
        t = max(xi, key=lambda i: share.get(i, 0.0))               # the template pick
        d_pool = [i for i in xi if share.get(i, 0.0) < 0.10] or xi  # under 10% owned
        d = max(d_pool, key=lambda i: ep[i][gw])
        tmpl.append(edge(t)); diff.append(edge(d))
        best.append(max(edge(i) for i in xi))
        RAW_T.append(actual.get(t,0.0)); RAW_D.append(actual.get(d,0.0))
        OWN_T.append(share.get(t,0.0)); OWN_D.append(share.get(d,0.0))

    n = len(tmpl)
    print(f"CAPTAINCY MEASURED AS RANK GAIN - {season}, {n} gameweeks\n")
    print("  points the armband wins you OVER THE FIELD (owner-weighted):\n")
    print(f"{'captain choice':<34}{'mean':>9}{'season':>10}{'sd':>8}")
    print("-" * 61)
    for lab, v in (("the template (most-owned)", tmpl),
                   ("a differential (<10% owned)", diff),
                   ("PERFECT (hindsight)", best)):
        print(f"{lab:<34}{statistics.fmean(v):>9.2f}{statistics.fmean(v)*38:>10.0f}"
              f"{statistics.pstdev(v):>8.2f}")
    print("")
    print("  RAW points scored by each captain (no ownership weighting):")
    print("     template     %.2f a gameweek, avg ownership %.0f%%"
          % (statistics.fmean(RAW_T), statistics.fmean(OWN_T) * 100))
    print("     differential %.2f a gameweek, avg ownership %.0f%%"
          % (statistics.fmean(RAW_D), statistics.fmean(OWN_D) * 100))
    rd = [a - b for a, b in zip(RAW_D, RAW_T)]
    rse = statistics.pstdev(rd) / len(rd) ** 0.5
    verdict = ("differentials really do score less"
               if statistics.fmean(rd) < -1.96 * rse
               else "no significant raw-points penalty")
    print("     raw difference %+.2f (se %.2f) -> %s"
          % (statistics.fmean(rd), rse, verdict))
    dd = [a - b for a, b in zip(diff, tmpl)]
    se = statistics.pstdev(dd) / len(dd) ** 0.5
    m = statistics.fmean(dd)
    print(f"\n  differential minus template: {m:+.2f} a gameweek (se {se:.2f})"
          f" = {m*38:+.0f} a season")
    print(f"  {'SIGNIFICANT' if abs(m) > 1.96*se else 'not significant'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
