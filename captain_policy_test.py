#!/usr/bin/env python3
"""
Which captain rule actually banks points?

captain_test.py measured the prize: perfect captaincy is worth +247 points a
season over picking the highest projection, and the highest projection is worth
only +40 over picking at random - a difference that is not statistically
distinguishable from zero. So nearly all of the armband's value is in timing,
and timing is exactly what the model is weakest at (within-player correlation
+0.159 against +0.713 across players).

That makes captaincy the single largest improvable lever in the whole system,
worth more than squad shape, fixture difficulty and form combined - all three of
which measured as noise. This tries every capturable rule against real results
to find one that banks part of the 247.

    python captain_policy_test.py
"""
from __future__ import annotations
import statistics, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fplbrain.api import ArchiveClient
from fplbrain.model import (TeamStrength, FixtureModel, PlayerModel,
                            penalty_uplift, setpiece_uplift, POS_NAME)
from fplbrain.sim import PlayerSim
from fplbrain import optimise
from backtest import build_state, f


def main(test_gws=tuple(range(6, 38, 2)), season="2025-26", runs=1500):
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
    POLICIES = ("highest projection", "highest ceiling", "most likely to haul",
                "projection + ceiling (current)", "highest projection at home",
                "cheapest of the top three")
    pol = {k: [] for k in POLICIES}
    perfect, base = [], []

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
        actual = {int(r["element"]): f(r, "total_points") for r in cache[gw]}

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
        xi_pts = sum(actual.get(i, 0.0) for i in xi)
        base.append(xi_pts)
        perfect.append(xi_pts + max(actual.get(i, 0.0) for i in xi))

        dist, home = {}, {}
        for i in xi:
            e = state[i]
            fxs = views.get(e["team"], [])
            pa, p60, em = pm.minutes_profile(e)
            sr = min(1.0, p60 / 0.9) if p60 > 0 else 0.0
            ps = min(pa, sr)
            s = PlayerSim(pos=e["element_type"], rates=pm.rates(e), p_start=ps,
                          p_sub=max(0.0, pa - ps),
                          avg_start_mins=max(45.0, min(90.0, em / max(0.05, ps))),
                          fixtures=fxs, base_lambda=ts.base_lambda,
                          pen_share=penalty_uplift(e), sp_share=setpiece_uplift(e))
            dist[i] = s.run(n=runs, seed=i)
            home[i] = bool(fxs and fxs[0].get("home"))

        top3 = sorted(xi, key=lambda i: -ep[i][gw])[:3]
        homers = [i for i in xi if home[i]] or xi
        picks = {
            "highest projection": max(xi, key=lambda i: ep[i][gw]),
            "highest ceiling": max(xi, key=lambda i: dist[i]["p90"]),
            "most likely to haul": max(xi, key=lambda i: dist[i]["p_haul"]),
            "projection + ceiling (current)": max(
                xi, key=lambda i: dist[i]["mean"] + 0.35 * (dist[i]["p90"] - dist[i]["mean"])),
            "highest projection at home": max(homers, key=lambda i: ep[i][gw]),
            "cheapest of the top three": min(top3, key=lambda i: by_id[i]["price"]),
        }
        for k, pid in picks.items():
            pol[k].append(xi_pts + actual.get(pid, 0.0))

    n = len(base)
    print(f"CAPTAIN POLICIES - {season}, {n} walk-forward gameweeks, real points\n")
    ref = statistics.fmean(pol["highest projection"])
    print(f"{'policy':<34}{'mean':>8}{'season':>9}{'vs current':>12}{'sig?':>7}")
    print("-" * 70)
    rows = sorted(((statistics.fmean(v), k) for k, v in pol.items()), reverse=True)
    for mean, k in rows:
        d = [a - b for a, b in zip(pol[k], pol["highest projection"])]
        se = statistics.pstdev(d) / len(d) ** 0.5 if len(d) > 1 else 0
        md = statistics.fmean(d)
        sig = "-" if k == "highest projection" else ("YES" if abs(md) > 1.96 * se else "no")
        print(f"{k:<34}{mean:>8.1f}{mean * 38:>9.0f}{(mean - ref) * 38:>+12.0f}{sig:>7}")
    print(f"\n  no captain at all   : {statistics.fmean(base):.1f}  ({statistics.fmean(base)*38:.0f})")
    print(f"  PERFECT (hindsight) : {statistics.fmean(perfect):.1f}  "
          f"({statistics.fmean(perfect)*38:.0f})   headroom "
          f"{(statistics.fmean(perfect)-ref)*38:+.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
