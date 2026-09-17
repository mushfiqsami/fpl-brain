#!/usr/bin/env python3
"""
Where the points actually are: the armband.

strategy_test.py found that squad shape - premium-heavy against balanced,
points against value - is indistinguishable over a season. Fixture difficulty
explains about 2.5% of week-to-week variance and recent form about 2%. So the
levers everyone argues about are nearly worthless.

The captain is different in kind. It is the one pick that is doubled, and a
double is not diluted by the other ten scores the way a single player's good
week is. This measures how much that is worth: the same squad, scored on real
results, under captain policies ranging from the worst plausible choice to
perfect hindsight.

The gap between "highest projected" and "perfect" is the prize still on the
table. The gap between "highest projected" and "random starter" is what the
model is already banking.

    python captain_test.py
"""
from __future__ import annotations
import statistics, sys, os, random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fplbrain.api import ArchiveClient
from fplbrain.model import TeamStrength, FixtureModel, PlayerModel
from fplbrain import optimise
from backtest import build_state, f


def main(test_gws=tuple(range(6, 38, 2)), season="2025-26"):
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
    rng = random.Random(7)
    pol = {k: [] for k in ("best projected", "2nd best projected", "random starter",
                           "worst starter", "PERFECT (hindsight)")}
    base_xi = []

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
        base_xi.append(xi_pts)
        order = sorted(xi, key=lambda i: -ep[i][gw])
        picks = {
            "best projected": order[0],
            "2nd best projected": order[1] if len(order) > 1 else order[0],
            "random starter": rng.choice(xi),
            "worst starter": order[-1],
            "PERFECT (hindsight)": max(xi, key=lambda i: actual.get(i, 0.0)),
        }
        for k, pid in picks.items():
            pol[k].append(xi_pts + actual.get(pid, 0.0))

    n = len(base_xi)
    print(f"CAPTAIN POLICIES - {season}, {n} walk-forward gameweeks, real points\n")
    print(f"{'policy':<24}{'mean GW':>10}{'per season':>13}{'vs best proj':>14}")
    print("-" * 61)
    ref = statistics.fmean(pol["best projected"])
    rows = sorted(((statistics.fmean(v), k) for k, v in pol.items()), reverse=True)
    for mean, k in rows:
        print(f"{k:<24}{mean:>10.1f}{mean * 38:>13.0f}{(mean - ref) * 38:>+13.0f}")
    print(f"\n  XI alone, no captain : {statistics.fmean(base_xi):.1f} a gameweek")
    up = statistics.fmean(pol["PERFECT (hindsight)"]) - ref
    dn = ref - statistics.fmean(pol["worst starter"])
    print(f"  captaincy is worth {(up + dn) * 38:.0f} points a season end to end")
    print(f"     already banked over the worst choice : {dn * 38:+.0f}")
    print(f"     still on the table before perfection  : {up * 38:+.0f}")
    d = [a - b for a, b in zip(pol["best projected"], pol["random starter"])]
    se = statistics.pstdev(d) / len(d) ** 0.5
    print(f"\n  best-projected minus random: {statistics.fmean(d):+.2f} a gameweek "
          f"(se {se:.2f}) -> {statistics.fmean(d) * 38:+.0f} a season, "
          f"{'significant' if abs(statistics.fmean(d)) > 1.96 * se else 'not significant'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
