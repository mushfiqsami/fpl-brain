#!/usr/bin/env python3
"""
Which STRATEGY actually wins?

Every other test here asks whether the model ranks players correctly. This asks
a different and more useful question: given that the model is what it is, which
way of USING it scores most? Squad shape is a choice the model never makes -
how much to spend on premiums, whether to captain the highest mean or the
highest ceiling, whether to chase fixtures - and those choices are argued about
endlessly on instinct.

So they are run as a tournament, walk-forward on 2025/26. At each checkpoint a
squad is built from what was knowable BEFORE that gameweek under each strategy,
then scored on what actually happened. No hindsight anywhere.

    python strategy_test.py
"""
from __future__ import annotations
import statistics, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fplbrain.api import ArchiveClient
from fplbrain.model import TeamStrength, FixtureModel, PlayerModel, POS_NAME
from fplbrain import optimise
from backtest import build_state, f


def strategies():
    """(name, price cap per player, max players above £8.0m, pick on value?)"""
    return [
        ("max points (what we do now)", 99.0, 15, False),
        ("no player above £10m",        10.0, 15, False),
        ("no player above £9m",          9.0, 15, False),
        ("at most 2 above £8m",         99.0,  2, False),
        ("at most 1 above £8m",         99.0,  1, False),
        ("best points per £m",          99.0, 15, True),
    ]


def main(test_gws=(8, 12, 16, 20, 24, 28, 32, 36), season="2025-26"):
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
    results = {s[0]: [] for s in strategies()}
    spend = {s[0]: [] for s in strategies()}

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

        for name, cap, max_prem, by_value in strategies():
            cand = [p for p in pool if p["price"] <= cap]
            table = ep
            if by_value:
                # rank on points per £m instead of points
                table = {i: {gw: ep[i][gw] / max(0.1, by_id[i]["price"])} for i in ep}
            banned = []
            if max_prem < 15:
                prem = sorted((p for p in cand if p["price"] > 8.0),
                              key=lambda p: -ep[p["id"]][gw])
                banned = [p["id"] for p in prem[max_prem:]]
            try:
                built = optimise.build_squad(cand, table, 100.0, [gw], decay=1.0,
                                             banned=banned)
                if built["status"] != "Optimal":
                    continue
                ids = [p["id"] for p in built["squad"]]
                # XI and captain always chosen on real expected points, never on
                # the value proxy - value is a BUYING rule, not a starting one.
                xi, capt, _v, _b, _d = optimise.rank_xi(ids, ep, gw, by_id)
            except Exception:
                continue
            got = sum(actual.get(i, 0.0) for i in xi) + actual.get(capt, 0.0)
            results[name].append(got)
            spend[name].append(sum(by_id[i]["price"] for i in xi))

    print(f"STRATEGY TOURNAMENT - {season}, walk-forward, scored on real points\n")
    print(f"{'strategy':<30}{'GWs':>5}{'mean':>9}{'sd':>8}{'XI spend':>10}{'per season':>12}")
    print("-" * 74)
    rows = []
    for name, _c, _m, _v in strategies():
        v = results[name]
        if not v:
            continue
        rows.append((statistics.fmean(v), name, v, statistics.fmean(spend[name])))
    rows.sort(reverse=True)
    for mean, name, v, sp in rows:
        print(f"{name:<30}{len(v):>5}{mean:>9.1f}{statistics.pstdev(v):>8.1f}"
              f"{sp:>9.1f}m{mean * 38:>12.0f}")
    if len(rows) > 1:
        best, worst = rows[0], rows[-1]
        pair = [a - b for a, b in zip(best[2], worst[2])]
        if len(pair) > 1:
            se = statistics.pstdev(pair) / len(pair) ** 0.5
            d = statistics.fmean(pair)
            print(f"\n  best minus worst: {d:+.1f} a gameweek (se {se:.1f})"
                  f" = {d * 38:+.0f} over a season")
            print(f"  {'significant' if abs(d) > 1.96 * se else 'NOT significant - these strategies are indistinguishable'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
