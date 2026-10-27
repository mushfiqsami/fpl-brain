#!/usr/bin/env python3
"""
Do high-rated players who keep blanking actually under-deliver? (the "fraud check")

Walk-forward over 2025-26 with the production player model (prior K=150, prior from
earlier seasons only). For each test gameweek, for every player the model rates
highly, compute from games STRICTLY BEFORE it:

  streak   consecutive most-recent appearances (45+ min) scoring 2 or fewer points
  gap      (goals + assists) - xGI over the last 8 appearances  (negative = finishing/
           creating below his chances, the "big xG but keeps blanking" profile)

Part 1 - diagnosis: among the model's top-100 projections each week, is
actual - projected different for players on a drought? If it is ~0, a penalty would
be punishing luck; if clearly negative, the model is over-rating them.

Part 2 - remedy: apply a multiplicative EP penalty by streak and by xGI gap and see
whether ranking (rho, top-30) and the best 100m squad improve. Both directions of
the gap are tried, because regression to the mean says a negative gap should be
rewarded, not punished.

    python fraud_check_test.py
"""
from __future__ import annotations
import os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fplbrain.api import ArchiveClient
from fplbrain import model as M, optimise, seed
from fplbrain.model import TeamStrength, FixtureModel, PlayerModel
from backtest import build_state, f

SEASON = "2025-26"
TEST_GWS = (6, 8, 10, 12, 14, 16, 19, 22, 25, 28, 31, 34)


def features(history_rows):
    """history_rows: that player's rows, oldest first, strictly before the test gw."""
    apps = [r for r in history_rows if f(r, "minutes") >= 45]
    streak = 0
    for r in reversed(apps):
        if f(r, "total_points") <= 2:
            streak += 1
        else:
            break
    last = apps[-8:]
    gap = (sum(f(r, "goals_scored") + f(r, "assists") for r in last)
           - sum(f(r, "expected_goal_involvements") for r in last)) if last else 0.0
    return streak, gap, len(apps)


def main():
    ac = ArchiveClient(SEASON)
    code_of = {int(r["id"]): r["code"] for r in ac.players_raw()}
    teams = ac.teams()
    tbn = {t["name"]: int(t["id"]) for t in teams}
    boot_teams = [dict(id=int(t["id"]), name=t["name"], short_name=t["short_name"],
                       strength_attack_home=int(t["strength_attack_home"]),
                       strength_attack_away=int(t["strength_attack_away"]),
                       strength_defence_home=int(t["strength_defence_home"]),
                       strength_defence_away=int(t["strength_defence_away"])) for t in teams]
    fixtures_all = []
    for r in ac.fixtures():
        try:
            fixtures_all.append(dict(
                id=int(r["id"]), event=int(r["event"]) if r["event"] else None,
                team_h=int(r["team_h"]), team_a=int(r["team_a"]),
                finished=(str(r["finished"]).lower() == "true"),
                team_h_score=float(r["team_h_score"]) if r["team_h_score"] else None,
                team_a_score=float(r["team_a_score"]) if r["team_a_score"] else None,
                team_h_difficulty=3, team_a_difficulty=3))
        except Exception:
            continue
    prior = seed._blend([(seed._fetch("2024-25")[0], 1.0), (seed._fetch("2023-24")[0], 0.55)])

    cache = {}
    # diagnosis accumulators: bucket -> list of (actual - projected)
    diag = {}
    # remedy accumulators: label -> dict(rho, top, real)
    REM = [("baseline", None)]
    for a in (0.05, 0.10, 0.20):
        REM.append((f"streak>=3 penalty {a:.2f}/game (max 6)", ("streak", a)))
    for a in (0.10, 0.20):
        REM.append((f"gap<-1 PENALISE x{1-a:.2f}", ("gap_pen", a)))
        REM.append((f"gap<-1 REWARD  x{1+a:.2f}", ("gap_rew", a)))
    res = {l: dict(rho=[], top=[], real=[]) for l, _ in REM}

    for gw in TEST_GWS:
        state = build_state(ac, gw - 1, tbn, cache)
        for eid, e in state.items():
            e["code"] = code_of.get(eid)
        fx_known = []
        for x in fixtures_all:
            y = dict(x)
            if y["event"] is not None and y["event"] >= gw:
                y["finished"] = False
                y["team_h_score"] = y["team_a_score"] = None
            fx_known.append(y)
        boot = dict(teams=boot_teams, elements=list(state.values()), events=[])
        ts = TeamStrength.build(boot, fx_known, prior_weight_games=6.0)
        fm = FixtureModel(ts, 1.10, 0.90)
        views = fm.team_view(fx_known, gw)
        for g in range(1, gw + 1):
            if g not in cache:
                cache[g] = ac.gw(g)
        actual = {int(r["element"]): f(r, "total_points") for r in cache[gw]}
        played = {int(r["element"]) for r in cache[gw] if f(r, "minutes") >= 1}
        hist = {}
        for g in range(1, gw):
            for r in cache[g]:
                hist.setdefault(int(r["element"]), []).append(r)

        pm = PlayerModel(ts, fm, player_prior=prior)
        pm.prior_minutes_weight = 150.0
        pm.calibrate_depth(list(state.values()))
        rows = []
        for eid, e in state.items():
            p = pm.project(e, views.get(e["team"], []))
            if p["ep"] <= 0:
                continue
            streak, gap, napps = features(hist.get(eid, []))
            rows.append(dict(id=eid, e=e, ep=p["ep"], streak=streak, gap=gap, napps=napps))

        # ---- part 1: diagnosis on the model's top 100 who actually played
        top = sorted(rows, key=lambda r: -r["ep"])[:100]
        for r in top:
            if r["id"] not in played:
                continue
            res_ = actual.get(r["id"], 0) - r["ep"]
            b = "streak 0-1" if r["streak"] <= 1 else "streak 2" if r["streak"] == 2 else "streak 3" if r["streak"] == 3 else "streak 4+"
            diag.setdefault(("streak", b), []).append(res_)
            if r["napps"] >= 6:
                gb = "gap <= -1.5" if r["gap"] <= -1.5 else "gap -1.5..-0.5" if r["gap"] <= -0.5 else "gap -0.5..+0.5" if r["gap"] < 0.5 else "gap >= +0.5"
                diag.setdefault(("gap", gb), []).append(res_)

        # ---- part 2: remedies
        for label, rule in REM:
            pool, ep, pairs = [], {}, []
            for r in rows:
                m = 1.0
                if rule and r["e"]["element_type"] != 1:
                    kind, a = rule
                    if kind == "streak" and r["streak"] >= 3:
                        m = 1.0 - a * min(r["streak"] - 2, 4)
                    elif kind == "gap_pen" and r["napps"] >= 6 and r["gap"] <= -1.0:
                        m = 1.0 - a
                    elif kind == "gap_rew" and r["napps"] >= 6 and r["gap"] <= -1.0:
                        m = 1.0 + a
                x = r["ep"] * m
                e = r["e"]
                pool.append(dict(name=e["web_name"], club_id=e["team"], pos=e["element_type"],
                                 price=e["now_cost"] / 10.0, id=r["id"]))
                ep[r["id"]] = {gw: x}
                if r["id"] in played:
                    pairs.append((x, actual.get(r["id"], 0)))
            rho = statistics.correlation([a for a, _ in pairs], [b for _, b in pairs])
            topm = statistics.fmean(b for _, b in sorted(pairs, key=lambda t: -t[0])[:30])
            real = None
            try:
                built = optimise.build_squad(pool, ep, 100.0, [gw], decay=1.0)
                if built["status"] == "Optimal":
                    ids = [q["id"] for q in built["squad"]]
                    xi, cap, _v, _b, _d = optimise.rank_xi(ids, ep, gw, {q["id"]: q for q in pool})
                    real = sum(actual.get(i, 0) for i in xi) + actual.get(cap, 0)
            except Exception:
                pass
            R = res[label]
            R["rho"].append(rho); R["top"].append(topm)
            if real is not None:
                R["real"].append(real)
        print(f"  GW{gw} done", flush=True)

    print("\n" + "=" * 78)
    print("PART 1 - among the model's top-100 each week: actual points minus projection")
    print("=" * 78)
    for key in sorted(diag):
        v = diag[key]
        se = statistics.stdev(v) / len(v) ** 0.5 if len(v) > 2 else 0
        print(f"  {key[0]:6} {key[1]:16} n={len(v):4d}   mean residual {statistics.fmean(v):+6.2f}  (se {se:.2f})")
    print("\n" + "=" * 78)
    print("PART 2 - remedies")
    print("=" * 78)
    print(f"{'rule':42}{'rho':>8}{'top30':>8}{'squad':>8}")
    for label, _ in REM:
        R = res[label]
        print(f"{label:42}{statistics.fmean(R['rho']):8.4f}{statistics.fmean(R['top']):8.2f}"
              f"{statistics.fmean(R['real']) if R['real'] else float('nan'):8.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
