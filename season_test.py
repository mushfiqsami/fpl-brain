#!/usr/bin/env python3
"""
What would this system actually have scored?

backtest.py measures whether the model ranks players correctly, which is the
right question for picking a transfer and the wrong one for setting a target. It
also scores only players who featured, so it never counts the weeks somebody was
dropped - a real squad carries those.

This does the whole job end to end, walk-forward:

  1. Build the best legal £100m squad from what was knowable BEFORE gameweek N.
  2. Pick the XI and captain from projections alone. No hindsight anywhere.
  3. Score it with what those players ACTUALLY got, zeros included.

The output is a real points-per-gameweek figure for the system as a whole, which
is the only honest thing to compare a season target against. A projection is
what the model expects; this is what the expectation was worth.

    python season_test.py
"""
from __future__ import annotations
import statistics, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fplbrain.api import ArchiveClient
from fplbrain.model import TeamStrength, FixtureModel, PlayerModel
from fplbrain import optimise
from backtest import build_state, f


def main(test_gws=(8, 11, 14, 17, 20, 23, 26, 29, 32, 35), season="2025-26"):
    ac = ArchiveClient(season)
    teams = ac.teams()
    teams_by_name = {t["name"]: int(t["id"]) for t in teams}
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
                team_a_score=float(r["team_a_score"]) if r["team_a_score"] else None))
        except Exception:
            continue
    cache = {}

    print(f"Season simulation - {season}")
    print("Squad built from projections only, scored on actual points.\n")
    print(f"{'GW':>4}{'squad £m':>10}{'proj XI':>9}{'ACTUAL':>8}"
          f"{'captain':>16}{'cap got':>9}{'blanks':>8}")
    print("-" * 64)

    rows = []
    for gw in test_gws:
        state = build_state(ac, gw, teams_by_name, cache)
        if not state:
            continue
        # nothing from gw onward has been played yet
        fx_known = []
        for x in fixtures_all:
            y = dict(x)
            if y["event"] is not None and y["event"] >= gw:
                y["finished"] = False
                y["team_h_score"] = y["team_a_score"] = None
            fx_known.append(y)
        bootstrap = dict(teams=boot_teams, elements=list(state.values()), events=[])
        ts = TeamStrength.build(bootstrap, fx_known, prior_weight_games=6.0)
        fm = FixtureModel(ts, 1.10, 0.90)
        pm = PlayerModel(ts, fm)
        views = fm.team_view(fx_known, gw)

        if gw not in cache:
            cache[gw] = ac.gw(gw)
        actual = {int(r["element"]): f(r, "total_points") for r in cache[gw]}

        recent = {}
        for eid in state:
            hist = []
            for gg in range(max(1, gw - 5), gw):
                for r in cache.get(gg, []):
                    if int(r["element"]) == eid:
                        hist.append(f(r, "starts"))
            if hist:
                recent[eid] = sum(hist) / len(hist)
        pm.start_rates = recent

        pool, ep = [], {}
        for eid, e in state.items():
            proj = pm.project(e, views.get(e["team"], []))
            if proj["ep"] <= 0:
                continue
            ep[eid] = {gw: proj["ep"]}
            pool.append(dict(id=eid, name=e.get("web_name", str(eid)),
                             club_id=e["team"], pos=e["element_type"],
                             price=e["now_cost"] / 10.0))
        if len(pool) < 120:
            continue

        try:
            built = optimise.build_squad(pool, ep, 100.0, [gw], decay=1.0)
            ids = [p["id"] for p in built["squad"]]
            by_id = {p["id"]: p for p in pool}
            xi, cap, _v, _b, _d = optimise.rank_xi(ids, ep, gw, by_id)
        except Exception as exc:
            print(f"{gw:>4}  solver failed: {exc}")
            continue

        # score on reality. A player absent from the gameweek file did not play.
        got = sum(actual.get(i, 0.0) for i in xi)
        cap_got = actual.get(cap, 0.0)
        total = got + cap_got
        blanks = sum(1 for i in xi if actual.get(i, 0.0) <= 2)
        proj_xi = sum(ep[i][gw] for i in xi) + ep.get(cap, {}).get(gw, 0.0)
        rows.append((gw, built["cost"], proj_xi, total, cap_got, blanks))
        print(f"{gw:>4}{built['cost']:>10.1f}{proj_xi:>9.1f}{total:>8.0f}"
              f"{by_id[cap]['name'][:14]:>16}{cap_got:>9.0f}{blanks:>8}")

    if not rows:
        print("no gameweeks completed")
        return 1

    act = [r[3] for r in rows]
    prj = [r[2] for r in rows]
    print("-" * 64)
    print(f"{'MEAN':>4}{'':>10}{statistics.fmean(prj):>9.1f}{statistics.fmean(act):>8.1f}")
    print()
    print(f"  actual points per gameweek : {statistics.fmean(act):.2f}")
    print(f"  the model expected         : {statistics.fmean(prj):.2f}")
    bias = statistics.fmean(act) - statistics.fmean(prj)
    print(f"  bias (actual - expected)   : {bias:+.2f}"
          f"   {'model UNDER-projects' if bias > 0 else 'model OVER-projects'}")
    print(f"  spread across gameweeks    : {min(act):.0f} to {max(act):.0f}, "
          f"sd {statistics.pstdev(act):.1f}")
    # Paired differences, because each gameweek's projection and result belong
    # together. Judging the bias against the spread of raw scores would compare
    # it to week-to-week variance it has nothing to do with.
    diffs = [a - p_ for a, p_ in zip(act, prj)]
    n = len(diffs)
    sd = statistics.pstdev(diffs) if n > 1 else 0.0
    se = sd / (n ** 0.5) if n else 0.0
    lo, hi = statistics.fmean(diffs) - 1.96 * se, statistics.fmean(diffs) + 1.96 * se
    print(f"  paired bias                : {statistics.fmean(diffs):+.2f} "
          f"(sd {sd:.1f}, se {se:.1f}, n={n})")
    print(f"  95% interval               : {lo:+.1f} to {hi:+.1f}"
          f"   {'-> real bias' if lo > 0 or hi < 0 else '-> NOT significant, could be zero'}")
    print()
    print(f"  over 38 gameweeks that is  : {statistics.fmean(act) * 38:.0f} points")
    print("  no transfers, no chips, no price rises - a squad rebuilt fresh each")
    print("  time from projections alone, which is the floor for a managed season.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
