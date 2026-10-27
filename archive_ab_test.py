#!/usr/bin/env python3
"""
Model A/B harness - full completed season, archive data.

model_ab_test.py can only test against THIS season's few played gameweeks, and
momentum specifically can't be tested on any of them at all - the snapshot
mechanism (fplbrain/momentum.py) only started recording this session, so there
is no historical transfer data for GW1-4 to check it against.

The vaastav-style archive (fplbrain.api.ArchiveClient) solves both problems at
once: it is a complete, finished 38-gameweek season, and - unlike live
bootstrap-static - its per-gameweek player rows already include transfers_in,
transfers_out and selected (ownership count) for THAT gameweek specifically,
not just today. So momentum can finally be tested honestly, with real
historical numbers, not a neutral fallback.

Tests the same three candidates as model_ab_test.py (V1, V2 flat-form, the
V1+momentum hybrid) across a spread of real gameweeks through a full season,
each predicted from strictly-prior data - same walk-forward discipline as
backtest.py / season_test.py.

One approximation, stated plainly: the archive's per-player `selected` field
is a raw ownership COUNT, not a percentage, and there is no per-gameweek
"how many managers exist" figure in this dataset. TOTAL_PLAYERS_ESTIMATE below
is a fixed stand-in (roughly what a season this size actually drew) - it only
scales the momentum terms, which are capped anyway, so an estimate is fine for
comparing candidates against each other, not fine as a literal total.

    python archive_ab_test.py
"""
from __future__ import annotations
import statistics, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fplbrain.api import ArchiveClient
from fplbrain.model import TeamStrength, FixtureModel, PlayerModel
from fplbrain import optimise
from fplbrain.model_v2 import project_v2, momentum as v2_momentum, MOMENTUM_WEIGHT, REL_MOMENTUM_WEIGHT
from backtest import build_state, f

SEASON = "2025-26"
TEST_GWS = (8, 11, 14, 17, 20, 23, 26, 29, 32, 35)   # same cadence as season_test.py
TOTAL_PLAYERS_ESTIMATE = 9_000_000                    # see module docstring


def main():
    ac = ArchiveClient(SEASON)
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
                team_a_score=float(r["team_a_score"]) if r["team_a_score"] else None,
                team_h_difficulty=int(r["team_h_difficulty"]) if r.get("team_h_difficulty") else 3,
                team_a_difficulty=int(r["team_a_difficulty"]) if r.get("team_a_difficulty") else 3))
        except Exception:
            continue
    cache = {}

    print(f"Archive walk-forward - {SEASON}, gameweeks {TEST_GWS}\n")
    names = ["V1", "V2 (flat form)", "Hybrid (V1 + momentum)"]
    totals = {n: dict(rho=[], mae=[], real=[], proj=[]) for n in names}

    for gw in TEST_GWS:
        state = build_state(ac, gw - 1, teams_by_name, cache)
        if not state:
            continue
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
        mins_now = {int(r["element"]): f(r, "minutes") for r in cache[gw]}

        # this gameweek's own real transfer/ownership numbers - not today's,
        # not a neutral fallback, the actual historical figures for GW `gw`
        transfers_this_gw = {int(r["element"]): (f(r, "transfers_in"), f(r, "transfers_out"),
                                                   f(r, "selected"))
                              for r in cache[gw]}

        # this player's own real per-gameweek points, oldest first, for V2's
        # flat-form average - strictly gw 1..gw-1, no peeking
        history = {}
        for g in range(1, gw):
            if g not in cache:
                cache[g] = ac.gw(g)
            for r in cache[g]:
                eid = int(r["element"])
                history.setdefault(eid, []).append(dict(minutes=f(r, "minutes"),
                                                          total_points=f(r, "total_points")))

        diff = {}
        for x in fx_known:
            if x["event"] == gw:
                diff[x["team_h"]] = x["team_h_difficulty"]
                diff[x["team_a"]] = x["team_a_difficulty"]

        v1_pairs, v2_pairs, hy_pairs = [], [], []
        v1_pool, v1_ep = [], {}
        v2_pool, v2_ep = [], {}
        hy_pool, hy_ep = [], {}

        for eid, e in state.items():
            v = views.get(e["team"], [])
            proj = pm.project(e, v)
            fd = diff.get(e["team"], 3)
            t_in, t_out, selected = transfers_this_gw.get(eid, (0, 0, 0))
            sbp = (selected / TOTAL_PLAYERS_ESTIMATE) * 100.0
            mom_abs, mom_rel = v2_momentum(t_in, t_out, TOTAL_PLAYERS_ESTIMATE, sbp)
            hybrid_ep = proj["ep"] * (1.0 + MOMENTUM_WEIGHT * mom_abs + REL_MOMENTUM_WEIGHT * mom_rel)

            fake_element = dict(e, team=e["team"], selected_by_percent=sbp)
            v2r = project_v2(fake_element, history.get(eid, []), fd, TOTAL_PLAYERS_ESTIMATE,
                              transfers=(t_in, t_out))

            if eid in actual and mins_now.get(eid, 0) >= 1:
                v1_pairs.append((proj["ep"], actual[eid]))
                v2_pairs.append((v2r["ep"], actual[eid]))
                hy_pairs.append((hybrid_ep, actual[eid]))

            club = dict(name=e.get("web_name", str(eid)), club_id=e["team"], pos=e["element_type"],
                       price=e["now_cost"] / 10.0, id=eid)
            if proj["ep"] > 0:
                v1_pool.append(club); v1_ep[eid] = {gw: proj["ep"]}
            if v2r["ep"] > 0 and v2r["minutes_share"] >= 0.35:
                v2_pool.append(club); v2_ep[eid] = {gw: v2r["ep"]}
            if hybrid_ep > 0:
                hy_pool.append(club); hy_ep[eid] = {gw: hybrid_ep}

        def score(pool, ep, pairs, label):
            rho = statistics.correlation([p[0] for p in pairs], [p[1] for p in pairs]) if len(pairs) > 2 else None
            mae = sum(abs(x - y) for x, y in pairs) / len(pairs) if pairs else None
            real = proj_ = None
            try:
                built = optimise.build_squad(pool, ep, 100.0, [gw], decay=1.0)
                ids = [p["id"] for p in built["squad"]]
                by_id = {p["id"]: p for p in pool}
                xi, cap, _v, _b, _d = optimise.rank_xi(ids, ep, gw, by_id)
                real = sum(actual.get(i, 0) for i in xi) + actual.get(cap, 0)
                proj_ = sum(ep[i][gw] for i in xi) + ep.get(cap, {}).get(gw, 0.0)
            except Exception as exc:
                print(f"    ({label} squad build failed GW{gw}: {exc})")
            return rho, mae, real, proj_

        for label, pool, ep, pairs in (("V1", v1_pool, v1_ep, v1_pairs),
                                        ("V2 (flat form)", v2_pool, v2_ep, v2_pairs),
                                        ("Hybrid (V1 + momentum)", hy_pool, hy_ep, hy_pairs)):
            rho, mae, real, proj_ = score(pool, ep, pairs, label)
            rho_s = f"{rho:.4f}" if rho is not None else "n/a"
            mae_s = f"{mae:.3f}" if mae is not None else "n/a"
            print(f"  GW{gw:2d}  {label:24}  n={len(pairs):3d}  rho={rho_s}  MAE={mae_s}  "
                  f"real={real}  (proj {proj_:.1f})" if proj_ is not None else
                  f"  GW{gw:2d}  {label:24}  n={len(pairs):3d}  rho={rho_s}  MAE={mae_s}  real={real}")
            if rho is not None: totals[label]["rho"].append(rho)
            if mae is not None: totals[label]["mae"].append(mae)
            if real is not None:
                totals[label]["real"].append(real)
                totals[label]["proj"].append(proj_)

    print("\n" + "=" * 78)
    print(f"AGGREGATE ACROSS {len(TEST_GWS)} GAMEWEEKS, FULL SEASON {SEASON}")
    print("=" * 78)
    for name in names:
        t = totals[name]
        rho_m = statistics.fmean(t["rho"]) if t["rho"] else None
        mae_m = statistics.fmean(t["mae"]) if t["mae"] else None
        real_m = statistics.fmean(t["real"]) if t["real"] else None
        proj_m = statistics.fmean(t["proj"]) if t["proj"] else None
        print(f"{name}")
        print(f"  mean correlation      : {rho_m:.4f}" if rho_m is not None else "  mean correlation: n/a")
        print(f"  mean MAE              : {mae_m:.3f}" if mae_m is not None else "  mean MAE: n/a")
        print(f"  mean best-squad points: {real_m:.1f} real vs {proj_m:.1f} projected"
              if real_m is not None else "  mean best-squad points: n/a")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
