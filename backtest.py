#!/usr/bin/env python3
"""
Walk-forward validation.

For each test gameweek N:
  1. Rebuild the exact league state as it stood after GW N-1 (cumulative minutes,
     xG, xA, BPS, defensive actions, results) from the real 2025/26 gameweek files.
  2. Run the full model on that state -- no peeking at GW N.
  3. Compare projected points against what actually happened in GW N.

This is the honest test: the model only ever sees the past.
"""
import sys, os, statistics, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fplbrain.api import ArchiveClient
from fplbrain.model import TeamStrength, FixtureModel, PlayerModel, POS_NAME
from fplbrain import optimise, calibrate

POS_ID = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
NUMERIC = ["minutes", "starts", "expected_goals", "expected_assists", "bps", "saves",
           "clearances_blocks_interceptions", "tackles", "recoveries", "total_points",
           "goals_conceded", "expected_goals_conceded"]


def f(row, key, d=0.0):
    try:
        return float(row.get(key) or 0)
    except (TypeError, ValueError):
        return d


def build_state(ac, upto_gw, teams_by_name, cache):
    """Cumulative per-player state after gameweeks 1..upto_gw."""
    agg = {}
    for g in range(1, upto_gw + 1):
        if g not in cache:
            cache[g] = ac.gw(g)
        for r in cache[g]:
            eid = int(r["element"])
            a = agg.setdefault(eid, dict(
                id=eid, web_name=r["name"], element_type=POS_ID.get(r["position"], 3),
                team=teams_by_name.get(r["team"]), now_cost=f(r, "value"),
                status="a", chance_of_playing_next_round=None, news="",
                selected_by_percent=0.0, penalties_order=None, form=0.0,
                **{k: 0.0 for k in NUMERIC}))
            a["now_cost"] = f(r, "value")
            for k in NUMERIC:
                a[k] += f(r, k)
    for a in agg.values():
        m = a["minutes"]
        a["expected_goals_conceded_per_90"] = (a["expected_goals_conceded"] / m * 90) if m else 0.0
        a["saves_per_90"] = (a["saves"] / m * 90) if m else 0.0
        a["defensive_contribution"] = (a["clearances_blocks_interceptions"]
                                       + a["tackles"] + a["recoveries"])
    return agg


def main(test_gws, season="2025-26"):
    ac = ArchiveClient(season)
    teams = ac.teams()
    teams_by_name = {t["name"]: int(t["id"]) for t in teams}
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

    boot_teams = [dict(id=int(t["id"]), name=t["name"], short_name=t["short_name"],
                       strength_attack_home=int(t["strength_attack_home"]),
                       strength_attack_away=int(t["strength_attack_away"]),
                       strength_defence_home=int(t["strength_defence_home"]),
                       strength_defence_away=int(t["strength_defence_away"])) for t in teams]

    cache = {}
    results, baseline_results = [], []
    print(f"Walk-forward backtest — {season}")
    print("=" * 88)
    print(f"{'GW':>4}{'n':>6}{'model rho':>12}{'model MAE':>12}{'naive rho':>12}"
          f"{'naive MAE':>12}{'top10 hit':>11}{'mean EP':>10}")
    print("-" * 88)

    for gw in test_gws:
        state = build_state(ac, gw - 1, teams_by_name, cache)
        if not state:
            continue
        # only fixtures up to gw-1 count as "played"
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
        act_min = {int(r["element"]): f(r, "minutes") for r in cache[gw]}

        # recent-form start rate: last 5 gameweeks, as the live system computes it
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

        preds, acts, naive = [], [], []
        prices = []
        for eid, e in state.items():
            if eid not in actual:
                continue
            if act_min.get(eid, 0) < 1:          # score only players who featured
                continue
            proj = pm.project(e, views.get(e["team"], []))
            preds.append(proj["ep"])
            acts.append(actual[eid])
            prices.append(e["now_cost"] / 10.0)
            # naive baseline: points per game so far
            gplayed = max(1, ts.games.get(e["team"], 1))
            naive.append(e["total_points"] / gplayed)

        if len(preds) < 60:
            continue
        rho = calibrate._spearman(preds, acts)
        mae = statistics.fmean(abs(p - a) for p, a in zip(preds, acts))
        nrho = calibrate._spearman(naive, acts)
        nmae = statistics.fmean(abs(p - a) for p, a in zip(naive, acts))

        # did the model's top 10 actually outscore the field average?
        order = sorted(range(len(preds)), key=lambda i: -preds[i])[:10]
        top10 = statistics.fmean(acts[i] for i in order)
        norder = sorted(range(len(naive)), key=lambda i: -naive[i])[:10]
        ntop10 = statistics.fmean(acts[i] for i in norder)
        field = statistics.fmean(acts)

        # restricted to the realistic decision set: players you would actually consider
        sel = [i for i in range(len(preds)) if prices[i] >= 5.0]
        rho_sel = calibrate._spearman([preds[i] for i in sel], [acts[i] for i in sel]) if len(sel) > 40 else 0
        nrho_sel = calibrate._spearman([naive[i] for i in sel], [acts[i] for i in sel]) if len(sel) > 40 else 0

        results.append((gw, len(preds), rho, mae, nrho, nmae, top10, field, ntop10, rho_sel, nrho_sel))
        print(f"{gw:>4}{len(preds):>6}{rho:>12.3f}{mae:>12.2f}{nrho:>12.3f}"
              f"{nmae:>12.2f}{top10:>11.2f}{statistics.fmean(preds):>10.2f}")

    if not results:
        print("no usable gameweeks"); return
    print("-" * 88)
    mr = statistics.fmean(r[2] for r in results)
    mm = statistics.fmean(r[3] for r in results)
    nr = statistics.fmean(r[4] for r in results)
    nm = statistics.fmean(r[5] for r in results)
    t10 = statistics.fmean(r[6] for r in results)
    fld = statistics.fmean(r[7] for r in results)
    nt10 = statistics.fmean(r[8] for r in results)
    rs = statistics.fmean(r[9] for r in results)
    nrs = statistics.fmean(r[10] for r in results)
    print(f"{'MEAN':>4}{'':>6}{mr:>12.3f}{mm:>12.2f}{nr:>12.3f}{nm:>12.2f}{t10:>11.2f}")
    print()
    print(f"  Rank correlation, all players   model {mr:+.3f}   naive {nr:+.3f}")
    print(f"  Rank correlation, >=5.0m only   model {rs:+.3f}   naive {nrs:+.3f}   "
          f"<- the actual decision set")
    print(f"  Mean absolute error             model {mm:.2f}     naive {nm:.2f}")
    print(f"  Top-10 picks scored             model {t10:.2f}     naive {nt10:.2f}     "
          f"field {fld:.2f}")
    print(f"  Model top-10 vs field           {t10/fld:.2f}x   (naive {nt10/fld:.2f}x)")
    print()
    print("  Rank correlation is the number that matters. FPL is a selection problem:")
    print("  you need the ordering right, not the absolute total.")


if __name__ == "__main__":
    gws = [int(x) for x in sys.argv[1:]] or [8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
    main(gws)
