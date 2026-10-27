#!/usr/bin/env python3
"""
How much should the model lean on past seasons?

archive_ab_test.py builds PlayerModel() with NO player prior, so it can say nothing
about this question - the prior simply is not there. This harness replays 2025-26
with a prior built ONLY from the two seasons before it (2024-25 at weight 1.0,
2023-24 at 0.55 - the same recency shape production uses one year later), then
sweeps the knobs that decide prior-versus-current:

  K  prior_minutes_weight   pseudo-minutes of credit the prior gets on per-90 rates
  B  PRIOR_BLEND_GAMES      games of credit the archive start-rate gets vs this season's starts

and scores each setting on what actually happened, walk-forward (strictly the
games before the one being predicted):

  rho    correlation of projected vs actual points, players who played
  top30  mean actual points of the 30 highest projections (ranking at the top,
         which is where transfers are decided)
  squad  real points of the best legal 100m squad built from the projections

    python prior_weight_test.py
"""
from __future__ import annotations
import os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fplbrain.api import ArchiveClient
from fplbrain import model as M, optimise, seed
from fplbrain.model import TeamStrength, FixtureModel, PlayerModel
from backtest import build_state, f

SEASON = "2025-26"
TEST_GWS = (3, 5, 7, 9, 12, 16, 22, 28, 34)
CONFIGS = [  # (label, K, B, use_prior)
    ("no prior at all",        None, 4.0, False),
    ("K=700 B=4  (current)",   700.0, 4.0, True),
    ("K=450 B=4",              450.0, 4.0, True),
    ("K=300 B=4",              300.0, 4.0, True),
    ("K=150 B=4",              150.0, 4.0, True),
    ("K=700 B=2",              700.0, 2.0, True),
    ("K=300 B=2",              300.0, 2.0, True),
    ("K=150 B=2",              150.0, 2.0, True),
    ("K=300 B=1",              300.0, 1.0, True),
    ("K=150 B=1",              150.0, 1.0, True),
]


def main():
    ac = ArchiveClient(SEASON)
    code_of = {int(r["id"]): r["code"] for r in ac.players_raw()}
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
                team_h_difficulty=3, team_a_difficulty=3))
        except Exception:
            continue

    print("Building the pre-season prior from 2024-25 + 2023-24 only...")
    prior = seed._blend([(seed._fetch("2024-25")[0], 1.0), (seed._fetch("2023-24")[0], 0.55)])
    print(f"  {len(prior)} players carry a prior\n")

    cache = {}
    res = {c[0]: dict(rho=[], top=[], real=[], rho_early=[], top_early=[]) for c in CONFIGS}

    for gw in TEST_GWS:
        state = build_state(ac, gw - 1, teams_by_name, cache)
        if not state:
            continue
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
        if gw not in cache:
            cache[gw] = ac.gw(gw)
        actual = {int(r["element"]): f(r, "total_points") for r in cache[gw]}
        played = {int(r["element"]) for r in cache[gw] if f(r, "minutes") >= 1}

        for label, K, B, use_prior in CONFIGS:
            M.PRIOR_BLEND_GAMES = B
            pm = PlayerModel(ts, fm, player_prior=(prior if use_prior else {}))
            if K is not None:
                pm.prior_minutes_weight = K
            pm.calibrate_depth(list(state.values()))
            pool, ep, pairs = [], {}, []
            for eid, e in state.items():
                p = pm.project(e, views.get(e["team"], []))
                if p["ep"] > 0:
                    pool.append(dict(name=e["web_name"], club_id=e["team"], pos=e["element_type"],
                                     price=e["now_cost"] / 10.0, id=eid))
                    ep[eid] = {gw: p["ep"]}
                if eid in played and eid in actual:
                    pairs.append((p["ep"], actual[eid], eid))
            rho = statistics.correlation([a for a, _, _ in pairs], [b for _, b, _ in pairs])
            top = statistics.fmean(b for _, b, _ in sorted(pairs, key=lambda t: -t[0])[:30])
            real = None
            try:
                built = optimise.build_squad(pool, ep, 100.0, [gw], decay=1.0)
                if built["status"] == "Optimal":
                    ids = [q["id"] for q in built["squad"]]
                    byid = {q["id"]: q for q in pool}
                    xi, cap, _v, _b, _d = optimise.rank_xi(ids, ep, gw, byid)
                    real = sum(actual.get(i, 0) for i in xi) + actual.get(cap, 0)
            except Exception as exc:
                print(f"   squad build failed {label} GW{gw}: {exc}")
            r = res[label]
            r["rho"].append(rho); r["top"].append(top)
            if real is not None:
                r["real"].append(real)
            if gw <= 12:
                r["rho_early"].append(rho); r["top_early"].append(top)
        print(f"  GW{gw} done", flush=True)

    print("\n" + "=" * 92)
    print(f"{'config':26}{'rho':>8}{'top30':>8}{'squad':>8}   {'rho GW<=12':>11}{'top30 GW<=12':>13}")
    print("=" * 92)
    for label, *_ in CONFIGS:
        r = res[label]
        print(f"{label:26}{statistics.fmean(r['rho']):8.4f}{statistics.fmean(r['top']):8.2f}"
              f"{statistics.fmean(r['real']) if r['real'] else float('nan'):8.1f}"
              f"   {statistics.fmean(r['rho_early']):11.4f}{statistics.fmean(r['top_early']):13.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
