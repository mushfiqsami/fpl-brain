#!/usr/bin/env python3
"""
Model A/B harness - live current-season data only.

Walk forward through every real gameweek played so far THIS season. For each one,
build every candidate model's projections from only the gameweeks strictly before
it (no peeking), then score two ways:

  1. Player-level: correlation and MAE of projected points vs what every player
     who featured actually scored. Answers "does this ranking track reality".
  2. Squad-level: build the best legal £100m squad from projections alone, score
     it on real results (season_test.py's method). Answers "would this version
     actually have scored more points running your team".

This is how a change like prior_minutes_weight gets adopted or rejected - checked
against this season's own real results, not argued from feel or carried over
unexamined from last season's backtest. Add a candidate to CANDIDATES below and
rerun; every additional real gameweek played makes the comparison stronger.

Needs live_gw<N>.json for every gameweek up to and including the current one, plus
bs_tmp.json / fx_tmp.json (fresh bootstrap-static / fixtures). Fetch with:

    curl -s "https://fantasy.premierleague.com/api/bootstrap-static/" -o bs_tmp.json
    curl -s "https://fantasy.premierleague.com/api/fixtures/" -o fx_tmp.json
    for g in 1 2 3 4; do
      curl -s "https://fantasy.premierleague.com/api/event/$g/live/" -o "live_gw$g.json"
    done

    python model_ab_test.py
"""
from __future__ import annotations
import json, os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fplbrain.model import TeamStrength, FixtureModel, PlayerModel
from fplbrain import seed as seedmod, optimise
from fplbrain.model_v2 import project_v2, momentum as v2_momentum, MOMENTUM_WEIGHT, REL_MOMENTUM_WEIGHT
from fplbrain import momentum as momentummod

HERE = os.path.dirname(os.path.abspath(__file__))

# Every candidate to test. "params" are attributes set on the PlayerModel instance
# after construction - anything on the class can be swept this way without forking
# the model into two files. V1 is the config that actually scored your real GW1-4
# points; add more entries here as new ideas come up.
CANDIDATES = [
    dict(name="V1 (production, prior_minutes_weight=700)", params=dict(prior_minutes_weight=700.0)),
    dict(name="V2 (prior_minutes_weight=150)",              params=dict(prior_minutes_weight=150.0)),
]


def load_json(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)


def build_cumulative_state(bs, through_gw):
    """Real per-player totals from gameweeks 1..through_gw, live data only."""
    byid = {e["id"]: e for e in bs["elements"]}
    agg = {}
    for g in range(1, through_gw + 1):
        live = load_json(f"live_gw{g}.json")
        for el in live["elements"]:
            eid = int(el["id"])
            s = el["stats"]
            a = agg.setdefault(eid, dict(
                minutes=0.0, expected_goals=0.0, expected_assists=0.0, bps=0.0,
                clearances_blocks_interceptions=0.0, tackles=0.0, recoveries=0.0,
                saves=0.0, expected_goals_conceded=0.0))
            a["minutes"] += s.get("minutes", 0)
            a["expected_goals"] += float(s.get("expected_goals", 0) or 0)
            a["expected_assists"] += float(s.get("expected_assists", 0) or 0)
            a["bps"] += s.get("bps", 0)
            a["clearances_blocks_interceptions"] += s.get("clearances_blocks_interceptions", 0)
            a["tackles"] += s.get("tackles", 0)
            a["recoveries"] += s.get("recoveries", 0)
            a["saves"] += s.get("saves", 0)
            a["expected_goals_conceded"] += float(s.get("expected_goals_conceded", 0) or 0)

    state_elements = []
    for eid, a in agg.items():
        base = byid.get(eid)
        if not base:
            continue
        e = dict(base)
        e.update(a)
        e["saves_per_90"] = (a["saves"] / a["minutes"] * 90) if a["minutes"] else 0.0
        state_elements.append(e)
    return state_elements


def fixtures_known_before(fx, gw):
    out = []
    for x in fx:
        y = dict(x)
        if y.get("event") and y["event"] >= gw:
            y["finished"] = False
            y["team_h_score"] = None
            y["team_a_score"] = None
        out.append(y)
    return out


def evaluate(cfg, bs, fx, cfg_json, prior, test_gw):
    """One candidate, one gameweek. Returns player-level and squad-level results."""
    state_elements = build_cumulative_state(bs, test_gw - 1)
    fx_known = fixtures_known_before(fx, test_gw)
    bootstrap = dict(teams=bs["teams"], elements=state_elements, events=[])

    ts = TeamStrength.build(bootstrap, fx_known, cfg_json.get("prior_attack"),
                             cfg_json.get("prior_defence"), 6.0)
    fm = FixtureModel(ts, float(cfg_json["home_multiplier"]), float(cfg_json["away_multiplier"]))
    views = fm.team_view(fx, test_gw)

    pm = PlayerModel(ts, fm, player_prior=prior, live_is_last_season=False)
    for k, v in cfg["params"].items():
        setattr(pm, k, v)
    pm.calibrate_depth(state_elements)

    live_actual = load_json(f"live_gw{test_gw}.json")
    actual = {int(e["id"]): e["stats"]["total_points"] for e in live_actual["elements"]}
    mins = {int(e["id"]): e["stats"]["minutes"] for e in live_actual["elements"]}

    pairs = []
    pool, ep = [], {}
    teams = {t["id"]: t["short_name"] for t in bs["teams"]}
    for e in state_elements:
        eid = e["id"]
        v = views.get(e["team"], [])
        proj = pm.project(e, v)
        if eid in actual and mins.get(eid, 0) >= 1:
            pairs.append((proj["ep"], actual[eid]))
        p_appear = proj["p_appear"]
        if p_appear >= 0.35 and (e.get("status", "a") == "a"):
            ep[eid] = {test_gw: proj["ep"]}
            pool.append(dict(id=eid, name=e.get("web_name", str(eid)), club=teams.get(e["team"], "?"),
                             club_id=e["team"], pos=e["element_type"],
                             price=e["now_cost"] / 10.0))

    rho = statistics.correlation([p[0] for p in pairs], [p[1] for p in pairs]) if len(pairs) > 2 else None
    mae = sum(abs(x - y) for x, y in pairs) / len(pairs) if pairs else None

    real_total = None
    proj_total = None
    try:
        built = optimise.build_squad(pool, ep, 100.0, [test_gw], decay=1.0)
        ids = [p["id"] for p in built["squad"]]
        by_id = {p["id"]: p for p in pool}
        xi, cap, _v, _b, _d = optimise.rank_xi(ids, ep, test_gw, by_id)
        real_total = sum(actual.get(i, 0) for i in xi) + actual.get(cap, 0)
        proj_total = sum(ep[i][test_gw] for i in xi) + ep.get(cap, {}).get(test_gw, 0.0)
    except Exception as exc:
        print(f"    (squad build failed for GW{test_gw}: {exc})")

    return dict(n=len(pairs), rho=rho, mae=mae, real_squad_points=real_total,
                proj_squad_points=proj_total)


def evaluate_hybrid(bs, fx, cfg_json, prior, test_gw):
    """
    V1's process-stat + Dixon-Coles engine (the strongest single piece measured
    so far: 0.288 correlation, the backbone) with V2's ownership-weighted
    transfer momentum grafted on as a small secondary multiplier - the one real
    signal V1 structurally cannot see at all. Not a guess: both pieces already
    won their own individual tests in this file: V1 as the CANDIDATES baseline,
    momentum's ownership-weighting in fplbrain/model_v2.py's own history.
    """
    state_elements = build_cumulative_state(bs, test_gw - 1)
    fx_known = fixtures_known_before(fx, test_gw)
    bootstrap = dict(teams=bs["teams"], elements=state_elements, events=[])

    ts = TeamStrength.build(bootstrap, fx_known, cfg_json.get("prior_attack"),
                             cfg_json.get("prior_defence"), 6.0)
    fm = FixtureModel(ts, float(cfg_json["home_multiplier"]), float(cfg_json["away_multiplier"]))
    views = fm.team_view(fx, test_gw)

    pm = PlayerModel(ts, fm, player_prior=prior, live_is_last_season=False)
    pm.calibrate_depth(state_elements)

    live_actual = load_json(f"live_gw{test_gw}.json")
    actual = {int(e["id"]): e["stats"]["total_points"] for e in live_actual["elements"]}
    mins = {int(e["id"]): e["stats"]["minutes"] for e in live_actual["elements"]}
    total_players = bs.get("total_players")
    snap = momentummod.for_gw(test_gw)

    pairs = []
    pool, ep = [], {}
    teams = {t["id"]: t["short_name"] for t in bs["teams"]}
    for e in state_elements:
        eid = e["id"]
        v = views.get(e["team"], [])
        proj = pm.project(e, v)
        t_in, t_out = snap.get(eid, (0, 0))
        mom_abs, mom_rel = v2_momentum(t_in, t_out, total_players, e.get("selected_by_percent"))
        hybrid_ep = proj["ep"] * (1.0 + MOMENTUM_WEIGHT * mom_abs + REL_MOMENTUM_WEIGHT * mom_rel)
        if eid in actual and mins.get(eid, 0) >= 1:
            pairs.append((hybrid_ep, actual[eid]))
        p_appear = proj["p_appear"]
        if p_appear >= 0.35 and (e.get("status", "a") == "a"):
            ep[eid] = {test_gw: hybrid_ep}
            pool.append(dict(id=eid, name=e.get("web_name", str(eid)), club=teams.get(e["team"], "?"),
                             club_id=e["team"], pos=e["element_type"],
                             price=e["now_cost"] / 10.0))

    rho = statistics.correlation([p[0] for p in pairs], [p[1] for p in pairs]) if len(pairs) > 2 else None
    mae = sum(abs(x - y) for x, y in pairs) / len(pairs) if pairs else None

    real_total = proj_total = None
    try:
        built = optimise.build_squad(pool, ep, 100.0, [test_gw], decay=1.0)
        ids = [p["id"] for p in built["squad"]]
        by_id = {p["id"]: p for p in pool}
        xi, cap, _v, _b, _d = optimise.rank_xi(ids, ep, test_gw, by_id)
        real_total = sum(actual.get(i, 0) for i in xi) + actual.get(cap, 0)
        proj_total = sum(ep[i][test_gw] for i in xi) + ep.get(cap, {}).get(test_gw, 0.0)
    except Exception as exc:
        print(f"    (hybrid squad build failed for GW{test_gw}: {exc})")

    return dict(n=len(pairs), rho=rho, mae=mae, real_squad_points=real_total,
                proj_squad_points=proj_total)


def evaluate_v2(bs, fx, test_gw):
    """
    model_v2's own evaluation path - deliberately not routed through PlayerModel
    at all, since V2's whole point is to not share V1's machinery. Builds each
    player's own gameweek-by-gameweek history from strictly-prior real data,
    the fixture difficulty for test_gw, and this week's live transfer counts.
    """
    byid = {e["id"]: e for e in bs["elements"]}
    total_players = bs.get("total_players")

    # this player's own games, oldest first, through test_gw - 1
    history = {eid: [] for eid in byid}
    for g in range(1, test_gw):
        live = load_json(f"live_gw{g}.json")
        seen = set()
        for el in live["elements"]:
            eid = int(el["id"])
            seen.add(eid)
            history.setdefault(eid, []).append(dict(minutes=el["stats"].get("minutes", 0),
                                                      total_points=el["stats"].get("total_points", 0),
                                                      xgi=el["stats"].get("expected_goal_involvements", 0),
                                                      cs=el["stats"].get("clean_sheets", 0),
                                                      dc=el["stats"].get("defensive_contribution", 0)))
        # players who didn't feature that week still get a 0-minute entry, so a
        # dropped player's recency weighting sees the benching, not silence
        for eid in byid:
            if eid not in seen:
                history.setdefault(eid, []).append(dict(minutes=0, total_points=0, xgi=0, cs=0, dc=0))

    # fixture difficulty per team for test_gw
    diff = {}
    for f in fx:
        if f.get("event") == test_gw:
            diff[f["team_h"]] = f.get("team_h_difficulty", 3)
            diff[f["team_a"]] = f.get("team_a_difficulty", 3)

    live_actual = load_json(f"live_gw{test_gw}.json")
    actual = {int(e["id"]): e["stats"]["total_points"] for e in live_actual["elements"]}
    mins = {int(e["id"]): e["stats"]["minutes"] for e in live_actual["elements"]}

    # real recorded transfer counts for THIS gameweek, or {} if it predates
    # fplbrain.momentum being wired into run_job() - in that case every player
    # falls back to neutral (0, 0) rather than leaking today's live numbers
    # into a backtest of a gameweek that already happened.
    snap = momentummod.for_gw(test_gw)
    if not snap:
        print(f"    (no recorded transfer snapshot for GW{test_gw} - momentum term is neutral)")

    pairs = []
    pool, ep = [], {}
    teams = {t["id"]: t["short_name"] for t in bs["teams"]}
    for eid, e in byid.items():
        fd = diff.get(e["team"], 3)
        transfers = snap.get(eid, (0, 0))
        r = project_v2(e, history.get(eid, []), fd, total_players, transfers=transfers)
        if eid in actual and mins.get(eid, 0) >= 1:
            pairs.append((r["ep"], actual[eid]))
        if r["minutes_share"] >= 0.35 and e.get("status", "a") == "a":
            ep[eid] = {test_gw: r["ep"]}
            pool.append(dict(id=eid, name=e.get("web_name", str(eid)), club=teams.get(e["team"], "?"),
                             club_id=e["team"], pos=e["element_type"], price=e["now_cost"] / 10.0))

    rho = statistics.correlation([p[0] for p in pairs], [p[1] for p in pairs]) if len(pairs) > 2 else None
    mae = sum(abs(x - y) for x, y in pairs) / len(pairs) if pairs else None

    real_total = proj_total = None
    try:
        built = optimise.build_squad(pool, ep, 100.0, [test_gw], decay=1.0)
        ids = [p["id"] for p in built["squad"]]
        by_id = {p["id"]: p for p in pool}
        xi, cap, _v, _b, _d = optimise.rank_xi(ids, ep, test_gw, by_id)
        real_total = sum(actual.get(i, 0) for i in xi) + actual.get(cap, 0)
        proj_total = sum(ep[i][test_gw] for i in xi) + ep.get(cap, {}).get(test_gw, 0.0)
    except Exception as exc:
        print(f"    (V2 squad build failed for GW{test_gw}: {exc})")

    return dict(n=len(pairs), rho=rho, mae=mae, real_squad_points=real_total,
                proj_squad_points=proj_total)


def main():
    bs = load_json("bs_tmp.json")
    fx = load_json("fx_tmp.json")
    cfg_json = load_json("config.json")
    prior, _ = seedmod.load()

    events = bs["events"]
    played = [e["id"] for e in events if e.get("finished")]
    # test gameweeks 2..N: each predicted from strictly-prior real data
    test_gws = [g for g in played if g >= 2]
    if not test_gws:
        print("Need at least 2 finished gameweeks to test anything. Nothing to do.")
        return 1

    print(f"Testing gameweeks {test_gws} (each predicted from the real gameweeks before it)\n")

    names = ([c["name"] for c in CANDIDATES]
             + ["V2 (current-season-only, experimental)", "Hybrid (V1 + ownership-weighted momentum)"])
    totals = {n: dict(rho=[], mae=[], real=[], proj=[]) for n in names}
    for gw in test_gws:
        print(f"=== GW{gw} ===")
        for c in CANDIDATES:
            r = evaluate(c, bs, fx, cfg_json, prior, gw)
            rho_s = f"{r['rho']:.4f}" if r["rho"] is not None else "n/a"
            mae_s = f"{r['mae']:.3f}" if r["mae"] is not None else "n/a"
            print(f"  {c['name']:42} n={r['n']:3d}  rho={rho_s}  MAE={mae_s}  "
                  f"best-squad real pts={r['real_squad_points']}  (projected {r['proj_squad_points']})")
            if r["rho"] is not None:
                totals[c["name"]]["rho"].append(r["rho"])
            if r["mae"] is not None:
                totals[c["name"]]["mae"].append(r["mae"])
            if r["real_squad_points"] is not None:
                totals[c["name"]]["real"].append(r["real_squad_points"])
                totals[c["name"]]["proj"].append(r["proj_squad_points"])

        r = evaluate_v2(bs, fx, gw)
        name = "V2 (current-season-only, experimental)"
        rho_s = f"{r['rho']:.4f}" if r["rho"] is not None else "n/a"
        mae_s = f"{r['mae']:.3f}" if r["mae"] is not None else "n/a"
        print(f"  {name:42} n={r['n']:3d}  rho={rho_s}  MAE={mae_s}  "
              f"best-squad real pts={r['real_squad_points']}  (projected {r['proj_squad_points']})")
        if r["rho"] is not None:
            totals[name]["rho"].append(r["rho"])
        if r["mae"] is not None:
            totals[name]["mae"].append(r["mae"])
        if r["real_squad_points"] is not None:
            totals[name]["real"].append(r["real_squad_points"])
            totals[name]["proj"].append(r["proj_squad_points"])

        r = evaluate_hybrid(bs, fx, cfg_json, prior, gw)
        name = "Hybrid (V1 + ownership-weighted momentum)"
        rho_s = f"{r['rho']:.4f}" if r["rho"] is not None else "n/a"
        mae_s = f"{r['mae']:.3f}" if r["mae"] is not None else "n/a"
        print(f"  {name:42} n={r['n']:3d}  rho={rho_s}  MAE={mae_s}  "
              f"best-squad real pts={r['real_squad_points']}  (projected {r['proj_squad_points']})")
        if r["rho"] is not None:
            totals[name]["rho"].append(r["rho"])
        if r["mae"] is not None:
            totals[name]["mae"].append(r["mae"])
        if r["real_squad_points"] is not None:
            totals[name]["real"].append(r["real_squad_points"])
            totals[name]["proj"].append(r["proj_squad_points"])
        print()

    print("=" * 78)
    print("AGGREGATE ACROSS ALL TESTED GAMEWEEKS")
    print("=" * 78)
    for name in names:
        t = totals[name]
        rho_m = statistics.fmean(t["rho"]) if t["rho"] else None
        mae_m = statistics.fmean(t["mae"]) if t["mae"] else None
        real_m = statistics.fmean(t["real"]) if t["real"] else None
        proj_m = statistics.fmean(t["proj"]) if t["proj"] else None
        print(f"{name}")
        print(f"  mean correlation      : {rho_m:.4f}" if rho_m is not None else "  mean correlation      : n/a")
        print(f"  mean MAE              : {mae_m:.3f}" if mae_m is not None else "  mean MAE              : n/a")
        print(f"  mean best-squad points: {real_m:.1f} real vs {proj_m:.1f} projected"
              if real_m is not None else "  mean best-squad points: n/a")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
