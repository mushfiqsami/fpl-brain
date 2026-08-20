#!/usr/bin/env python3
"""
Does the simulator still agree with the analytic model?

CLAUDE.md states this as an invariant and says to re-run it after any change to
the model - but shipped nothing that does it, so it was only ever checked by
hand. This is that check.

The two are built from the same fitted components, so the mean of the simulated
distribution should land on the analytic expected points. When they drift apart
one of them is wrong, and there is no way to tell which from the dashboard: both
produce confident numbers either way.

The per-position split is the useful part. A bias that shows up in DEF and MID
but not GK points at something the outfield positions share and keepers do not -
which is how the DefCon double-discount was found.

    python agreement_test.py            # normal run
    python agreement_test.py --runs 6000 --tolerance 0.25
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fplbrain.api import LiveClient
from fplbrain.model import (TeamStrength, FixtureModel, PlayerModel, POS_NAME,
                            penalty_uplift, setpiece_uplift)
from fplbrain.sim import PlayerSim
from fplbrain import seed as seedmod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3000)
    ap.add_argument("--tolerance", type=float, default=0.30,
                    help="fail if the mean gap exceeds this")
    ap.add_argument("--worst", type=int, default=10)
    args = ap.parse_args()

    cfg = {}
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        pass

    cl = LiveClient(cfg.get("cache_minutes", 60))
    bs, fx = cl.bootstrap(), cl.fixtures()
    pprior, pmeta = seedmod.load()
    ts = TeamStrength.build(bs, fx, cfg.get("prior_attack"), cfg.get("prior_defence"),
                            float(cfg.get("prior_weight_games", 6.0)))
    fm = FixtureModel(ts, float(cfg.get("home_multiplier", 1.10)),
                      float(cfg.get("away_multiplier", 0.90)))
    pm = PlayerModel(ts, fm, player_prior=pprior)
    evs = bs["events"]
    gw = (next((e["id"] for e in evs if e.get("is_next")), None)
          or next((e["id"] for e in evs if e.get("is_current")), None) or 1)
    views = fm.team_view(fx, gw)

    print(f"GW{gw} · {args.runs} sims/player · baselines: "
          f"{len(pprior)} players ({pmeta.get('season', '?')})")

    rows = []
    for e in bs["elements"]:
        fxs = views.get(e["team"], [])
        if not fxs:
            continue
        proj = pm.project(e, fxs)
        if proj["ep"] <= 0:
            continue
        p_app, p60, exp_min = pm.minutes_profile(e)
        r = pm.rates(e)
        start_rate = min(1.0, p60 / 0.9) if p60 > 0 else 0.0
        ps = min(p_app, start_rate)
        sim = PlayerSim(pos=e["element_type"], rates=r, p_start=ps,
                        p_sub=max(0.0, p_app - ps),
                        avg_start_mins=max(45.0, min(90.0, exp_min / max(0.05, ps))),
                        fixtures=fxs, base_lambda=ts.base_lambda,
                        pen_share=penalty_uplift(e), sp_share=setpiece_uplift(e))
        d = sim.run(n=args.runs, seed=e["id"])
        rows.append((e["web_name"], POS_NAME[e["element_type"]],
                     proj["ep"], d["mean"], d["mean"] - proj["ep"]))

    if not rows:
        print("no players to compare")
        return 1

    diffs = [r[4] for r in rows]
    mean = sum(diffs) / len(diffs)
    worst = max(abs(d) for d in diffs)
    over = sum(1 for d in diffs if abs(d) > 1.0)

    print(f"\ncompared      : {len(rows)} players")
    print(f"mean gap      : {mean:+.3f}   (sim minus analytic)")
    print(f"largest gap   : {worst:.3f}")
    print(f"gaps over 1.0 : {over}")

    print("\nby position:")
    by = {}
    for _, pos, _, _, d in rows:
        by.setdefault(pos, []).append(d)
    for pos in ("GK", "DEF", "MID", "FWD"):
        v = by.get(pos)
        if v:
            print(f"  {pos:<4} {sum(v)/len(v):+.3f}   (n={len(v)})")
    print("  a bias in the outfield positions but not GK points at something only\n"
          "  they share - defensive contribution, most likely")

    rows.sort(key=lambda r: -abs(r[4]))
    print(f"\nworst {args.worst}:")
    print(f"  {'player':<18}{'pos':<5}{'analytic':>9}{'sim':>8}{'gap':>8}")
    for n, pos, ep, sm, d in rows[:args.worst]:
        print(f"  {n[:17]:<18}{pos:<5}{ep:>9.2f}{sm:>8.2f}{d:>+8.2f}")

    ok = abs(mean) <= args.tolerance and over == 0
    print("\n" + ("PASS" if ok else "FAIL")
          + f" - mean gap within {args.tolerance}, no player over 1.0"
          if ok else
          "\nFAIL - the simulator and the analytic model disagree; one of them is wrong")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
