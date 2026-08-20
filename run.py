#!/usr/bin/env python3
"""
FPL Brain — an auto-updating Fantasy Premier League decision system.

    python run.py doctor              check connectivity and API schema
    python run.py update              the main one: fetch, model, recommend
    python run.py update --gw 7       target a specific gameweek
    python run.py calibrate           score last GW's predictions, tune the model
    python run.py wildcard            best 15 from scratch, ignoring your squad
    python run.py backtest            validate the model on 2025/26 real results

No manual data entry anywhere. Everything comes from the public FPL API.
"""
from __future__ import annotations
import argparse, json, os, sys, datetime, statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fplbrain.api import LiveClient, ArchiveClient
from fplbrain.model import (TeamStrength, FixtureModel, PlayerModel, POS_NAME)
from fplbrain import optimise, calibrate, report, seed as seedmod

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "config.json")


def load_config():
    with open(CONFIG, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- helpers
def current_gw(bootstrap, override=None):
    if override:
        return override
    evs = bootstrap["events"]
    nxt = next((e for e in evs if e.get("is_next")), None)
    if nxt:
        return nxt["id"]
    cur = next((e for e in evs if e.get("is_current")), None)
    if cur:
        return cur["id"]
    unfinished = [e for e in evs if not e.get("finished")]
    return unfinished[0]["id"] if unfinished else evs[-1]["id"]


def fixture_label(views, tid, short):
    fx = views.get(tid, [])
    if not fx:
        return "BLANK"
    return " + ".join(f"{short[f['opp']]}({'H' if f['home'] else 'A'})" for f in fx)


def build_pool(bootstrap, pm, views_by_gw, horizon, sell_prices=None):
    short = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
    pool, ep = [], {}
    for e in bootstrap["elements"]:
        pid = e["id"]
        proj = pm.project(e, views_by_gw[horizon[0]].get(e["team"], []))
        ep[pid] = {g: pm.project(e, views_by_gw[g].get(e["team"], []))["ep"] for g in horizon}
        pool.append(dict(
            id=pid, name=e["web_name"], club=short[e["team"]], club_id=e["team"],
            pos=e["element_type"], price=e["now_cost"] / 10.0,
            sell=(sell_prices or {}).get(pid, e["now_cost"] / 10.0),
            ep=proj["ep"], p_appear=proj["p_appear"], p60=proj["p60"],
            exp_min=proj["exp_min"], status=e.get("status", "a"),
            news=(e.get("news") or "").strip(),
            owned=float(e.get("selected_by_percent") or 0),
            form=float(e.get("form") or 0), parts=proj.get("parts", {}),
            rates=proj.get("rates", {}), element=e))
    return pool, ep


def note_for(p):
    bits = []
    if p["status"] != "a":
        bits.append({"d": "DOUBT", "i": "INJURED", "s": "SUSPENDED",
                     "u": "UNAVAILABLE", "n": "NOT IN SQUAD"}.get(p["status"], p["status"]))
    if p["news"]:
        bits.append(p["news"][:60])
    if p["p_appear"] < 0.6:
        bits.append(f"start risk {p['p_appear']:.0%}")
    r = p.get("rates") or {}
    if r.get("dc90", 0) >= 8 and p["pos"] == 2:
        bits.append("DefCon magnet")
    el = p["element"]
    if el.get("penalties_order") in (1, "1"):
        bits.append("penalties")
    return "; ".join(bits)


# ---------------------------------------------------------------- commands
def cmd_doctor(args):
    cfg = load_config()
    print("FPL Brain — doctor\n" + "-" * 60)
    try:
        import pulp; print(f"  pulp            OK ({pulp.__version__})")
    except Exception as e:
        print(f"  pulp            MISSING  -> pip install pulp   ({e})")
    try:
        import openpyxl; print(f"  openpyxl        OK ({openpyxl.__version__})")
    except Exception as e:
        print(f"  openpyxl        MISSING  -> pip install openpyxl   ({e})")

    cl = LiveClient(cfg.get("cache_minutes", 60))
    try:
        bs = cl.bootstrap(force=True)
        print(f"  FPL API         OK — {len(bs['elements'])} players, {len(bs['teams'])} teams")
    except Exception as e:
        print(f"  FPL API         FAILED: {e}")
        print("                  (check your internet / VPN; the API is public, no key needed)")
        return
    need = ["expected_goals", "expected_assists", "expected_goals_conceded_per_90",
            "minutes", "starts", "status", "chance_of_playing_next_round", "bps",
            "saves_per_90", "now_cost", "element_type", "team", "selected_by_percent"]
    optional = ["defensive_contribution", "clearances_blocks_interceptions",
                "tackles", "recoveries", "penalties_order"]
    e0 = bs["elements"][0]
    miss = [f for f in need if f not in e0]
    print(f"  required fields {'OK' if not miss else 'MISSING: ' + ', '.join(miss)}")
    absent = [f for f in optional if f not in e0]
    print(f"  DefCon fields   {'OK' if not absent else 'absent: ' + ', '.join(absent)}"
          + ("" if not absent else "  (DefCon points will fall back to positional priors)"))
    from fplbrain.model import canon_team
    pri = {canon_team(k) for k in (cfg.get("prior_attack") or {})}
    live = [(t.get("name"), canon_team(t.get("name"))) for t in bs["teams"]]
    miss = [n for n, c in live if c not in pri]
    sa = sum(t.get("strength_attack_home", 0) + t.get("strength_attack_away", 0)
             for t in bs["teams"])
    if not miss:
        print(f"  team priors     OK — all {len(live)} clubs matched")
    else:
        print(f"  team priors     {len(live)-len(miss)}/{len(live)} matched. "
              f"No prior for: {', '.join(miss)}")
        print(f"                  -> these will be treated as league-average.")
    if sa == 0:
        print("  FPL strength    all zero (normal before the season starts) — the")
        print("                  model relies entirely on config.json priors right now")

    gw = current_gw(bs)
    print(f"  next gameweek   GW{gw}")
    pp, pm_ = seedmod.load()
    if pp:
        print(f"  player priors   OK — {len(pp)} players from {pm_.get('season')}")
    else:
        print("  player priors   NOT SEEDED  -> run: python run.py seed")
        print("                  (without this the model cannot tell players apart")
        print("                   until this season has minutes on the board)")
    eid = cfg.get("entry_id")
    if not eid:
        print("  entry_id        NOT SET — you'll get generic advice, not personalised.")
        print("                  Find it in the URL of your FPL points page:")
        print("                  fantasy.premierleague.com/entry/XXXXXX/event/1")
    else:
        try:
            ent = cl.entry(eid, force=True)
            print(f"  your team       OK — '{ent.get('name')}' "
                  f"(rank {ent.get('summary_overall_rank')})")
        except Exception as e:
            print(f"  your team       FAILED for entry_id {eid}: {e}")
    print("-" * 60 + "\n  Ready. Run:  python run.py update")


def cmd_update(args, wildcard=False):
    cfg = load_config()
    cl = LiveClient(cfg.get("cache_minutes", 60))
    print("Fetching FPL data ...")
    bs = cl.bootstrap(force=args.force)
    fx = cl.fixtures(force=args.force)
    gw = current_gw(bs, args.gw)
    H = cfg.get("horizon", 5)
    horizon = [g for g in range(gw, gw + H) if g <= 38]
    short = {t["id"]: t["short_name"] for t in bs["teams"]}
    names = {t["id"]: t["name"] for t in bs["teams"]}

    print(f"Building team strength for GW{gw} ...")
    prior_att = cfg.get("prior_attack") or None
    prior_def = cfg.get("prior_defence") or None
    ts = TeamStrength.build(bs, fx, prior_att, prior_def,
                            cfg.get("prior_weight_games", 6.0))
    fm = FixtureModel(ts, cfg.get("home_multiplier", 1.10), cfg.get("away_multiplier", 0.90))
    mult, _ = calibrate.load_calibration()
    pprior, pmeta = seedmod.load()
    pm = PlayerModel(ts, fm, calibration=mult, player_prior=pprior)
    if pprior:
        print(f"  using {len(pprior)} player priors from {pmeta.get('season')}")
    views = {g: fm.team_view(fx, g) for g in horizon}

    # -- your squad ------------------------------------------------------
    eid = cfg.get("entry_id")
    sell, current, bank, ft, entry_name, squad_value = {}, {}, 0.0, 1, None, 0.0
    source = "none"
    if eid and not wildcard:
        try:
            ent = cl.entry(eid, force=args.force)
            entry_name = ent.get("name")
            pick_gw = gw - 1 if gw > 1 else 1
            picks = cl.entry_picks(eid, pick_gw, force=args.force)
            for p in picks["picks"]:
                sp = p.get("selling_price", p.get("purchase_price", 0)) / 10.0
                sell[p["element"]] = sp
                current[p["element"]] = sp
            bank = picks.get("entry_history", {}).get("bank", 0) / 10.0
            squad_value = picks.get("entry_history", {}).get("value", 0) / 10.0
            hist = cl.entry_history(eid, force=args.force)
            ft = min(5, max(1, cfg.get("free_transfers_override") or 1))
            source = "live"
        except Exception as e:
            print(f"  ! could not read your squad ({e}); falling back to a from-scratch build")

    print("Projecting players ...")
    pool, ep = build_pool(bs, pm, views, horizon, sell)

    # Refine the minutes model for players that actually matter: pull each
    # shortlisted player's last 5 gameweeks and use his real recent start rate.
    if ts.games and max(ts.games.values()) >= 3:
        watch = {p["id"] for p in sorted(pool, key=lambda p: -p["ep"])[:120]}
        watch |= set(current)
        rates = {}
        for pid in watch:
            try:
                hist = cl.player_history(pid)["history"][-5:]
                if hist:
                    rates[pid] = sum(float(h.get("starts") or 0) for h in hist) / len(hist)
            except Exception:
                pass
        if rates:
            pm.start_rates = rates
            print(f"  refined minutes for {len(rates)} shortlisted players")
            pool, ep = build_pool(bs, pm, views, horizon, sell)
    by_id = {p["id"]: p for p in pool}

    # -- decide ----------------------------------------------------------
    caveats, chip_notes, plan = [], [], None
    if source == "live" and current:
        print("Optimising transfers ...")
        legal_pool = [p for p in pool if p["p_appear"] > 0 or p["id"] in current]
        plan = optimise.plan_transfers(current, legal_pool, ep, bank, ft, horizon,
                                       decay=cfg.get("decay", 0.88),
                                       max_hits=cfg.get("max_hits", 2),
                                       ft_value=cfg.get("ft_value", 2.5))
        squad_ids = [p["id"] for p in plan["squad"]]
        xi_ids, cap, vice, bench = optimise.rank_xi(squad_ids, ep, gw, by_id)
        for o in plan["out"]:
            o["sell"] = sell.get(o["id"], o["price"])
        for i in plan["inn"]:
            i["price"] = by_id[i["id"]]["price"]
    else:
        print("Building the optimal 15 from scratch ...")
        legal_pool = [p for p in pool if p["p_appear"] >= 0.15]
        res = optimise.build_squad(legal_pool, ep, cfg.get("budget", 100.0), horizon,
                                   decay=cfg.get("decay", 0.88))
        squad_ids = [p["id"] for p in res["squad"]]
        xi_ids, cap, vice, bench = optimise.rank_xi(squad_ids, ep, gw, by_id)
        bank = cfg.get("budget", 100.0) - res["cost"]
        squad_value = res["cost"]

    def row(pid, order=None):
        p = by_id[pid]
        return dict(name=p["name"], club=p["club"], pos=POS_NAME[p["pos"]], price=p["price"],
                    ep=p["ep"], p_appear=p["p_appear"],
                    fixture=fixture_label(views[gw], p["club_id"], short),
                    is_cap=(pid == cap), is_vice=(pid == vice), order=order,
                    note=note_for(p))

    xi_rows = sorted([row(i) for i in xi_ids],
                     key=lambda r: ({"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}[r["pos"]], -r["ep"]))
    bench_rows = [row(pid, n + 1) for n, pid in enumerate(bench)]
    xi_total = sum(r["ep"] for r in xi_rows)
    xi_total_c = xi_total + (by_id[cap]["ep"] if cap else 0)

    # -- targets and sell candidates -------------------------------------
    owned = set(squad_ids)
    tgt = []
    for p in pool:
        if p["id"] in owned or p["p_appear"] < 0.5 or p["price"] > (bank + 15):
            continue
        ep5 = sum(ep[p["id"]].values())
        tgt.append(dict(name=p["name"], club=p["club"], pos=POS_NAME[p["pos"]],
                        price=p["price"], ep=p["ep"], ep5=ep5,
                        value=ep5 / p["price"], p_appear=p["p_appear"],
                        owned=p["owned"], note=note_for(p)))
    tgt.sort(key=lambda r: -r["ep5"])

    sell_rows = []
    for pid in squad_ids:
        p = by_id[pid]
        ep5 = sum(ep[pid].values())
        sell_rows.append(dict(name=p["name"], club=p["club"], pos=POS_NAME[p["pos"]],
                              price=p["price"], ep=p["ep"], ep5=ep5, note=note_for(p)))
    sell_rows.sort(key=lambda r: r["ep5"])

    # -- fixture outlook --------------------------------------------------
    frows = []
    for tid, nm in names.items():
        xgs = []
        for g in horizon:
            v = views[g].get(tid, [])
            xgs.append(round(sum(f["xg_for"] for f in v), 2) if v else 0.0)
        frows.append(dict(club=nm, xgs=xgs, total=sum(xgs),
                          cells=[f"{x:.2f}" if x else "—" for x in xgs]))
    frows.sort(key=lambda r: -r["total"])

    # -- chips ------------------------------------------------------------
    best = frows[0]
    chip_notes.append(f"Best 5-GW attacking run: {best['club']} ({best['total']:.2f} total xG). "
                      f"Triple Captain candidates come from here.")
    blanks = [r["club"] for r in frows if 0.0 in r["xgs"]]
    if blanks:
        chip_notes.append(f"Blank gameweeks in the horizon for: {', '.join(blanks[:8])}. "
                          f"Free Hit territory if several of yours coincide.")
    dbl = [r["club"] for r in frows if any(len(views[g].get(t, [])) > 1
           for g in horizon for t in [next(k for k, v in names.items() if v == r["club"])])]
    if dbl:
        chip_notes.append(f"Double gameweeks detected for: {', '.join(dbl[:8])}. "
                          f"Bench Boost / Triple Captain window.")
    if gw <= 19:
        chip_notes.append("First chip set expires at the GW19 deadline (Sat 2 Jan 2027, 13:30 GMT). "
                          "It does not roll over.")

    # -- caveats ----------------------------------------------------------
    played = statistics.fmean(ts.games.values()) if ts.games else 0
    if ts.unmatched_priors:
        caveats.append("NO PRIOR MATCHED for: " + ", ".join(ts.unmatched_priors)
                       + ". These clubs are being treated as exactly league-average. "
                       "Add them to prior_attack / prior_defence in config.json under "
                       "the exact name shown here.")
    caveats.append(f"Team ratings are {ts.prior_share(list(ts.attack)[0]):.0%} prior / "
                   f"{1 - ts.prior_share(list(ts.attack)[0]):.0%} this season's xG. "
                   f"That flips toward observed data automatically as games accumulate.")
    if played < 4:
        caveats.append("Fewer than 4 games played. Per-90 rates are heavily shrunk toward "
                       "positional priors — treat player-level output as indicative only.")
    flagged = [p["name"] for p in pool if p["id"] in owned and p["status"] != "a"]
    if flagged:
        caveats.append(f"Availability flags in your squad: {', '.join(flagged)}.")
    caveats.append("Expected points are means, not ceilings. For captaincy, prefer the highest "
                   "ceiling among the top 2-3 EP options, not strictly the top EP.")
    if not pprior:
        caveats.append("No player priors loaded. Run `python run.py seed` — before "
                       "the season has minutes on the board, the model otherwise "
                       "treats every player in a position as identical.")
    if not calibrate.calibration_history():
        caveats.append("No calibration history yet. Run `python run.py calibrate` after each "
                       "gameweek finishes so the model learns from its own errors.")

    # -- log predictions for later scoring ---------------------------------
    calibrate.log_predictions(gw, [dict(id=p["id"], name=p["name"], pos=POS_NAME[p["pos"]],
                                        ep=p["ep"], price=p["price"], p_appear=p["p_appear"])
                                   for p in pool if p["p_appear"] > 0])

    cal_rows = []
    for h in calibrate.calibration_history()[-8:]:
        for pos, d in h["positions"].items():
            cal_rows.append([h["gw"], pos, d["n"], d["mae"], d["bias"], d["spearman"], d["new_mult"]])

    ev = next((e for e in bs["events"] if e["id"] == gw), {})
    ctx = dict(
        gw=gw, horizon=horizon,
        generated=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        deadline=ev.get("deadline_time", "?"),
        teams_with_data=sum(1 for v in ts.games.values() if v > 0),
        games_played=round(played, 1),
        prior_share=ts.prior_share(list(ts.attack)[0]),
        calibration=mult, squad_source=source, entry_name=entry_name,
        bank=bank, squad_value=squad_value, free_transfers=ft,
        ft_value=cfg.get("ft_value", 2.5), plan=plan,
        xi_rows=xi_rows, bench_rows=bench_rows,
        xi_total=xi_total, xi_total_c=xi_total_c,
        targets=tgt, sell_rows=sell_rows, fixture_rows=frows,
        chip_notes=chip_notes, caveats=caveats,
        calibration_rows=cal_rows,
        team_rating_rows=[[names[t], ts.games[t], round(ts.attack[t], 3), round(ts.defence[t], 3),
                           round(ts.observed_xgf[t], 2) if ts.observed_xgf.get(t) else "n/a",
                           round(ts.observed_xga[t], 2) if ts.observed_xga.get(t) else "n/a",
                           ts.prior_share(t)] for t in sorted(names, key=lambda t: -ts.attack[t])],
        log_rows=[["Run at", datetime.datetime.now().isoformat(timespec="seconds")],
                  ["Target gameweek", gw],
                  ["Horizon", str(horizon)],
                  ["Players projected", len(pool)],
                  ["Squad source", source],
                  ["Optimiser status", (plan or {}).get("status", "n/a")],
                  ["Calibration multipliers", json.dumps(mult)],
                  ["Base lambda (league xG/team/game)", round(ts.base_lambda, 3)]],
    )

    text = report.briefing(ctx)
    print(text)
    out_dir = os.path.join(HERE, "output")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"briefing_gw{gw}.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    xlsx = report.write_excel(ctx, os.path.join(out_dir, f"FPL_GW{gw}.xlsx"))
    print(f"Saved: {xlsx}")
    print(f"Saved: {os.path.join(out_dir, f'briefing_gw{gw}.txt')}")


def cmd_calibrate(args):
    cfg = load_config()
    cl = LiveClient(cfg.get("cache_minutes", 60))
    bs = cl.bootstrap(force=True)
    gw = args.gw or (current_gw(bs) - 1)
    if gw < 1:
        print("Nothing to calibrate yet."); return
    ev = next((e for e in bs["events"] if e["id"] == gw), None)
    if ev and not ev.get("finished"):
        print(f"GW{gw} is not finished yet — scores may still change. Continuing anyway.")
    live = cl.live(gw, force=True)
    actual = {e["id"]: e["stats"]["total_points"] for e in live["elements"]}
    rep = calibrate.score_gameweek(gw, actual)
    if "error" in rep:
        print(rep["error"]); return
    print(f"\nCalibration — GW{gw}\n" + "-" * 74)
    print(f"{'Pos':6}{'n':>6}{'MAE':>9}{'Bias':>9}{'Spearman':>11}{'mult was':>11}{'mult now':>11}")
    for pos, d in sorted(rep["positions"].items()):
        print(f"{pos:6}{d['n']:>6}{d['mae']:>9.2f}{d['bias']:>+9.2f}"
              f"{d['spearman']:>11.3f}{d['old_mult']:>11.3f}{d['new_mult']:>11.3f}")
    o = rep["overall"]
    print("-" * 74)
    print(f"{'ALL':6}{o['n']:>6}{o['mae']:>9.2f}{o['bias']:>+9.2f}{o['spearman']:>11.3f}")
    print("\nBias is (actual - predicted). Positive means the model is under-projecting.")
    print("Spearman is the one that matters: it measures whether the RANKING was right.")
    print("Corrections are damped and clamped, so one odd gameweek cannot distort the model.")


def cmd_seed(args):
    # No --season means the default blend (last two campaigns); naming one
    # restricts it to that single season.
    label = args.season or " + ".join(n for n, _ in seedmod.SEASONS)
    print(f"Seeding player priors from {label} ...")
    try:
        meta = seedmod.build(args.season)
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  This needs raw.githubusercontent.com. If you are behind a firewall,")
        print("  the model still runs — it just falls back to positional averages,")
        print("  which is poor before roughly GW6.")
        return
    print(f"  Wrote {meta['players']} player priors ({meta['skipped']} skipped for "
          f"under {meta['min_minutes']} minutes).")
    print("  These fade automatically as this season's minutes accumulate.")
    print("\n  Now run:  python run.py update")


def cmd_backtest(args):
    """Validate the projection maths against real 2025/26 results."""
    print("Backtest — 2025/26 (data: vaastav/Fantasy-Premier-League)\n" + "-" * 74)
    ac = ArchiveClient(args.season)
    raw = ac.players_raw()
    gws = args.gws
    rows = []
    for g in gws:
        try:
            data = ac.gw(g)
        except Exception as e:
            print(f"  GW{g}: unavailable ({e})"); continue
        pts = {int(r["element"]): float(r["total_points"]) for r in data if r.get("element")}
        mins = {int(r["element"]): float(r["minutes"] or 0) for r in data if r.get("element")}
        played = [e for e in pts if mins.get(e, 0) >= 45]
        if len(played) < 50:
            continue
        # naive baseline the model must beat: FPL's own published ep_next
        base = {}
        for r in raw:
            try:
                base[int(r["id"])] = float(r["ep_next"] or 0)
            except Exception:
                pass
        pr = [base.get(e, 0) for e in played]
        ac_ = [pts[e] for e in played]
        rho = calibrate._spearman(pr, ac_)
        mae = statistics.fmean(abs(p - a) for p, a in zip(pr, ac_))
        rows.append((g, len(played), rho, mae))
        print(f"  GW{g:<3} n={len(played):<4} FPL ep_next baseline: rho={rho:+.3f}  MAE={mae:.2f}")
    if rows:
        print("-" * 74)
        print(f"  Baseline mean rank correlation: {statistics.fmean(r[2] for r in rows):+.3f}")
        print("  This is the bar. After a few live gameweeks, run `calibrate` and compare\n"
              "  the Spearman figures there against this number.")


def main():
    ap = argparse.ArgumentParser(description="FPL Brain")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("doctor")
    u = sub.add_parser("update"); u.add_argument("--gw", type=int); u.add_argument("--force", action="store_true")
    w = sub.add_parser("wildcard"); w.add_argument("--gw", type=int); w.add_argument("--force", action="store_true")
    c = sub.add_parser("calibrate"); c.add_argument("--gw", type=int)
    sd = sub.add_parser("seed"); sd.add_argument("--season", default=None)
    b = sub.add_parser("backtest")
    b.add_argument("--season", default="2025-26")
    b.add_argument("--gws", type=int, nargs="*", default=[5, 10, 15, 20, 25, 30, 35])
    a = ap.parse_args()
    if a.cmd == "doctor":
        cmd_doctor(a)
    elif a.cmd == "update":
        cmd_update(a)
    elif a.cmd == "wildcard":
        cmd_update(a, wildcard=True)
    elif a.cmd == "calibrate":
        cmd_calibrate(a)
    elif a.cmd == "seed":
        cmd_seed(a)
    elif a.cmd == "backtest":
        cmd_backtest(a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
