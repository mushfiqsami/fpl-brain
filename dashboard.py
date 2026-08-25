#!/usr/bin/env python3
"""
FPL Brain — local web dashboard.

Double-click START.bat (Windows) or run `python dashboard.py`.
Your browser opens automatically. No terminal use required.

Why a local server instead of a plain .html file: the FPL API does not send CORS
headers, so a browser cannot fetch it directly from a file:// page. The request is
made here, in Python, and the result is served to your browser from localhost.
"""
import datetime, hashlib, json, os, random, sys, threading, time, traceback, webbrowser, socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CONFIG = os.path.join(HERE, "config.json")

# Stamped by build_standalone.py. Shown in the header so it is possible to tell,
# from the page itself, whether the thing you are looking at is the thing that
# was last published - a stale service worker or a deploy that never finished
# otherwise look exactly like a model that is ignoring you.
BUILD = "dev"

# ---------------------------------------------------------------------------
# ZeroGPU compatibility.
#
# ZeroGPU Spaces refuse to start unless the module exposes a function decorated
# with @spaces.GPU, and that check runs while the module is being imported. A
# registration inside a function is too late, which is why this sits at the top
# level.
#
# This app is pure CPU maths and wants no GPU. "CPU basic" is free and is the
# correct hardware. The placeholder exists only so that a Space created on
# ZeroGPU boots instead of crash-looping.
# ---------------------------------------------------------------------------
try:
    import spaces as _spaces

    @_spaces.GPU(duration=1)
    def zerogpu_placeholder():
        """Unused. Present so the ZeroGPU startup check passes."""
        return "ok"

    _ZEROGPU = True
except Exception:
    _ZEROGPU = False

STATE = {"status": "idle", "message": "", "data": None, "error": None,
         "updated": None, "log": [], "by_team": {}}
LOCK = threading.Lock()


def log(msg):
    with LOCK:
        STATE["message"] = msg
        STATE["log"].append(msg)
        STATE["log"] = STATE["log"][-40:]
    print("  " + msg, flush=True)


# Filled in by build_standalone.py. config.json is not one of the files published
# to the host, so the bundled app had no measured team priors at all - and FPL
# zeroes its own strength ratings before a season starts, so there was nothing to
# fall back to either. Every club came out exactly league average, which flattens
# fixture difficulty completely and quietly costs a striker at a strong club
# around two points a week. Same trap as the player baselines, same fix: carry it
# inside the bundle rather than hoping the file is there.
EMBEDDED_CONFIG = {}


def load_config():
    """Embedded defaults, overlaid with config.json when one exists."""
    cfg = dict(EMBEDDED_CONFIG)
    try:
        with open(CONFIG, encoding="utf-8") as f:
            cfg.update(json.load(f))
    except Exception:
        pass
    return cfg


def save_config(cfg):
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# FPL will not autosub into an illegal shape: three at the back and one up top
# are the floors, and there is always exactly one keeper.
XI_MIN_DEF = 3
XI_MIN_FWD = 1


def _horizon_total(squad_ids, ep, gws, by_id):
    """Best XI plus captain, summed over the horizon, for a given 15."""
    from fplbrain import optimise
    total = 0.0
    for g in gws:
        try:
            xi, cap, _v, _b, _d = optimise.rank_xi(squad_ids, ep, g, by_id)
        except Exception:
            continue
        total += sum(ep[i].get(g, 0.0) for i in xi)
        if cap in ep:
            total += ep[cap].get(g, 0.0)
    return round(total, 2)


# ---------------------------------------------------------------------------
# What part of the season are we in?
#
# The same two fields on a live element mean completely different things either
# side of the first deadline. Before it, `minutes` and `starts` are last
# season's finished totals. The moment FPL rolls over they reset to zero and
# start counting this season. Code that reads them has to know which it is
# looking at, and there is no flag in the API that says so - so it is worked out
# here, once, and handed down.
#
# There is also a third state, and it is the one that bites. Between a deadline
# and the results being confirmed, the counters have reset but almost nothing
# has been recorded: a man who has just played 67 minutes shows one start, out
# of a season that has barely begun. Treated as evidence that is a disaster -
# it read Saka as a 3% starter and dragged the whole projection down with him,
# which in turn told brain.py the target was hundreds of points out of reach.
# The honest answer during that window is that the gameweek has not told us
# anything reliable yet, so keep using what we knew going in.
#
# The primary signal is a recorded score, not FPL's `finished`/`data_checked`
# flags. Those usually agree, but this feed never once flipped them for GW1
# even after every one of its ten fixtures had a final score sitting in the
# data for days - so anything that waited on them stayed wrong indefinitely,
# not just for the few hours a deadline-based guess would have been off by.
# Scores are the actual event; the flags are FPL's bookkeeping about the event,
# and bookkeeping that never runs must not be allowed to block on.
#
# PHASE_OVERRIDES survives as a manual circuit-breaker for whichever gameweek
# needs one - a corrected score, a feed outage, a genuinely stuck flag on a
# gameweek this logic still gets wrong for some other reason - but it is no
# longer required reading week to week, because the score check re-derives the
# same answer on its own as each new gameweek is played.
PHASE_OVERRIDES = {
    # "underway": treat this gameweek as settled regardless of what the scores
    # or flags say. "preseason"/"settling": hold the season at that phase no
    # matter what has actually been played. Leave empty in the normal case.
}

PHASE_PRESEASON = "preseason"
PHASE_SETTLING = "settling"
PHASE_UNDERWAY = "underway"


def _utc(s):
    try:
        return datetime.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _gw_fully_scored(gw_id, fixtures):
    """Every fixture in this gameweek has a final score recorded.

    A postponement or a genuine in-progress match leaves one fixture without a
    score, and that correctly keeps the whole gameweek from reading as settled
    - a partial result is not evidence of the players who have not featured
    yet, and averaging it in early would be exactly the mistake this module
    exists to prevent.
    """
    rows = [f for f in fixtures if f.get("event") == gw_id]
    if not rows:
        return False
    return all(f.get("team_h_score") is not None and f.get("team_a_score") is not None
              for f in rows)


def season_phase(events, fixtures=(), now=None, overrides=PHASE_OVERRIDES):
    """
    ('preseason' | 'settling' | 'underway', human sentence).

      preseason  nothing has kicked off. Live counters are last season's, and
                 are the best guide to who plays where he is now.
      settling   a gameweek is in progress - kicked off, not every fixture
                 scored yet. Counters have reset and hold a game or less of
                 noise. Read nothing into them; keep last season's picture.
      underway   at least one gameweek is fully scored. Real data.
    """
    now = now or datetime.datetime.now(datetime.timezone.utc)
    forced = (overrides or {}).get("phase")
    if forced in (PHASE_PRESEASON, PHASE_SETTLING, PHASE_UNDERWAY):
        return forced, f"Phase pinned to '{forced}' by PHASE_OVERRIDES."

    scored_ids = [e["id"] for e in events if _gw_fully_scored(e["id"], fixtures)]
    if scored_ids:
        last = max(scored_ids)
        return PHASE_UNDERWAY, f"GW{last} is fully scored - projections run on this season's data."

    started = any(e.get("is_current") for e in events)
    if not started:
        for e in events:
            d = _utc(e.get("deadline_time"))
            if d and now >= d:
                started = True
                break
    if not started:
        return PHASE_PRESEASON, "Season has not started - built from last season's record."

    return PHASE_SETTLING, ("A gameweek is under way with results still coming in, so the "
                            "model is holding last season's picture rather than reading a "
                            "part-played week as evidence.")


# ---------------------------------------------------------------------------
# One team, kept in sync with the official app instead of typed twice.
#
# The public FPL API exposes a team's picks with no login - only its numeric
# id, which is right there in the URL of its own points page. Setting that id
# once on a stored team was already how the SINGLE team named in config.json
# worked; this is the same mechanism made available per team, for however many
# of them you actually run, so a transfer made in the real app is what every
# later run here sees too, with nothing to re-type and nothing that can drift.
# ---------------------------------------------------------------------------
def resolve_team(t, elements, teams, cl, gw, force=False):
    """
    (resolved, problems, source) for one stored team.

    resolved is the same shape squad.resolve() returns, so every caller that
    already consumes that shape - the transfer planner, the team-card summary,
    the squad editor's confirmation - needs no separate code path for a team
    synced this way versus one typed by hand.

    Falls back to the typed players on any failure (before the first deadline,
    a network hiccup, a wrong id) rather than reporting the team as empty -
    a team with real typed players is still usable advice even when the live
    pull briefly is not.
    """
    from fplbrain import squad as squadmod
    eid = t.get("entry_id")
    if not eid:
        return (*squadmod.resolve(t.get("players") or [], elements, teams), "manual")
    try:
        ent = cl.entry(int(eid), force=force)
        picks = cl.entry_picks(int(eid), max(1, gw - 1), force=force)
        by_id = {e["id"]: e for e in elements}
        short = {tm["id"]: tm.get("short_name", "") for tm in teams}
        resolved = []
        for p in picks["picks"]:
            e = by_id.get(p["element"])
            if not e:
                continue
            # The public picks endpoint only carries selling_price/purchase_price
            # for the CURRENT gameweek's squad, not a past one - so a squad
            # pulled the week after its deadline (exactly when this is normally
            # called) gets neither field. Silently defaulting that to 0 told the
            # transfer planner every player was worth nothing sold, which priced
            # every transfer as unaffordable and made it recommend nothing at
            # all. now_cost is what the player actually sells for absent a
            # price change since the pick was made, which is the common case in
            # the first few gameweeks of a season and the only information
            # available without asking the account holder to log in.
            raw_sell = p.get("selling_price") or p.get("purchase_price")
            resolved.append(dict(
                name=e.get("web_name"), club=short.get(e["team"], ""), role=None,
                id=e["id"], price=e["now_cost"] / 10.0,
                sell=(raw_sell / 10.0) if raw_sell else e["now_cost"] / 10.0,
                matched_name=e.get("web_name"), matched_club=short.get(e["team"], ""),
                pos=e.get("element_type"), exact=True,
                live_entry_name=ent.get("name"),
                live_bank=picks.get("entry_history", {}).get("bank", 0) / 10.0,
                live_value=picks.get("entry_history", {}).get("value", 0) / 10.0,
                live_rank=ent.get("summary_overall_rank"),
                live_points=ent.get("summary_overall_points")))
        return resolved, [], "live"
    except Exception as exc:
        res, probs = squadmod.resolve(t.get("players") or [], elements, teams)
        if t.get("players"):
            probs = probs + [dict(
                name=None, club=None,
                reason=(f"Could not load the synced team ({exc}) - showing the last typed "
                        "copy instead. Before the first deadline of the season a team's "
                        "picks are not published yet; that is normal, not an error."))]
        else:
            probs = probs + [dict(
                name=None, club=None,
                reason=(f"Could not load the synced team ({exc}), and there is no typed "
                        "copy to fall back to."))]
        return res, probs, "manual"


def free_rein(events, gw):
    """
    True while the squad can still be rebuilt for free.

    FPL charges nothing for changes until the first deadline of the season, so
    before then there is no such thing as a hit and no reason to hoard a free
    transfer. After that deadline the normal economy starts and this is False
    for the rest of the season.
    """
    if gw != 1:
        return False
    ev = next((e for e in events if e.get("id") == 1), None)
    if not ev or ev.get("finished"):
        return False
    dl = ev.get("deadline_time")
    if not dl:
        return False
    try:
        # FPL stamps these as "2026-08-21T22:30:00Z"
        when = datetime.datetime.fromisoformat(str(dl).replace("Z", "+00:00"))
    except ValueError:
        return False
    return datetime.datetime.now(datetime.timezone.utc) < when


# =====================================================================
# COMPUTE
# =====================================================================
def compute(force=False, for_team=None):
    from fplbrain.api import LiveClient
    from fplbrain.model import TeamStrength, FixtureModel, PlayerModel, POS_NAME
    from fplbrain import optimise, calibrate, seed as seedmod, squad as squadmod
    from fplbrain.sim import PlayerSim, captain_value
    from fplbrain.model import penalty_uplift, setpiece_uplift

    cfg = load_config()
    cl = LiveClient(cfg.get("cache_minutes", 60))

    # --- auto-seed on first run, so the user never has to think about it ---
    pprior, pmeta = seedmod.load()
    if not pprior:
        log("First run — downloading past-season player data (one time only)...")
        try:
            # No season argument: take seed.SEASONS, which blends the last two.
            # Passing one explicitly silently halved the evidence behind every
            # per-90 rate and dropped 134 players out of the prior entirely.
            pmeta = seedmod.build()
            pprior, pmeta = seedmod.load()
            log(f"Loaded {len(pprior)} player baselines.")
        except Exception as e:
            log(f"Could not download baselines ({e}). Continuing with position averages.")

    log("Fetching live FPL data...")
    bs = cl.bootstrap(force=force)
    fx = cl.fixtures(force=force)

    evs = bs["events"]
    gw = (next((e["id"] for e in evs if e.get("is_next")), None)
          or next((e["id"] for e in evs if e.get("is_current")), None)
          or next((e["id"] for e in evs if not _gw_fully_scored(e["id"], fx)), evs[-1]["id"]))
    H = int(cfg.get("horizon", 5))
    horizon = [g for g in range(gw, gw + H) if g <= 38]
    # The Targets tab is about who to bring in for what is coming, so "next 5"
    # has to mean five gameweeks AFTER this one - the old figure silently
    # included the current week, which is the one you are already committed to.
    # Planning still runs on `horizon`; this is a wider lens for scouting only.
    look_ahead = [g for g in range(gw + 1, gw + 1 + H) if g <= 38]
    span = sorted(set(horizon) | set(look_ahead))
    short = {t["id"]: t["short_name"] for t in bs["teams"]}
    names = {t["id"]: t["name"] for t in bs["teams"]}

    log("Building team ratings...")
    ts = TeamStrength.build(bs, fx, cfg.get("prior_attack"), cfg.get("prior_defence"),
                            float(cfg.get("prior_weight_games", 6.0)))
    fm = FixtureModel(ts, float(cfg.get("home_multiplier", 1.10)),
                      float(cfg.get("away_multiplier", 0.90)))
    mult, _ = calibrate.load_calibration()
    # Which season the live minutes/starts belong to decides how they may be
    # read at all, so it is settled before the model is built.
    phase, phase_note = season_phase(evs, fx)
    log(phase_note)
    pm = PlayerModel(ts, fm, calibration=mult, player_prior=pprior,
                     live_is_last_season=(phase == PHASE_PRESEASON))
    # Only one keeper per club can play, so their start probabilities have to be
    # shared out rather than assessed one at a time. Without this a club's second
    # and third keepers project as starters too, and the cheap one looks like a
    # bargain first choice.
    pm.calibrate_depth(bs["elements"])
    views = {g: fm.team_view(fx, g) for g in span}

    # --- your squad -------------------------------------------------------
    sell, current, bank, ft, entry_name, squad_value = {}, {}, 0.0, 1, None, 0.0
    overall_rank = None
    live_points_so_far = None
    source = "none"
    squad_error = None
    squad_problems = []

    # 1st preference: a team with an entry_id syncs from the official FPL API;
    # otherwise fall back to whatever was typed into My Squad. FPL does not
    # publish a team's picks until the first deadline, so before that manual
    # entry is the only way to get real advice regardless of which this is.
    all_teams = squadmod.load_all()
    mysq = squadmod.load_one(for_team) if for_team else squadmod.load()
    formation = "auto"
    force_start, force_bench = set(), set()
    # What the server actually resolved each row to - live pick or typed row.
    # The squad editor used to answer that question with its own separate
    # matcher written in the page, which scores differently and knows nothing
    # about which players are already spoken for - so it could confirm one
    # player under a row while the model was scoring a different one entirely.
    # One resolver, and the editor shows its answer.
    squad_resolved = []
    if mysq:
        res, squad_problems, res_source = resolve_team(mysq, bs["elements"], bs["teams"], cl, gw, force=force)
        squad_resolved = [dict(slot=r.get("slot"), typed=r.get("name"),
                               typed_club=r.get("club"), id=r["id"],
                               name=r["matched_name"], club=r["matched_club"],
                               pos=POS_NAME[r["pos"]], price=round(r["price"], 1),
                               exact=bool(r.get("exact")))
                          for r in res[:15]]
        if len(res) >= 15:
            formation = mysq.get("formation") or "auto"
            for r in res[:15]:
                sell[r["id"]] = r["sell"]
                current[r["id"]] = r["sell"]
                if r.get("role") == "start":
                    force_start.add(r["id"])
                elif r.get("role") == "bench":
                    force_bench.add(r["id"])
            live_bit = res[0] if res_source == "live" else {}
            bank = float(live_bit.get("live_bank", mysq.get("bank", 0.0)))
            ft = (min(5, max(1, int(cfg.get("free_transfers_override") or 1)))
                  if res_source == "live" else int(mysq.get("free_transfers", 1)))
            squad_value = round(live_bit.get("live_value") or
                                sum(r["price"] for r in res[:15]), 1)
            entry_name = live_bit.get("live_entry_name") or mysq.get("name") or "My squad"
            source = res_source
            overall_rank = live_bit.get("live_rank")
            live_points_so_far = live_bit.get("live_points")
            log(f"{'Synced' if res_source == 'live' else 'Using'} '{entry_name}' — "
                f"{len(res)} players matched.")
        elif res:
            squad_error = (f"Only {len(res)} of 15 players matched. Open the My Squad "
                           "tab and fix the ones flagged in red.")
            log(squad_error)

    # Legacy fallback: a global entry_id in config.json, for anyone still using
    # the pre-per-team mechanism and whose active team has not been given its
    # own entry_id. New teams should set entry_id on the team itself instead.
    eid = cfg.get("entry_id")
    if eid and source == "none":
        try:
            log(f"Loading your team ({eid})...")
            ent = cl.entry(int(eid), force=force)
            entry_name = ent.get("name")
            overall_rank = ent.get("summary_overall_rank")
            picks = cl.entry_picks(int(eid), max(1, gw - 1), force=force)
            by_id_live = {e["id"]: e for e in bs["elements"]}
            for p in picks["picks"]:
                # See resolve_team()'s identical fallback: the picks endpoint
                # only carries a sell price for the current gameweek, so a squad
                # fetched the week after its deadline needs now_cost instead of
                # silently pricing every player at 0.
                raw_sell = p.get("selling_price") or p.get("purchase_price")
                e_live = by_id_live.get(p["element"])
                sp = (raw_sell / 10.0) if raw_sell else (
                    e_live["now_cost"] / 10.0 if e_live else 0.0)
                sell[p["element"]] = sp
                current[p["element"]] = sp
            eh = picks.get("entry_history", {})
            bank = eh.get("bank", 0) / 10.0
            squad_value = eh.get("value", 0) / 10.0
            ft = min(5, max(1, int(cfg.get("free_transfers_override") or 1)))
            source = "live"
            log(f"Loaded '{entry_name}' — {len(current)} players.")
        except Exception as e:
            squad_error = (f"Could not load team {eid}: {e}. "
                           "Before GW1 your squad isn't published yet — that's normal.")
            log(squad_error)

    log(f"Projecting {len(bs['elements'])} players...")
    pool, ep = [], {}
    for e in bs["elements"]:
        pid = e["id"]
        proj = pm.project(e, views[gw].get(e["team"], []))
        # gw_offset lets a knock or a ban wear off across the horizon instead of
        # being frozen at today's value for five straight gameweeks
        ep[pid] = {g: pm.project(e, views[g].get(e["team"], []),
                                 gw_offset=g - gw)["ep"] for g in span}
        pool.append(dict(id=pid, name=e["web_name"], club=short[e["team"]],
                         club_id=e["team"], pos=e["element_type"],
                         price=e["now_cost"] / 10.0,
                         sell=sell.get(pid, e["now_cost"] / 10.0),
                         ep=proj["ep"], p_appear=proj["p_appear"],
                         status=e.get("status", "a"), news=(e.get("news") or "").strip(),
                         owned=float(e.get("selected_by_percent") or 0),
                         element=e))
    by_id = {p["id"]: p for p in pool}

    # ---- Monte Carlo the players that matter --------------------------------
    # Simulating all 570 would be slow and pointless; only the plausible picks
    # need a distribution. Everything owned, plus the top EP by position.
    log("Simulating points distributions...")
    runs = int(cfg.get("sim_runs", 3000))
    shortlist = set(current)
    for want_pos in (1, 2, 3, 4):
        # Ranked over the whole span, not just this gameweek. Targets are listed
        # on the forward window, so ranking the shortlist on today alone left the
        # players that tab actually recommends without a simulated distribution -
        # they rendered with an empty range bar and "NaN%" for the haul chance.
        ranked = sorted((p for p in pool if p["pos"] == want_pos and p["p_appear"] >= 0.4),
                        key=lambda p: -sum(ep[p["id"]][g] for g in span))
        shortlist |= {p["id"] for p in ranked[:40]}

    dists = {}
    sim_by_id = {}          # kept so the XI can be simulated jointly, see below
    for pid in shortlist:
        p = by_id.get(pid)
        if not p:
            continue
        e = p["element"]
        prof = pm.minutes_profile(e)
        r = pm.rates(e)
        fxs = views[gw].get(p["club_id"], [])
        if not fxs or prof[0] <= 0:
            dists[pid] = dict(mean=0.0, median=0, p90=0, p10=0, sd=0.0,
                              p_haul=0.0, p_big=0.0, p_blank=1.0)
            continue
        # reuse the model's own decomposition rather than re-deriving it
        p_app, p60, exp_min = prof
        start_rate = min(1.0, p60 / 0.9) if p60 > 0 else 0.0
        p_start_est = min(p_app, start_rate)
        p_sub_est = max(0.0, p_app - p_start_est)
        sim = PlayerSim(pos=p["pos"], rates=r, p_start=p_start_est, p_sub=p_sub_est,
                        avg_start_mins=max(45.0, min(90.0, exp_min / max(0.05, p_start_est))),
                        fixtures=fxs, base_lambda=ts.base_lambda,
                        pen_share=penalty_uplift(e), sp_share=setpiece_uplift(e))
        d = sim.run(n=runs, seed=pid)
        d.pop("samples", None)
        dists[pid] = d
        sim_by_id[pid] = sim

    for p in pool:
        d = dists.get(p["id"])
        if d:
            p["ceiling"] = d["p90"]
            p["floor"] = d["p10"]
            p["p_haul"] = d["p_haul"]
            p["p_big"] = d["p_big"]
            p["p_blank"] = d["p_blank"]
            p["sd"] = d["sd"]
            p["sim_mean"] = d["mean"]
            p["cap_score"] = captain_value(d)
        else:
            p["ceiling"] = p["floor"] = 0
            p["p_haul"] = p["p_big"] = 0.0
            p["p_blank"] = 1.0
            p["sd"] = 0.0
            p["sim_mean"] = p["ep"]
            p["cap_score"] = p["ep"]

    # Until the very first deadline passes, FPL lets you rebuild the squad as
    # often as you like for nothing. Charging 4 a hit and capping the plan at
    # one free transfer invents a penalty that does not exist yet, and hides the
    # fact that the whole squad is still free to change.
    unlimited = free_rein(evs, gw)

    plan = None
    if source in ("live", "manual") and current:
        log("Planning the best squad you can still reach..." if unlimited
            else "Optimising transfers...")
        legal = [p for p in pool if p["p_appear"] > 0 or p["id"] in current]
        plan = optimise.plan_transfers(
            current, legal, ep, bank,
            15 if unlimited else ft, horizon,
            decay=float(cfg.get("decay", 0.88)),
            max_hits=0 if unlimited else int(cfg.get("max_hits", 2)),
            # Before the deadline a transfer is free, but "free" must not mean
            # "costless to recommend": with no charge at all the solver would
            # advise a swap worth a hundredth of a point and then, run again from
            # the new squad, advise swapping back. used_ft tracks the number of
            # moves, so a small value here is a per-transfer friction that leaves
            # real rebuilds alone and kills the churn.
            ft_value=(float(cfg.get("churn_cost", 0.25)) if unlimited
                      else float(cfg.get("ft_value", 2.5))))
        # Show the squad you actually own, not the one the optimiser would build.
        # plan["squad"] is post-transfer, and rendering it as "your XI" made the
        # app look like it had silently edited the team - incoming targets showed
        # up in the XI while players you still own disappeared. The transfers are
        # already spelled out separately in "the call"; this stays factual.
        squad_ids = list(current)
    else:
        log("Building the best squad from scratch...")
        legal = [p for p in pool if p["p_appear"] >= 0.15]
        res = optimise.build_squad(legal, ep, float(cfg.get("budget", 100.0)), horizon,
                                   decay=float(cfg.get("decay", 0.88)))
        squad_ids = [p["id"] for p in res["squad"]]
        bank = float(cfg.get("budget", 100.0)) - res["cost"]
        squad_value = res["cost"]

    # ------------------------------------------------------------------
    # The route: not just this week's move, but the sequence.
    #
    # Fixtures swing, and all 38 gameweeks are already known, so the useful
    # question is not "who is best now" but "when should I own whom". This lets
    # ownership vary per gameweek under real transfer rules and reports the path.
    # ------------------------------------------------------------------
    route = None
    if source in ("live", "manual") and current:
        try:
            log("Working out a transfer route...")
            r_pool = [p for p in pool if p["p_appear"] >= 0.35 or p["id"] in current]
            route = optimise.plan_route(
                current, r_pool, ep, bank, ft, span,
                decay=float(cfg.get("decay", 0.88)),
                free_first=unlimited, max_squad=90,
                time_limit=int(cfg.get("route_seconds", 25)))
        except Exception as exc:
            log(f"Route planning unavailable ({exc}).")
            route = None

    xi_ids, cap, vice, bench, xi_dropped = optimise.rank_xi(
        squad_ids, ep, gw, by_id, formation=formation,
        force_start=force_start, force_bench=force_bench)

    # Re-pick the captain on ceiling rather than mean. Doubling a score is a
    # convex payoff, so the right captain is the fattest tail among the credible
    # options, not strictly the highest average.
    cands = sorted((by_id[i] for i in xi_ids if by_id[i]["p_appear"] > 0.5),
                   key=lambda p: -p["ep"])[:5]
    if cands:
        best = max(cands, key=lambda p: p.get("cap_score", p["ep"]))
        rest = [c for c in cands if c["id"] != best["id"]]
        cap = best["id"]
        vice = max(rest, key=lambda p: p.get("cap_score", p["ep"]))["id"] if rest else None
    captain_table = [dict(name=c["name"], club=c["club"], ep=round(c["ep"], 2),
                          ceiling=c.get("ceiling", 0), floor=c.get("floor", 0),
                          p_haul=round(c.get("p_haul", 0), 3),
                          score=round(c.get("cap_score", c["ep"]), 2),
                          chosen=(c["id"] == cap)) for c in cands]

    # ------------------------------------------------------------------
    # What a good and a bad week actually look like.
    #
    # Percentiles do not add. Summing each player's own 90th percentile
    # describes the week where all eleven peak at once, which essentially never
    # happens - measured against a joint simulation that figure sat at the
    # 99.99th percentile, not the 90th, and the matching "bad week" was equally
    # unreachable below. Means are safe to add, which is why the projection was
    # right while the range around it was not.
    #
    # So simulate the XI together: one draw per player per iteration, summed.
    # The captain doubles the realisation he actually had, not a second
    # independent one, or the tail comes out too fat.
    # ------------------------------------------------------------------
    # Autosubs and the vice-captain are part of the score, so they are part of
    # the simulation.
    #
    # Ignoring them made the floor far darker than reality: a starter who does
    # not play was simply counted as zero, when FPL would have brought a bench
    # player on. That also made a cheap bench look free - the whole point of
    # paying for the bench is the weeks it rescues, and a model blind to
    # substitutions cannot see that value at all.
    #
    # The rules being followed here: a keeper can only be replaced by the
    # reserve keeper; outfielders come on in bench order and only if the XI is
    # still legal afterwards (3+ defenders, 1+ forward); and if the captain does
    # not appear the armband passes to the vice.
    log("Simulating the week...")
    _rng = random.Random(20260817)
    _pos_of = {i: by_id[i]["pos"] for i in squad_ids if i in by_id}
    _bench_gk = [i for i in bench if _pos_of.get(i) == 1]
    _bench_out = [i for i in bench if _pos_of.get(i) != 1]

    def _legal(counts):
        return (counts.get(1, 0) == 1 and counts.get(2, 0) >= XI_MIN_DEF
                and counts.get(4, 0) >= XI_MIN_FWD and sum(counts.values()) == 11)

    _totals = []
    for _ in range(runs):
        drawn = {}
        for pid in squad_ids:
            s = sim_by_id.get(pid)
            drawn[pid] = s.draw(_rng) if s else (0, False)

        playing, missing = [], []
        for pid in xi_ids:
            (playing if drawn[pid][1] else missing).append(pid)
        counts = {}
        for pid in playing:
            counts[_pos_of.get(pid, 3)] = counts.get(_pos_of.get(pid, 3), 0) + 1

        used = set()
        for pid in missing:
            pool_b = _bench_gk if _pos_of.get(pid) == 1 else _bench_out
            for b in pool_b:
                if b in used or not drawn[b][1]:
                    continue
                trial = dict(counts)
                bp = _pos_of.get(b, 3)
                trial[bp] = trial.get(bp, 0) + 1
                if _legal(trial):
                    used.add(b)
                    playing.append(b)
                    counts = trial
                    break

        t = sum(drawn[p][0] for p in playing)
        # armband follows the captain, or the vice if he did not appear
        arm = cap if (cap in drawn and drawn[cap][1]) else vice
        if arm in drawn and drawn[arm][1] and arm in playing:
            t += drawn[arm][0]
        _totals.append(t)
    _totals.sort()
    team_lo = _totals[int(runs * 0.10)]
    team_mid = _totals[runs // 2]
    team_hi = _totals[int(runs * 0.90)]

    def fixlabel(tid):
        v = views[gw].get(tid, [])
        if not v:
            return "BLANK"
        return " + ".join(f"{short[f['opp']]} ({'H' if f['home'] else 'A'})" for f in v)

    def flag(p):
        f = []
        if p["status"] != "a":
            f.append({"d": "Doubtful", "i": "Injured", "s": "Suspended",
                      "u": "Unavailable", "n": "Not in squad"}.get(p["status"], p["status"]))
        if p["p_appear"] < 0.6:
            f.append(f"Start risk {p['p_appear']:.0%}")
        if p["news"]:
            f.append(p["news"][:70])
        return f

    def prow(pid, order=None):
        p = by_id[pid]
        e = p["element"]
        return dict(id=pid, name=p["name"], club=p["club"], pos=POS_NAME[p["pos"]],
                    price=round(p["price"], 1), ep=round(p["ep"], 2),
                    ep5=round(sum(ep[pid][g] for g in horizon), 2),
                    p_appear=round(p["p_appear"], 3), fixture=fixlabel(p["club_id"]),
                    captain=(pid == cap), vice=(pid == vice), order=order,
                    owned=round(p["owned"], 1), flags=flag(p),
                    ceiling=p.get("ceiling", 0), floor=p.get("floor", 0),
                    p_haul=round(p.get("p_haul", 0), 3),
                    p_blank=round(p.get("p_blank", 0), 3),
                    sd=round(p.get("sd", 0), 2),
                    pens=bool(penalty_uplift(e) > 0),
                    setp=bool(setpiece_uplift(e) > 0),
                    form=round(p.get("form", 0), 1),
                    per_gw=[round(ep[pid][g], 2) for g in horizon])

    order_pos = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    xi = sorted([prow(i) for i in xi_ids], key=lambda r: (order_pos[r["pos"]], -r["ep"]))
    bench_rows = [prow(pid, n + 1) for n, pid in enumerate(bench)]
    owned_ids = set(squad_ids)

    targets = []
    for p in pool:
        if p["id"] in owned_ids or p["p_appear"] < 0.5:
            continue
        # ep5 is the scouting number: the five gameweeks AFTER this one, so a
        # player is judged on what he is about to walk into, not on a week you
        # have already committed to. ep_now stays available for a same-week call.
        e5 = sum(ep[p["id"]][g] for g in look_ahead)
        targets.append(dict(id=p["id"], name=p["name"], club=p["club"],
                            pos=POS_NAME[p["pos"]], price=round(p["price"], 1),
                            ep=round(p["ep"], 2), ep5=round(e5, 2),
                            ep_h=round(sum(ep[p["id"]][g] for g in horizon), 2),
                            value=round(e5 / max(0.1, p["price"]), 2),
                            # the tab renders a range bar and a haul chance from
                            # these; without them it drew empty bars and "NaN%"
                            ceiling=p.get("ceiling", 0), floor=p.get("floor", 0),
                            p_haul=round(p.get("p_haul", 0), 3),
                            sd=round(p.get("sd", 0), 2),
                            p_appear=round(p["p_appear"], 3),
                            owned=round(p["owned"], 1), flags=flag(p),
                            affordable=(p["price"] <= bank + 12)))
    targets.sort(key=lambda r: -r["ep5"])

    weak = sorted([prow(i) for i in squad_ids], key=lambda r: r["ep5"])[:6]

    # ------------------------------------------------------------------
    # Every saved team, summarised on the same basis.
    #
    # Running the full transfer optimiser for each would be slow, so each team
    # gets the cheap half: resolve the 15, pick the best legal XI and captain,
    # and total the projections. That is enough to compare them honestly,
    # because they are all scored by the identical model.
    # ------------------------------------------------------------------
    log("Summarising all teams...")
    team_cards = []
    all_sets = {}
    for t in all_teams["teams"]:
        res, probs, t_source = resolve_team(t, bs["elements"], bs["teams"], cl, gw, force=False)
        ids = [r["id"] for r in res][:15]
        all_sets[t["id"]] = set(ids)
        t_live = res[0] if (res and t_source == "live") else {}
        t_bank = t_live.get("live_bank", t.get("bank", 0.0))
        if len(ids) < 11:
            team_cards.append(dict(id=t["id"], name=t_live.get("live_entry_name") or t.get("name") or "Team",
                                   ok=False, matched=len(ids),
                                   problems=len(probs), value=0, bank=t_bank,
                                   free_transfers=t.get("free_transfers", 1)))
            continue
        try:
            # each team keeps its own shape and start/bench pins
            xi_i, cap_i, _vice, _bench, _drop = optimise.rank_xi(
                ids, ep, gw, by_id, formation=t.get("formation") or "auto",
                force_start={r["id"] for r in res[:15] if r.get("role") == "start"},
                force_bench={r["id"] for r in res[:15] if r.get("role") == "bench"})
        except Exception:
            xi_i, cap_i = ids[:11], ids[0]
        # captain on ceiling, matching the active team's logic
        cands_i = sorted((by_id[i] for i in xi_i if by_id[i]["p_appear"] > 0.5),
                         key=lambda p: -p["ep"])[:5]
        if cands_i:
            cap_i = max(cands_i, key=lambda p: p.get("cap_score", p["ep"]))["id"]
        xi_ep = sum(by_id[i]["ep"] for i in xi_i)
        cap_p = by_id.get(cap_i)
        team_cards.append(dict(
            id=t["id"], name=t_live.get("live_entry_name") or t.get("name") or "Team", ok=True,
            matched=len(ids), problems=len(probs),
            value=round(t_live.get("live_value") or sum(by_id[i]["price"] for i in ids), 1),
            bank=round(float(t_bank), 1),
            synced=(t_source == "live"),
            free_transfers=int(t.get("free_transfers", 1)),
            proj=round(xi_ep + (cap_p["ep"] if cap_p else 0), 2),
            floor=sum(by_id[i].get("floor", 0) for i in xi_i),
            ceiling=sum(by_id[i].get("ceiling", 0) for i in xi_i),
            captain=(cap_p["name"] if cap_p else "—"),
            captain_club=(cap_p["club"] if cap_p else ""),
            ep5=round(sum(sum(ep[i][g] for g in horizon) for i in xi_i), 1),
            risk=sum(1 for i in ids if by_id[i]["p_appear"] < 0.6),
            flagged=[by_id[i]["name"] for i in ids if by_id[i]["status"] != "a"][:4],
            active=(t["id"] == all_teams.get("active")),
            xi=[dict(name=by_id[i]["name"], club=by_id[i]["club"],
                     pos=POS_NAME[by_id[i]["pos"]], ep=round(by_id[i]["ep"], 2),
                     captain=(i == cap_i)) for i in xi_i],
        ))

    # how much do the teams actually differ? owning the same 15 twice is not
    # a portfolio, it is the same bet placed twice.
    for c in team_cards:
        mine = all_sets.get(c["id"], set())
        others = set()
        for k, v in all_sets.items():
            if k != c["id"]:
                others |= v
        c["unique"] = len(mine - others) if others else len(mine)
    if len(team_cards) > 1:
        pair_overlap = []
        keys = [c["id"] for c in team_cards]
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = all_sets.get(keys[i], set()), all_sets.get(keys[j], set())
                if a and b:
                    pair_overlap.append(dict(
                        a=next(c["name"] for c in team_cards if c["id"] == keys[i]),
                        b=next(c["name"] for c in team_cards if c["id"] == keys[j]),
                        shared=len(a & b),
                        pct=round(100 * len(a & b) / max(1, len(a | b)))))
    else:
        pair_overlap = []

    # ------------------------------------------------------------------
    # Fixtures for the whole rest of the season, not just the planning window.
    #
    # All 38 gameweeks are already published, and the run of games a club walks
    # into is the one thing about the future that is known rather than guessed.
    # Seeing only five of them hides exactly the swings worth planning around -
    # a side with a brutal opening but the best run in the league from GW6 is
    # invisible until it is too late to act on cheaply.
    #
    # This is affordable because a fixture view is only Poisson over the 380
    # fixtures; no player is projected here. Player EP stays on `span`.
    # ------------------------------------------------------------------
    log("Reading the fixture list to the end of the season...")
    season = [g for g in range(gw, 39)]
    views_season = {g: (views[g] if g in views else fm.team_view(fx, g)) for g in season}

    fixture_grid = []
    for tid, nm in names.items():
        cells = []
        for g in season:
            v = views_season[g].get(tid, [])
            cells.append(dict(gw=g, xg=round(sum(f["xg_for"] for f in v), 2) if v else None,
                              opp=" + ".join(f"{short[f['opp']]}{'(H)' if f['home'] else '(A)'}"
                                             for f in v) if v else "Blank",
                              n=len(v),
                              cs=round(max((f["p_cs"] for f in v), default=0), 3)))
        near = [c for c in cells if c["gw"] in horizon]
        # Where does this club's best run of games actually fall? Reported as the
        # strongest 5-gameweek stretch anywhere in the remainder of the season,
        # so a swing can be bought before the market prices it in.
        best_from, best_val = None, -1.0
        for i in range(len(cells) - 4):
            w = sum(c["xg"] or 0 for c in cells[i:i + 5])
            if w > best_val:
                best_val, best_from = w, cells[i]["gw"]
        fixture_grid.append(dict(
            club=nm, short=short[tid], cells=cells,
            total=round(sum(c["xg"] or 0 for c in near), 2),
            season_total=round(sum(c["xg"] or 0 for c in cells), 2),
            best_run_from=best_from, best_run=round(best_val, 2),
            blanks=[c["gw"] for c in cells if c["n"] == 0],
            doubles=[c["gw"] for c in cells if c["n"] > 1]))
    fixture_grid.sort(key=lambda r: -r["total"])

    # ------------------------------------------------------------------
    # The underlying numbers, shown rather than only consumed.
    #
    # Every projection on this site is built from xG, xA, defensive actions and
    # BPS, but none of it was ever visible - you had to trust the output. These
    # come from FPL's own feed, which is Opta data, plus last season's per-90
    # rates from the archive.
    #
    # Three columns matter and they are deliberately separate: what a player has
    # done THIS season, what he did LAST season, and what the model actually
    # believes about him now. The third is not an average of the first two - it
    # is shrunk toward a positional mean by how many minutes each sample rests
    # on, which is why a player with two good games does not leap to the top.
    # Before a ball is kicked the first column is all zeros, and that is correct
    # rather than broken.
    # ------------------------------------------------------------------
    log("Collecting the underlying numbers...")
    stats = []
    for p in pool:
        e = p["element"]
        mins_now = int(e.get("minutes") or 0)
        prior = (pprior or {}).get(str(e.get("code"))) or {}
        if p["p_appear"] <= 0 and not mins_now and not prior.get("minutes"):
            continue
        r = pm.rates(e)
        stats.append(dict(
            name=p["name"], club=p["club"], pos=POS_NAME[p["pos"]],
            price=round(p["price"], 1), owned=round(p["owned"], 1),
            # this season, straight from the feed
            mins=mins_now,
            xg=round(float(e.get("expected_goals") or 0), 2),
            xa=round(float(e.get("expected_assists") or 0), 2),
            xgi=round(float(e.get("expected_goal_involvements") or 0), 2),
            xgc90=round(float(e.get("expected_goals_conceded_per_90") or 0), 2),
            # last season, from the archive baselines
            p_mins=int(prior.get("minutes") or 0),
            p_xg90=round(float(prior.get("xg90") or 0), 3),
            p_xa90=round(float(prior.get("xa90") or 0), 3),
            p_ppg=round(float(prior.get("ppg") or 0), 2),
            # what the model actually uses, after shrinkage
            m_xg90=round(r["xg90"], 3), m_xa90=round(r["xa90"], 3),
            m_dc90=round(r["dc90"], 2), m_bps90=round(r["bps90"], 1),
            xgi90=round(r["xg90"] + r["xa90"], 3),
            ep=round(p["ep"], 2),
            mine=(p["id"] in current)))
    stats.sort(key=lambda r: -r["xgi90"])

    # ------------------------------------------------------------------
    # Chips.
    #
    # Four chips are worth real points and are mostly lost to bad timing rather
    # than bad choice: people play Bench Boost on a week their bench is asleep,
    # or Triple Captain on a single fixture because the captain "feels" right.
    # Both are arithmetic, so both are worth stating rather than feeling.
    #
    #   Triple Captain  the captain is counted three times instead of twice, so
    #                   the gain is exactly one more of his score.
    #   Bench Boost     the four bench players score, so the gain is their total.
    #
    # Doubles make both far better and blanks make both worse, which falls out
    # of the per-gameweek projection without any special casing.
    #
    # Free Hit and Wildcard are deliberately not scored here. Their value is the
    # squad you would build instead, which is a fresh optimisation per gameweek
    # and would cost more time than the rest of this run put together. What is
    # reported instead is the signal that makes them worth considering: how many
    # of your eleven are actually playing that week.
    # ------------------------------------------------------------------
    # Scored across every remaining gameweek, not just the planning window. The
    # weeks that actually justify a chip are doubles, and those land late - a
    # six-week view can only ever report the floor. Affordable because this needs
    # the fifteen players you own projected forward, not all 590.
    ep_season = {}
    for pid in squad_ids:
        p = by_id.get(pid)
        if not p:
            continue
        ep_season[pid] = {g: pm.project(p["element"], views_season[g].get(p["club_id"], []),
                                        gw_offset=g - gw)["ep"] for g in season}

    chips = []
    if squad_ids and ep_season:
        for g in season:
            try:
                xg, cg, _v, bg, _d = optimise.rank_xi(
                    squad_ids, ep_season, g, by_id, formation=formation,
                    force_start=force_start, force_bench=force_bench)
            except Exception:
                continue
            # Tripling a score is an even more convex payoff than doubling it,
            # so the chip belongs on the fattest tail among the credible names,
            # not on whoever happens to lead on average that week. rank_xi picks
            # its captain on mean, which had the chip hopping between players on
            # gaps of a tenth of a point - noise, presented as a decision.
            cands = sorted((i for i in xg if i in by_id),
                           key=lambda i: -ep_season.get(i, {}).get(g, 0.0))[:4]
            if cands:
                cg = max(cands, key=lambda i: by_id[i].get("cap_score", by_id[i]["ep"]))
            playing = sum(1 for i in xg if ep_season.get(i, {}).get(g, 0) > 0)
            n_fix = {i: len(views_season[g].get(by_id[i]["club_id"], [])) for i in xg if i in by_id}
            chips.append(dict(
                gw=g,
                triple_captain=round(ep_season.get(cg, {}).get(g, 0.0) if cg else 0.0, 2),
                captain=(by_id[cg]["name"] if cg in by_id else None),
                bench_boost=round(sum(ep_season.get(b, {}).get(g, 0.0) for b in bg), 2),
                xi_playing=playing,
                doubles=sum(1 for v in n_fix.values() if v > 1)))
    chip_best = {}
    if chips:
        chip_best = dict(
            triple_captain=max(chips, key=lambda c: c["triple_captain"]),
            bench_boost=max(chips, key=lambda c: c["bench_boost"]),
            # a week where the XI cannot field eleven is what a Free Hit is for
            free_hit=min(chips, key=lambda c: c["xi_playing"]))

    # ------------------------------------------------------------------
    # What has changed since the last run.
    #
    # A player worth owning is usually obvious; a player *becoming* worth owning
    # is not, and that is the window where he is still cheap and unowned. This
    # compares the model's current beliefs against its previous ones so a rise
    # shows up while it is happening rather than at the point everyone has it.
    # ------------------------------------------------------------------
    from fplbrain import history as histmod
    hist_rows = {}
    for p in pool:
        if p["p_appear"] <= 0:
            continue
        e = p["element"]
        hist_rows[str(p["id"])] = dict(
            name=p["name"], club=p["club"], pos=POS_NAME[p["pos"]],
            aps=round(sum(ep[p["id"]][g] for g in look_ahead) / max(1, len(look_ahead)), 3),
            ceiling=p.get("ceiling", 0), price=round(p["price"], 1),
            owned=round(p["owned"], 1), status=p["status"], news=(e.get("news") or "")[:90])
    try:
        updates, last_seen = histmod.movement(gw, hist_rows)
        histmod.snapshot(gw, hist_rows)
    except Exception as exc:
        log(f"Trend tracking unavailable ({exc}).")
        updates, last_seen = [], None

    # Price and availability news straight from FPL, which is fact rather than
    # projection and so is listed separately from the model's own movement.
    news_rows = []
    for p in pool:
        e = p["element"]
        chg = (e.get("cost_change_event") or 0) / 10.0
        flagged = p["status"] != "a" or (e.get("news") or "").strip()
        if not chg and not flagged:
            continue
        news_rows.append(dict(
            name=p["name"], club=p["club"], pos=POS_NAME[p["pos"]],
            price=round(p["price"], 1), price_change=chg,
            status=p["status"], owned=round(p["owned"], 1),
            news=(e.get("news") or "").strip()[:120],
            chance=e.get("chance_of_playing_next_round")))
    news_rows.sort(key=lambda r: (-abs(r["price_change"]), -r["owned"]))

    # Money the model has never been told about.
    #
    # Everything it can suggest is bounded by squad value plus bank, and bank is
    # typed in by hand. Leave it at zero while actually holding a few million and
    # the optimiser is solving a different, poorer problem than the real one -
    # then honestly reports that squad as the best available, because for the
    # budget it was given it is. That reads as the model being weak when it is
    # simply short of funds, so say so plainly.
    # The best 15 the model can build, full stop.
    #
    # Everything else on the page is phrased as edits to what you already own,
    # which answers "what should I change" but never "what would you pick". With
    # a whole squad still free to build before the first deadline that is the
    # question actually being asked, and a list of eleven swaps is a poor way to
    # answer it.
    dream = None
    try:
        d_legal = [p for p in pool if p["p_appear"] >= 0.35]
        d_res = optimise.build_squad(d_legal, ep, float(cfg.get("budget", 100.0)),
                                     horizon, decay=float(cfg.get("decay", 0.88)))
        d_ids = [p["id"] for p in d_res["squad"]]
        d_xi, d_cap, _dv, d_bench, _dd = optimise.rank_xi(d_ids, ep, gw, by_id)
        dream = dict(
            cost=round(d_res["cost"], 1),
            proj=round(sum(by_id[i]["ep"] for i in d_xi)
                       + (by_id[d_cap]["ep"] if d_cap in by_id else 0), 2),
            captain=(by_id[d_cap]["name"] if d_cap in by_id else None),
            xi=[dict(name=by_id[i]["name"], club=by_id[i]["club"],
                     pos=POS_NAME[by_id[i]["pos"]], price=round(by_id[i]["price"], 1),
                     ep=round(by_id[i]["ep"], 2), captain=(i == d_cap))
                for i in sorted(d_xi, key=lambda i: (by_id[i]["pos"], -by_id[i]["ep"]))],
            bench=[dict(name=by_id[i]["name"], club=by_id[i]["club"],
                        pos=POS_NAME[by_id[i]["pos"]], price=round(by_id[i]["price"], 1),
                        ep=round(by_id[i]["ep"], 2))
                   for i in d_bench if i in by_id],
            owned=sum(1 for i in d_ids if i in current))
    except Exception as exc:
        log(f"Could not build the reference squad ({exc}).")

    budget_seen = round(squad_value + bank, 1)
    budget_gap = round(float(cfg.get("budget", 100.0)) - budget_seen, 1)
    budget_short = bool(source == "manual" and budget_gap >= 0.5)

    caveats = []
    caveats.append(phase_note)
    if ts.unmatched_priors:
        caveats.append("No baseline matched for: " + ", ".join(ts.unmatched_priors)
                       + ". Treated as league-average.")
    ps = ts.prior_share(list(ts.attack)[0])
    caveats.append(f"Team ratings: {ps:.0%} last season, {1-ps:.0%} this season. "
                   "This shifts automatically as games are played.")
    if (sum(ts.games.values()) / max(1, len(ts.games))) < 4:
        caveats.append("Fewer than 4 games played this season — player numbers are "
                       "based mostly on last season. Treat as a starting point.")
    if squad_error:
        caveats.append(squad_error)
    if source == "manual" and squad_problems:
        caveats.append(
            f"{len(squad_problems)} player(s) in your squad no longer match the position "
            "they were entered under (FPL reclassified them, or a name matched the wrong "
            "player) - your squad may not actually have a legal 2/5/5/3 shape. Check the "
            "flagged rows in My Squad.")
    caveats.append("Expected points are averages, not ceilings. For captain, lean to the "
                   "highest-ceiling player among the top few, not strictly the top number.")

    ev = next((e for e in evs if e["id"] == gw), {})
    plan_out = None
    if plan:
        # Pair each sale with a purchase in the SAME position.
        #
        # The solver returns two unordered lists - who leaves, who arrives. Those
        # balance position by position ONLY when the squad going in already had
        # the legal 2/5/5/3 shape - the target squad quota is fixed, so if you
        # start illegal (see the squad-shape caveat above) the solver correctly
        # sells more of one position than it buys to reach 2/5/5/3, which is a
        # real, budget-balanced part of the plan.
        #
        # zip() truncates to the shorter list, so pairing with it silently
        # dropped that extra leg from what was shown - a real sale (or purchase)
        # vanished from the display while still being counted in the totals,
        # which made a fully-affordable plan look like it was asking for money
        # you did not have.
        #
        # The fix must keep `out` and `inn` the same length AND aligned by
        # index, because the page pairs them positionally (out[i] with inn[i]).
        # Emitting ragged lists there is worse than the truncation: every entry
        # after the imbalance shifts by one and gets drawn against the wrong
        # partner, which is how a legal plan renders as "a MID for a DEF".
        #
        # So: match like-for-like within a position first, then pair whatever is
        # left over against each other. Leftovers only exist when the squad
        # going in was not a legal 2/5/5/3, and pairing them is the honest
        # description of the move that repairs it - sell the surplus MID, buy
        # the missing FWD. The solver forces total sales to equal total buys, so
        # the two leftover piles are always the same size and nothing is dropped.
        outs_by_pos, ins_by_pos = {}, {}
        for o in plan["out"]:
            outs_by_pos.setdefault(o["pos"], []).append(o)
        for i in plan["inn"]:
            ins_by_pos.setdefault(i["pos"], []).append(i)
        # within a position, most expensive out against most expensive in, so the
        # money in each swap roughly matches instead of reading as a random pairing
        o_price = lambda p: -sell.get(p["id"], p["price"])
        i_price = lambda p: -by_id[p["id"]]["price"]
        pairs, spare_out, spare_in = [], [], []
        for pos in sorted(set(outs_by_pos) | set(ins_by_pos)):
            o_list = sorted(outs_by_pos.get(pos, []), key=o_price)
            i_list = sorted(ins_by_pos.get(pos, []), key=i_price)
            n = min(len(o_list), len(i_list))
            pairs += list(zip(o_list[:n], i_list[:n]))
            spare_out += o_list[n:]
            spare_in += i_list[n:]
        pairs += list(zip(sorted(spare_out, key=o_price), sorted(spare_in, key=i_price)))
        plan_out = dict(
            out=[dict(name=o["name"], club=o["club"], pos=POS_NAME[o["pos"]],
                      price=round(sell.get(o["id"], o["price"]), 1)) for o, _ in pairs],
            inn=[dict(name=i["name"], club=i["club"], pos=POS_NAME[i["pos"]],
                      price=round(by_id[i["id"]]["price"], 1)) for _, i in pairs],
            hits=plan["hits"], used_ft=plan["used_ft"],
            # What the plan actually buys you, on the same basis as the headline
            # projection, so "is there a better option" has a number attached
            # rather than a list of swaps you have to price up yourself.
            proj=round(sum(by_id[i]["ep"] for i in plan["xi"] if i in by_id)
                       + (by_id[plan["captain"]]["ep"] if plan.get("captain") in by_id else 0), 2),
            # The plan maximises the whole horizon, so it may knowingly give up
            # points this week to gain more later. Reporting only the current
            # gameweek made that look like a mistake - the headline number went
            # down while the advice was to act. Both totals are carried so the
            # trade is visible instead of confusing.
            proj_h=_horizon_total([p["id"] for p in plan["squad"]], ep, horizon, by_id),
            mine_h=_horizon_total(list(current), ep, horizon, by_id))

    # ------------------------------------------------------------------
    # Close the self-correction loop.
    #
    # calibrate.py scores old predictions against what really happened and
    # nudges a per-position multiplier, and this file describes that as the
    # mechanism by which the model learns from the season being played. It has
    # never run. log_predictions() was only ever called from run.py, so the web
    # app - the thing actually used every week - recorded nothing, leaving
    # score_gameweek with no predictions to grade. Every multiplier has sat at
    # 1.000 since the project started, and the feature existed on paper only.
    #
    # Two halves, both needed: write down what was predicted, and grade it once
    # the gameweek is finished. Neither is allowed to break a run.
    # ------------------------------------------------------------------
    try:
        calibrate.log_predictions(gw, [
            dict(id=p["id"], name=p["name"], pos=POS_NAME[p["pos"]],
                 ep=p["ep"], price=p["price"], p_appear=p["p_appear"])
            for p in pool if p["p_appear"] > 0])
    except Exception as exc:
        log(f"Could not log predictions ({exc}).")

    try:
        already = {h.get("gw") for h in calibrate.calibration_history()}
        # oldest first, so multipliers move in the order the season happened.
        # Scored, not `finished` - the flag that never confirms GW1 in this
        # feed would have left the self-correction loop silently never running.
        pending = [e["id"] for e in evs
                   if _gw_fully_scored(e["id"], fx) and e["id"] not in already]
        for target_gw in sorted(pending)[-5:]:
            live = cl.live(int(target_gw), force=False)
            actual = {e["id"]: e["stats"]["total_points"] for e in live["elements"]}
            rep = calibrate.score_gameweek(int(target_gw), actual)
            if rep.get("error"):
                continue          # nothing was logged for that week; nothing to learn
            o = rep.get("overall", {})
            log(f"Scored GW{target_gw}: {o.get('n', 0)} players, "
                f"bias {o.get('bias', 0):+.2f}, correlation {o.get('spearman', 0):+.3f}.")
        mult, _cal_state = calibrate.load_calibration()
        _cal_hist = calibrate.calibration_history()
    except Exception as exc:
        log(f"Auto-calibration skipped ({exc}).")
    _cal_hist = locals().get("_cal_hist") or []

    # ------------------------------------------------------------------
    # Note-taking: what actually happened against what was expected.
    #
    # While a gameweek is being played the model must not rebuild plans on it
    # (see season_phase), but the matches are still information and discarding
    # them is its own blindness. So they are recorded and reported, and nothing
    # here feeds back into the projection - calibrate.py owns that loop, slowly
    # and with a minimum sample, so one loud week cannot move the model.
    # ------------------------------------------------------------------
    from fplbrain import form as formmod
    noting = None
    try:
        # The week worth noting is the one in play, or the last one finished -
        # NOT `gw`, which by now points at the next deadline.
        note_gw = next((e["id"] for e in evs if e.get("is_current")), None)
        if note_gw is None:
            done_ids = [e["id"] for e in evs if _gw_fully_scored(e["id"], fx)]
            note_gw = max(done_ids) if done_ids else None
        if note_gw:
            live_rows = cl.live(int(note_gw), force=False)["elements"]
            actual = {e["id"]: e["stats"] for e in live_rows}
            # Project that gameweek fresh rather than trusting a logged run: on
            # an ephemeral host there may be no log, and a squad edit since then
            # would have made it stale anyway.
            note_views = fm.team_view(fx, int(note_gw))
            rows = []
            for p in pool:
                st = actual.get(p["id"])
                if not st:
                    continue
                mins = int(st.get("minutes") or 0)
                # Everyone you own, plus anyone who actually did something.
                if p["id"] not in current and mins == 0:
                    continue
                if p["id"] not in current and float(st.get("total_points") or 0) < 6:
                    continue
                proj = pm.project(p["element"], note_views.get(p["club_id"], []))["ep"]
                rows.append(dict(
                    id=p["id"], name=p["name"], club=p["club"], pos=POS_NAME[p["pos"]],
                    price=round(p["price"], 1), owned=(p["id"] in current),
                    projected=proj, actual=float(st.get("total_points") or 0),
                    minutes=mins,
                    goals=int(st.get("goals_scored") or 0),
                    assists=int(st.get("assists") or 0),
                    bonus=int(st.get("bonus") or 0)))
            graded = formmod.note_players(rows)
            swing = formmod.peaking({p["id"]: [ep[p["id"]][g] for g in horizon]
                                     for p in pool if p["id"] in ep})
            hot = sorted((p for p in pool
                          if p["p_appear"] >= 0.5 and swing.get(p["id"], 0) > 0.35),
                         key=lambda p: -swing[p["id"]])[:12]
            noting = dict(
                gw=note_gw,
                settled=_gw_fully_scored(note_gw, fx),
                players=graded,
                mine=[r for r in graded if r["owned"]],
                comebacks=formmod.comebacks(bs["elements"], pm.availability,
                                            len(horizon))[:12],
                peaking=[dict(id=p["id"], name=p["name"], club=p["club"],
                              pos=POS_NAME[p["pos"]], price=round(p["price"], 1),
                              swing=swing[p["id"]],
                              ep5=round(sum(ep[p["id"]][g] for g in horizon), 2))
                         for p in hot])
            log(f"Noted GW{note_gw}: {len(graded)} players measured against projection.")
    except Exception as exc:
        log(f"Could not take notes on the gameweek ({exc}).")

    # ------------------------------------------------------------------
    # The second unit: hold the goal and judge the maths against it.
    #
    # Everything above is a calculator - it answers the question it was asked
    # and has no opinion about whether the season is going anywhere. This reads
    # the same numbers and says what they mean for the target, including when
    # the honest answer is that the target has gone.
    # ------------------------------------------------------------------
    from fplbrain import brain as brainmod
    brain_view = None
    try:
        # Scored, not `finished` - see season_phase. Getting this wrong means
        # brain.py's pace/target tracking permanently believes zero gameweeks
        # have been played, which understates the rate still needed for every
        # week that follows.
        gws_played = sum(1 for e in evs if _gw_fully_scored(e["id"], fx))
        # entry history if we have it, otherwise nothing banked yet
        points_so_far = int(cfg.get("points_so_far") or 0)
        if source == "live" and live_points_so_far is not None:
            points_so_far = int(live_points_so_far)
        elif source == "live" and overall_rank is not None:
            # the legacy single-entry_id path, which still sets `ent` locally
            points_so_far = int((locals().get("ent") or {}).get("summary_overall_points") or
                                points_so_far)
        proj_now = round(sum(r["ep"] for r in xi) + (by_id[cap]["ep"] if cap else 0), 2)
        best_now = (dream or {}).get("proj") or proj_now
        plan_gain = 0.0
        if plan_out and plan_out.get("proj_h") and plan_out.get("mine_h"):
            plan_gain = max(0.0, plan_out["proj_h"] - plan_out["mine_h"])
        chip_gain = 0.0
        if chip_best:
            chip_gain = (chip_best["triple_captain"]["triple_captain"]
                         + chip_best["bench_boost"]["bench_boost"])
        brain_view = brainmod.assess(
            target=int(cfg.get("season_target", 2700)),
            points_so_far=points_so_far, gws_played=gws_played,
            projection=proj_now, best_possible=best_now,
            plan_gain=plan_gain, chip_gain=chip_gain,
            floor=team_lo, ceiling=team_hi)
    except Exception as exc:
        log(f"Strategy view unavailable ({exc}).")

    return dict(
        gw=gw, horizon=horizon, deadline=ev.get("deadline_time"),
        entry_name=entry_name, entry_id=eid, overall_rank=overall_rank,
        squad_source=source, bank=round(bank, 1), squad_value=round(squad_value, 1),
        squad_problems=squad_problems,
        phase=phase, phase_note=phase_note, noting=noting,
        squad_resolved=squad_resolved,
        my_squad=(mysq["players"] if mysq else None),
        team_cards=team_cards,
        pair_overlap=pair_overlap,
        teams=[dict(id=t["id"], name=t.get("name") or "Team",
                    n=len(t.get("players") or []), bank=t.get("bank", 0.0),
                    free_transfers=t.get("free_transfers", 1),
                    formation=t.get("formation") or "auto")
               for t in all_teams["teams"]],
        active_team=(for_team or all_teams.get("active")),
        route=(None if not route else dict(
            total=route["total"],
            # The last gameweek in any finite plan is distorted: nothing follows
            # it, so the solver spends freely there. Flagged, not hidden.
            horizon_end=span[-1],
            steps=[dict(gw=st["gw"], hits=st["hits"], ft=st["ft"], ep=st["ep"],
                        captain=(by_id[st["captain"]]["name"]
                                 if st.get("captain") in by_id else None),
                        out=[dict(name=o["name"], club=o["club"],
                                  pos=POS_NAME[o["pos"]], price=round(o["price"], 1))
                             for o in st["out"]],
                        inn=[dict(name=n["name"], club=n["club"],
                                  pos=POS_NAME[n["pos"]], price=round(n["price"], 1))
                             for n in st["inn"]])
                   for st in route["route"]])),
        # Self-diagnosis.
        #
        # The team priors were silently missing on the host for weeks: the app
        # looked healthy, produced confident numbers, and rated every club
        # identically. Nothing on screen could have told you. These are the
        # inputs that quietly degrade rather than fail, so each one now reports
        # whether it actually arrived.
        health=dict(
            build=BUILD,
            priors=len(cfg.get("prior_attack") or {}),
            baselines=len(pprior or {}),
            sync=squadmod.sync_enabled(),
            squad_matched=(len(res) if mysq else 0),
            squad_problems=len(squad_problems),
            teams_rated=len(ts.attack),
            # if every club sits at 1.00 the priors did not load, whatever the
            # count above says
            attack_spread=round(max(ts.attack.values()) - min(ts.attack.values()), 3),
            calibrated=len(mult or {}),
            # what the model has actually learned, and from how many gameweeks
            cal_gws=len(_cal_hist),
            cal_mult=mult,
            sim_runs=runs,
            source=source),
        brain=brain_view,
        calibration=dict(gws=len(_cal_hist), multipliers=mult,
                         recent=[dict(gw=h.get("gw"),
                                      n=(h.get("overall") or {}).get("n"),
                                      bias=(h.get("overall") or {}).get("bias"),
                                      spearman=(h.get("overall") or {}).get("spearman"))
                                 for h in _cal_hist[-6:]]),
        dream=dream, budget_seen=budget_seen, budget_gap=budget_gap, budget_short=budget_short,
        stats=stats[:300], stats_season=(pmeta or {}).get("season", "last season"),
        chips=chips, chip_best=chip_best,
        look_ahead=look_ahead,
        updates=updates[:60] + updates[-40:] if len(updates) > 100 else updates,
        news=news_rows[:80],
        last_seen=last_seen,
        formation=formation,
        formations=[optimise.formation_name(f) for f in optimise.FORMATIONS],
        xi_note=xi_dropped,
        unlimited=unlimited,
        sync=squadmod.sync_enabled(),
        free_transfers=ft, plan=plan_out,
        xi=xi, bench=bench_rows, weak=weak, targets=targets[:80],
        captain_table=captain_table,
        xi_ceiling=team_hi,
        xi_floor=team_lo,
        xi_median=team_mid,
        sim_runs=runs,
        xi_total=round(sum(r["ep"] for r in xi), 2),
        xi_total_c=round(sum(r["ep"] for r in xi) + (by_id[cap]["ep"] if cap else 0), 2),
        fixtures=fixture_grid, caveats=caveats,
        prior_share=round(ps, 3), games_played=round(sum(ts.games.values()) / max(1, len(ts.games)), 1),
        seeded=len(pprior),
        players_projected=len(pool),
        all_players=sorted(
            [dict(n=p["name"], c=p["club"], p=round(p["price"], 1),
                  pos=POS_NAME[p["pos"]]) for p in pool],
            key=lambda x: (x["c"], x["n"])),
        generated=time.strftime("%d %b %Y, %H:%M"))


def warm_other_teams():
    """
    Pre-compute every other saved team in the background.

    Switching teams used to re-run the whole pipeline, which is why it took half
    a minute to look at a number the server could have worked out already. The
    shared parts - projections, fixtures, the API itself - are cached by then, so
    each extra team costs only its own optimisation.

    Deliberately silent: it must never touch STATE["status"] or the log, or the
    page would show a boot overlay for work nobody asked to wait on.
    """
    try:
        from fplbrain import squad as squadmod
        teams = [t["id"] for t in squadmod.load_all()["teams"]]
    except Exception:
        return
    for tid in teams:
        with LOCK:
            if tid in STATE["by_team"] or STATE["status"] == "running":
                continue
        try:
            d = compute(force=False, for_team=tid)
            with LOCK:
                STATE["by_team"][tid] = d
        except Exception:
            continue          # one bad team must not stop the rest


def run_job(force=False):
    with LOCK:
        if STATE["status"] == "running":
            return
        STATE["status"] = "running"
        STATE["error"] = None
        STATE["log"] = []
        # Every edit and every refresh routes through here, so this is the one
        # place stale per-team results can be dropped. Selecting a team does not
        # come this way, which is exactly why selecting stays instant.
        STATE["by_team"] = {}
    try:
        data = compute(force)
        with LOCK:
            STATE["data"] = data
            if data.get("active_team"):
                STATE["by_team"][data["active_team"]] = data
            STATE["status"] = "done"
            STATE["updated"] = time.time()
            STATE["message"] = "Ready"
        log("Done.")
        # Work out the other teams too, quietly, so opening one is instant
        # rather than a thirty-second wait. Running several teams is only
        # expensive the first time; after this every switch is served from here.
        threading.Thread(target=warm_other_teams, daemon=True).start()
    except Exception as e:
        tb = traceback.format_exc()
        with LOCK:
            STATE["status"] = "error"
            STATE["error"] = f"{type(e).__name__}: {e}"
            STATE["traceback"] = tb
        print(tb, flush=True)
        log(f"ERROR: {e}")


# =====================================================================
# HTTP
# =====================================================================
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        binary = ctype.startswith("image/") or not isinstance(body, str)
        self.send_header("Content-Type", ctype if binary else ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(b)
        except (BrokenPipeError, ConnectionAbortedError):
            pass

    def do_GET(self):
        code, ctype, body = route("GET", self.path, b"")
        self._send(code, body, ctype)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        code, ctype, body = route("POST", self.path, raw)
        self._send(code, body, ctype)


def route(method, path, raw=b""):
    """
    All request handling lives here, independent of which server is running it.

    The local build serves this through Python's stdlib HTTP server; the cloud
    build serves the same function through FastAPI. One implementation, so the
    two can never drift apart.

    Returns (status_code, content_type, body) where body is str or bytes.
    """
    p = path.split("?")[0]
    J = "application/json"

    if method == "GET":
        if p == "/":
            return 200, "text/html", PAGE
        if p == "/manifest.webmanifest":
            return 200, "application/manifest+json", MANIFEST
        if p == "/sw.js":
            return 200, "application/javascript", SW
        if p.startswith("/static/"):
            name = os.path.basename(p)
            if name in _ICON_SOURCE():
                return 200, "image/png", _ICON_BYTES(name)
            return 404, J, json.dumps({"error": "not found"})
        if p == "/api/status":
            with LOCK:
                return 200, J, json.dumps(dict(
                    status=STATE["status"], message=STATE["message"],
                    error=STATE["error"], log=STATE["log"],
                    updated=STATE["updated"], has_data=STATE["data"] is not None,
                    build=BUILD))
        if p == "/api/lan":
            return 200, J, json.dumps({"url": globals().get("LAN_URL", "")})
        if p == "/api/data":
            with LOCK:
                d = STATE["data"]
            if d is None:
                return 404, J, json.dumps({"error": "not computed yet"})
            return 200, J, json.dumps(d)
        if p == "/api/config":
            cfg = load_config()
            return 200, J, json.dumps({k: cfg.get(k) for k in
                ("entry_id", "horizon", "ft_value", "max_hits",
                 "prior_weight_games", "decay", "sim_runs")})
        if p == "/api/squad":
            from fplbrain import squad as squadmod, optimise as opt
            d = squadmod.load()
            d = d or {"players": squadmod.DEFAULT, "bank": 0.0, "free_transfers": 1,
                      "formation": "auto", "unsaved": True}
            d["formations"] = [opt.formation_name(f) for f in opt.FORMATIONS]
            return 200, J, json.dumps(d)
        if p == "/api/teams":
            from fplbrain import squad as squadmod
            return 200, J, json.dumps(squadmod.load_all())
        if p == "/api/players":
            with LOCK:
                d = STATE["data"]
            return 200, J, json.dumps(d.get("all_players", []) if d else [])
        return 404, J, json.dumps({"error": "not found"})

    if method == "POST":
        try:
            body = json.loads((raw or b"{}").decode() or "{}")
        except Exception:
            body = {}
        if p == "/api/refresh":
            threading.Thread(target=run_job, kwargs={"force": bool(body.get("force"))},
                             daemon=True).start()
            return 200, J, json.dumps({"ok": True})
        if p == "/api/config":
            cfg = load_config()
            for k in ("entry_id", "horizon", "ft_value", "max_hits",
                      "prior_weight_games", "decay", "sim_runs"):
                if k in body:
                    v = body[k]
                    if v in ("", None):
                        cfg[k] = None
                    else:
                        try:
                            cfg[k] = int(v) if k in ("entry_id", "horizon", "max_hits",
                                                     "sim_runs") else float(v)
                        except Exception:
                            cfg[k] = v
            save_config(cfg)
            threading.Thread(target=run_job, daemon=True).start()
            return 200, J, json.dumps({"ok": True})
        if p == "/api/squad":
            from fplbrain import squad as squadmod
            try:
                if body.get("clear"):
                    squadmod.clear()
                else:
                    pl = [x for x in (body.get("players") or []) if (x.get("name") or "").strip()]
                    # entry_id: absent from the body means "leave it alone" (an
                    # old client that predates this field must not blank out a
                    # sync it knows nothing about); an explicit "" clears it.
                    eid_in = body.get("entry_id")
                    eid_in = (str(eid_in).strip() if eid_in not in (None, "") else
                             ("" if "entry_id" in body else None))
                    squadmod.upsert_team(body.get("team_id"), body.get("name"), pl,
                                         body.get("bank", 0), body.get("free_transfers", 1),
                                         formation=body.get("formation"), entry_id=eid_in)
                threading.Thread(target=run_job, daemon=True).start()
                return 200, J, json.dumps({"ok": True})
            except Exception as e:
                return 200, J, json.dumps({"error": str(e)})
        if p == "/api/teams":
            from fplbrain import squad as squadmod
            try:
                act = body.get("action")
                if act == "select":
                    tid = body.get("team_id")
                    squadmod.set_active(tid)
                    # Already worked out? Serve it and skip the recompute
                    # entirely - this is what made switching instant.
                    with LOCK:
                        cached = STATE["by_team"].get(tid)
                        if cached:
                            STATE["data"] = cached
                            STATE["status"] = "done"
                            STATE["message"] = "Ready"
                    if cached:
                        return 200, J, json.dumps({"ok": True, "cached": True,
                                                   "teams": squadmod.load_all()})
                elif act == "delete":
                    squadmod.delete_team(body.get("team_id"))
                elif act == "duplicate":
                    squadmod.duplicate_team(body.get("team_id"))
                elif act == "new":
                    squadmod.upsert_team(None, body.get("name") or "New team",
                                         squadmod.DEFAULT, 0.0, 1)
                elif act == "rename":
                    d = squadmod.load_all()
                    for t in d["teams"]:
                        if t["id"] == body.get("team_id"):
                            squadmod.upsert_team(t["id"], body.get("name"), t["players"],
                                                 t.get("bank", 0), t.get("free_transfers", 1),
                                                 make_active=False)
                            break
                elif act == "apply_all":
                    squadmod.apply_to_all(formation=body.get("formation"),
                                          bank=body.get("bank"),
                                          free_transfers=body.get("free_transfers"),
                                          roles=body.get("roles"))
                elif act == "restore":
                    # Browser-side backup replayed after ephemeral hosted storage came
                    # back empty. Refuse if the server already has teams - never clobber
                    # a fresher server-side edit with a stale browser snapshot.
                    if not squadmod.load_all()["teams"]:
                        squadmod.restore_all(body.get("data") or {})
                else:
                    return 200, J, json.dumps({"error": "unknown action"})
                threading.Thread(target=run_job, daemon=True).start()
                return 200, J, json.dumps({"ok": True, "teams": squadmod.load_all()})
            except Exception as e:
                return 200, J, json.dumps({"error": str(e)})
        if p == "/api/calibrate":
            try:
                from fplbrain.api import LiveClient as _LC
                from fplbrain import calibrate
                cfg = load_config()
                cl = _LC(cfg.get("cache_minutes", 60))
                bs = cl.bootstrap(force=True)
                cur = next((e["id"] for e in bs["events"] if e.get("is_current")), None)
                target = body.get("gw") or (cur or 1)
                live = cl.live(int(target), force=True)
                actual = {e["id"]: e["stats"]["total_points"] for e in live["elements"]}
                return 200, J, json.dumps(calibrate.score_gameweek(int(target), actual))
            except Exception as e:
                return 200, J, json.dumps({"error": str(e)})
        return 404, J, json.dumps({"error": "not found"})

    return 405, J, json.dumps({"error": "method not allowed"})


def _ICON_SOURCE():
    """Icon names available. Overridden in the single-file build."""
    d = os.path.join(HERE, "static")
    return [f for f in os.listdir(d) if f.endswith(".png")] if os.path.isdir(d) else []


def _ICON_BYTES(name):
    with open(os.path.join(HERE, "static", name), "rb") as f:
        return f.read()


def free_port(start=8731, host="127.0.0.1"):
    for port in range(start, start + 40):
        with socket.socket() as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    return 0


def bind_with_retry(port, attempts=12, wait=3.0):
    """
    Take the port, waiting for it if something else still has it.

    When a container restarts, the previous process can hold the listening socket
    for a while. Failing instantly turns a few seconds of overlap into a dead
    Space, so we wait it out instead.
    """
    for i in range(attempts):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            s.listen(128)
            return s
        except OSError as e:
            s.close()
            print(f"  port {port} in use ({e.errno}); waiting for the previous "
                  f"process to release it [{i + 1}/{attempts}]")
            time.sleep(wait)
    return None


def lan_ip():
    """Best guess at this machine's address on the local network."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))          # no packets sent, just picks a route
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


MANIFEST = json.dumps({
    "name": "FPL Brain", "short_name": "FPL Brain",
    "description": "Fantasy Premier League squad and transfer advice",
    "start_url": "/", "scope": "/", "display": "standalone",
    "orientation": "portrait-primary",
    "background_color": "#0d1117", "theme_color": "#0d1117",
    "icons": [
        {"src": "/static/icon-192.png", "sizes": "192x192", "type": "image/png"},
        {"src": "/static/icon-512.png", "sizes": "512x512", "type": "image/png"},
        {"src": "/static/icon-maskable-512.png", "sizes": "512x512",
         "type": "image/png", "purpose": "maskable"},
    ],
})

# Cache the shell so the app opens instantly and still shows the last squad
# when there is no connection. Live data is always network-first.
SW = """
const C='fplbrain-__CACHE_VER__';
const SHELL=['/','/static/icon-192.png','/static/icon-512.png','/manifest.webmanifest'];
self.addEventListener('install',e=>{
  e.waitUntil(caches.open(C).then(c=>c.addAll(SHELL)).then(()=>self.skipWaiting()));
});
self.addEventListener('activate',e=>{
  e.waitUntil(caches.keys().then(k=>Promise.all(
    k.filter(x=>x!==C).map(x=>caches.delete(x)))).then(()=>self.clients.claim()));
});
self.addEventListener('fetch',e=>{
  const u=new URL(e.request.url);
  if(e.request.method!=='GET')return;
  if(u.pathname.startsWith('/api/')){
    e.respondWith(fetch(e.request).then(r=>{
      if(u.pathname==='/api/data'&&r.ok){const c=r.clone();caches.open(C).then(x=>x.put(e.request,c));}
      return r;
    }).catch(()=>caches.match(e.request)));
    return;
  }
  e.respondWith(caches.match(e.request).then(r=>r||fetch(e.request).then(res=>{
    if(res.ok){const c=res.clone();caches.open(C).then(x=>x.put(e.request,c));}
    return res;
  })));
});
"""

PAGE = ""   # filled in from ui.html at import time
_ui = os.path.join(HERE, "ui.html")
if os.path.exists(_ui):
    with open(_ui, encoding="utf-8") as f:
        PAGE = f.read()

# The service worker cache name is derived from the page content itself, so any
# UI change automatically busts phones that already installed the PWA. A fixed
# version string here previously meant updates silently never reached anyone who
# had already opened the app once - the SW script must itself change bytes for
# the browser to even notice there's a new version to install.
SW = SW.replace("__CACHE_VER__", hashlib.md5(PAGE.encode()).hexdigest()[:10])


def serve_cloud(port):
    """
    Serve through FastAPI, which is what Hugging Face's free Gradio Spaces run on.

    Docker Spaces went paid, and Static Spaces cannot run Python, so Gradio is
    the only free tier that can host this. Gradio ships FastAPI and uvicorn, so
    we borrow those and mount a token Gradio page to keep the SDK happy.

    Falls back to the stdlib server if any of that is missing - the platform only
    really needs something answering on the port.
    """
    try:
        from fastapi import FastAPI
        from starlette.routing import Route
        from starlette.responses import Response
        import uvicorn
    except Exception as e:
        print(f"  FastAPI unavailable ({e}); using the built-in server instead")
        threading.Thread(target=run_job, daemon=True).start()
        ThreadingHTTPServer.allow_reuse_address = True
        for _ in range(12):
            try:
                ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
                return
            except OSError:
                print(f"  port {port} busy, retrying in 3s")
                time.sleep(3)
        return

    api = FastAPI(docs_url=None, redoc_url=None)

    if _ZEROGPU:
        print("  spaces package present - GPU placeholder registered at import")
        print("  NOTE: this app needs no GPU. 'CPU basic' is free and correct.")

    # Mount Gradio FIRST so its routes win over our catch-all below.
    if os.environ.get("FPL_NO_GRADIO") != "1":
        try:
            import gradio as gr
            with gr.Blocks(title="FPL Brain") as _demo:
                gr.Markdown("### FPL Brain -- the app is at the root URL of this Space.")
            api = gr.mount_gradio_app(api, _demo, path="/_gradio")
            print("  gradio mounted at /_gradio")
        except Exception as e:
            print(f"  gradio not mounted ({e}) - harmless, the app does not need it")

    # Registered as a raw Starlette route rather than with @api.get/@api.post.
    # FastAPI's decorators inspect type annotations to decide what to inject, and
    # this file carries "from __future__ import annotations", which turns those
    # annotations into strings it cannot resolve for names imported inside a
    # function. The symptom is a 422 on every POST. A Starlette Route takes the
    # request positionally and does no introspection at all.
    async def _handler(request):
        body = await request.body() if request.method == "POST" else b""
        code, ctype, out = route(request.method, request.url.path, body)
        b = out.encode("utf-8") if isinstance(out, str) else out
        return Response(content=b, status_code=code, media_type=ctype,
                        headers={"Cache-Control": "no-store"})

    api.router.routes.append(
        Route("/{full_path:path}", _handler, methods=["GET", "POST"]))

    sock = bind_with_retry(port)
    if sock is None:
        print(f"  FATAL: port {port} never became free.")
        print("  On Hugging Face this nearly always means the Space is on ZeroGPU")
        print("  hardware. Go to Settings -> Hardware and pick 'CPU basic' (free),")
        print("  then Factory rebuild.")
        return
    print(f"  serving on 0.0.0.0:{port}")
    # Start the model only once the port is held. Binding first means the platform
    # health check succeeds straight away rather than waiting on a 30s first run.
    threading.Thread(target=run_job, daemon=True).start()
    uvicorn.Server(uvicorn.Config(api, log_level="warning")).run(sockets=[sock])


def main():
    global PAGE
    if not PAGE:
        print("ERROR: ui.html is missing from this folder.")
        input("Press Enter to close...")
        return

    # Bind to every interface so phones and laptops on the same wifi can connect.
    # PORT is set by cloud hosts; honour it when present.
    env_port = os.environ.get("PORT")
    # Any host that hands us a PORT is a hosted environment. Render, Koyeb, Cloud
    # Run and Hugging Face all do this; a local double-click never does.
    cloud = bool(os.environ.get("SPACE_ID") or os.environ.get("FPL_CLOUD")
                 or os.environ.get("RENDER") or env_port)
    host = "0.0.0.0"
    if cloud:
        port = int(env_port or 7860)
        print("=" * 62)
        print("  FPL Brain - cloud mode")
        serve_cloud(port)
        return
    port = int(env_port) if env_port else free_port(8731, host)
    if not port:
        print("Could not find a free port.")
        input("Press Enter to close...")
        return
    url = f"http://127.0.0.1:{port}/"
    ip = lan_ip()
    srv = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=run_job, daemon=True).start()
    print("=" * 62)
    print("  FPL Brain is running.")
    print(f"  On this computer:      {url}")
    if ip:
        print(f"  On your phone/tablet:  http://{ip}:{port}/")
        print("  (same wifi only - type it into your phone's browser)")
    print()
    print("  Leave this window open. Close it when you're finished.")
    print("=" * 62)
    globals()["LAN_URL"] = f"http://{ip}:{port}/" if ip else ""
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
