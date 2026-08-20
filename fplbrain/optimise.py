"""
Optimisation.

Two jobs:

  build_squad()  - pick 15 from scratch inside the budget (wildcard / season start)
  plan_transfers() - given the squad you actually own, decide what to do this week

Both are integer programs, because FPL decisions are binary and constrained.
You cannot own 0.6 of a player, and "best 15 by value" is usually not a legal
team. A ranking cannot express that; a solver can.

Selling price matters and is handled properly: FPL gives you purchase price plus
half of any rise, rounded down to 0.1. The API hands us `selling_price` directly,
so we use it rather than guessing.
"""
from __future__ import annotations
import pulp

SQUAD_QUOTA = {1: 2, 2: 5, 3: 5, 4: 3}
XI_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
XI_MAX = {1: 1, 2: 5, 3: 5, 4: 3}
MAX_PER_CLUB = 3
HIT_COST = 4


def _solve(prob):
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return pulp.LpStatus[prob.status]


def _solve_limited(prob, seconds):
    """
    Solve with a wall-clock budget.

    The multi-period model is big enough that proving optimality can take far
    longer than finding an answer that is optimal in practice. A dashboard that
    returns a very good route in twenty seconds beats one that returns a
    provably perfect route after five minutes, so CBC is cut off and whatever
    incumbent it holds is used.
    """
    try:
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=seconds))
    except TypeError:                       # older pulp without timeLimit
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return pulp.LpStatus[prob.status]


def build_squad(pool, ep, budget=100.0, horizon_gws=None, decay=0.88,
                bench_weight=0.12, locked=None, banned=None):
    """
    pool   : list of dicts with id, name, club_id, pos (1-4), price
    ep     : {player_id: {gw: expected_points}}
    locked : ids that must be in the squad
    banned : ids that must not be
    """
    gws = horizon_gws or sorted(next(iter(ep.values())).keys())
    locked, banned = set(locked or []), set(banned or [])
    P = [p for p in pool if p["id"] not in banned]
    idx = range(len(P))
    w = {g: decay ** i for i, g in enumerate(gws)}

    prob = pulp.LpProblem("squad", pulp.LpMaximize)
    s = {i: pulp.LpVariable(f"s{i}", cat="Binary") for i in idx}
    x = {(i, g): pulp.LpVariable(f"x{i}_{g}", cat="Binary") for i in idx for g in gws}
    c = {(i, g): pulp.LpVariable(f"c{i}_{g}", cat="Binary") for i in idx for g in gws}

    def E(i, g):
        return ep.get(P[i]["id"], {}).get(g, 0.0)

    prob += pulp.lpSum(
        w[g] * (E(i, g) * x[(i, g)] + E(i, g) * c[(i, g)]
                + bench_weight * E(i, g) * (s[i] - x[(i, g)]))
        for i in idx for g in gws)

    prob += pulp.lpSum(P[i]["price"] * s[i] for i in idx) <= budget
    prob += pulp.lpSum(s.values()) == 15
    for pos, n in SQUAD_QUOTA.items():
        prob += pulp.lpSum(s[i] for i in idx if P[i]["pos"] == pos) == n
    for club in set(p["club_id"] for p in P):
        prob += pulp.lpSum(s[i] for i in idx if P[i]["club_id"] == club) <= MAX_PER_CLUB
    for i in idx:
        if P[i]["id"] in locked:
            prob += s[i] == 1

    for g in gws:
        prob += pulp.lpSum(x[(i, g)] for i in idx) == 11
        for pos in SQUAD_QUOTA:
            prob += pulp.lpSum(x[(i, g)] for i in idx if P[i]["pos"] == pos) >= XI_MIN[pos]
            prob += pulp.lpSum(x[(i, g)] for i in idx if P[i]["pos"] == pos) <= XI_MAX[pos]
        prob += pulp.lpSum(c[(i, g)] for i in idx) == 1
        for i in idx:
            prob += x[(i, g)] <= s[i]
            prob += c[(i, g)] <= x[(i, g)]

    status = _solve(prob)
    squad = [P[i] for i in idx if s[i].value() and s[i].value() > 0.5]
    xi = {g: [P[i]["id"] for i in idx if x[(i, g)].value() and x[(i, g)].value() > 0.5] for g in gws}
    cap = {g: next((P[i]["id"] for i in idx if c[(i, g)].value() and c[(i, g)].value() > 0.5), None)
           for g in gws}
    return dict(status=status, squad=squad, xi=xi, captain=cap,
                objective=pulp.value(prob.objective),
                cost=sum(p["price"] for p in squad))


def plan_route(current_ids, pool, ep, bank, free_transfers, gws,
               decay=0.88, bench_weight=0.12, free_first=False,
               max_squad=None, time_limit=25, churn_cost=0.25):
    """
    Plan a transfer *route*, not just this week's move.

    plan_transfers() picks one squad and holds it for the whole horizon, which
    quietly assumes the best team for GW1 is the best team for GW5. It is not:
    fixtures swing, and the whole point of knowing all 38 in advance is to buy a
    player before the run that makes him good rather than after.

    So ownership becomes a per-gameweek decision. The solver may sell in GW3 and
    buy back in GW6 if the fixtures justify it, and it pays for the privilege
    under real FPL rules - one free transfer a week, banked up to five, four
    points a hit beyond that.

    Two simplifications, both deliberate and both stated rather than hidden:
      * prices are held at today's value, so this does not try to farm rises.
        Team value drift is unmodelled elsewhere too.
      * availability is today's, so a knock is assumed to persist across the
        route. That makes the plan conservative about anyone flagged now.

    Returns a per-gameweek route: what to sell, what to buy, the XI, the
    captain, hits taken and free transfers left standing.
    """
    have = set(current_ids)
    P = list(pool)
    # Ownership is decided per gameweek, so the model is |pool| x |gws| times
    # bigger than the single-week one. Trimming to a credible shortlist is what
    # keeps it solvable; everything currently owned survives regardless.
    if max_squad and len(P) > max_squad:
        def tot(p):
            return sum(ep.get(p["id"], {}).get(g, 0.0) for g in gws)

        keep = {p["id"] for p in P if p["id"] in have}
        budget_left = max(0, max_squad - len(keep))
        # Split the allowance between the best players and the best *value*
        # players, position by position.
        #
        # Ranking purely on points fills the shortlist with premiums, and a
        # 15-man squad cannot be built out of premiums on 100m. With no cheap
        # enablers available the budget constraint has nowhere to give, so the
        # solver's only feasible answer is to keep the squad it was handed - it
        # reported "hold" not because holding was good but because every upgrade
        # was unaffordable inside the shortlist it had been given.
        per_pos = max(4, budget_left // 8)
        for pos in SQUAD_QUOTA:
            cands = [p for p in P if p["pos"] == pos and p["id"] not in have]
            by_points = sorted(cands, key=lambda p: -tot(p))[:per_pos]
            by_value = sorted(cands, key=lambda p: -tot(p) / max(0.1, p["price"]))[:per_pos]
            keep |= {p["id"] for p in by_points} | {p["id"] for p in by_value}
        # spend anything still going on raw points, regardless of position
        rest = sorted((p for p in P if p["id"] not in keep), key=lambda p: -tot(p))
        keep |= {p["id"] for p in rest[:max(0, max_squad - len(keep))]}
        P = [p for p in P if p["id"] in keep]

    idx = range(len(P))
    by_id = {p["id"]: p for p in P}
    w = {g: decay ** i for i, g in enumerate(gws)}
    g0 = gws[0]

    prob = pulp.LpProblem("route", pulp.LpMaximize)
    s = {(i, g): pulp.LpVariable(f"s{i}_{g}", cat="Binary") for i in idx for g in gws}
    x = {(i, g): pulp.LpVariable(f"x{i}_{g}", cat="Binary") for i in idx for g in gws}
    c = {(i, g): pulp.LpVariable(f"c{i}_{g}", cat="Binary") for i in idx for g in gws}
    buy = {(i, g): pulp.LpVariable(f"b{i}_{g}", cat="Binary") for i in idx for g in gws}
    sel = {(i, g): pulp.LpVariable(f"d{i}_{g}", cat="Binary") for i in idx for g in gws}
    hits = {g: pulp.LpVariable(f"h{g}", lowBound=0, upBound=3, cat="Integer") for g in gws}
    ftv = {g: pulp.LpVariable(f"f{g}", lowBound=0, upBound=5, cat="Integer") for g in gws}
    # Cash in hand entering each gameweek. Without this, the money constraint
    # below checked every gameweek's spend against the same starting `bank`
    # figure instead of what was actually left after earlier weeks' business,
    # letting the route re-spend the same bank allowance every week and drift
    # the squad's value up indefinitely across a multi-gameweek plan.
    cash = {g: pulp.LpVariable(f"cash{g}", lowBound=0) for g in gws}

    def E(i, g):
        return ep.get(P[i]["id"], {}).get(g, 0.0)

    # Points, minus hits, minus a small charge for every transfer made.
    #
    # That charge is what stops the plan oscillating. Two squads are regularly
    # within a rounding error of each other, and with a free transfer costing
    # literally nothing the solver had no reason to prefer the one already
    # owned - so it would advise a swap worth a hundredth of a point, and on the
    # next run, from the new squad, advise swapping straight back. Following the
    # advice produced the reverse advice.
    #
    # A quarter point per move is far below anything worth acting on and far
    # above the noise, so genuine upgrades are untouched while ties resolve
    # toward the squad you already have. Banked transfers keep a small credit so
    # flexibility with no immediate use is not simply burned.
    prob += (pulp.lpSum(w[g] * (E(i, g) * x[(i, g)] + E(i, g) * c[(i, g)]
                                + bench_weight * E(i, g) * (s[(i, g)] - x[(i, g)]))
                        for i in idx for g in gws)
             - HIT_COST * pulp.lpSum(hits[g] for g in gws)
             - churn_cost * pulp.lpSum(buy[(i, g)] for i in idx for g in gws)
             + 0.30 * ftv[gws[-1]])

    for g_i, g in enumerate(gws):
        prev = gws[g_i - 1] if g_i else None
        for i in idx:
            was = (1 if P[i]["id"] in have else 0) if prev is None else s[(i, prev)]
            prob += s[(i, g)] - was == buy[(i, g)] - sel[(i, g)]
            prob += buy[(i, g)] + sel[(i, g)] <= 1
            prob += x[(i, g)] <= s[(i, g)]
            prob += c[(i, g)] <= x[(i, g)]

        prob += pulp.lpSum(s[(i, g)] for i in idx) == 15
        for pos, n in SQUAD_QUOTA.items():
            prob += pulp.lpSum(s[(i, g)] for i in idx if P[i]["pos"] == pos) == n
        for club in set(p["club_id"] for p in P):
            prob += pulp.lpSum(s[(i, g)] for i in idx if P[i]["club_id"] == club) <= MAX_PER_CLUB

        prob += pulp.lpSum(x[(i, g)] for i in idx) == 11
        for pos in SQUAD_QUOTA:
            prob += pulp.lpSum(x[(i, g)] for i in idx if P[i]["pos"] == pos) >= XI_MIN[pos]
            prob += pulp.lpSum(x[(i, g)] for i in idx if P[i]["pos"] == pos) <= XI_MAX[pos]
        prob += pulp.lpSum(c[(i, g)] for i in idx) == 1

        # Money. Selling raises what the player is worth to you now; buying costs
        # list price. Holding prices flat means this balances period to period.
        #
        # `bank_in` is what is actually left going into this gameweek: the
        # original bank for the first period, or last period's cash carried
        # forward otherwise. Checking spend against a fixed `bank` in every
        # period let the solver treat each week's allowance as freshly
        # available again, so a route could keep "spending" the same money
        # over and over and finish the horizon worth more than it started -
        # over budget in exactly the way a real squad cannot be.
        spend = pulp.lpSum(P[i]["price"] * buy[(i, g)] for i in idx)
        raise_ = pulp.lpSum(current_ids.get(P[i]["id"], P[i]["price"]) * sel[(i, g)] for i in idx)
        bank_in = bank if prev is None else cash[prev]
        prob += spend <= bank_in + raise_
        prob += cash[g] == bank_in + raise_ - spend

        moves = pulp.lpSum(buy[(i, g)] for i in idx)
        prob += moves == pulp.lpSum(sel[(i, g)] for i in idx)
        if g_i == 0 and free_first:
            prob += hits[g] == 0            # before the first deadline nothing costs
            prob += ftv[g] == free_transfers
        else:
            prob += moves <= ftv[g] + hits[g]
            if prev is None:
                prob += ftv[g] == free_transfers
            elif g_i == 1 and free_first:
                # Rebuilding before the first deadline is free and, crucially,
                # does not spend the free transfer for the week after: you come
                # out of the opening deadline with one regardless of how much you
                # changed. Charging those moves against it left the next
                # gameweek showing zero and planning as if it were stuck.
                prob += ftv[g] == 1
            else:
                # one a week, capped at five. `<=` is safe because the objective
                # wants transfers banked, so it will not sit below the cap.
                prob += ftv[g] <= ftv[prev] - pulp.lpSum(buy[(i, prev)] for i in idx) + 1

    _solve_limited(prob, time_limit)

    def val(v):
        return v.value() and v.value() > 0.5

    route = []
    for g in gws:
        outs = [P[i] for i in idx if val(sel[(i, g)])]
        ins = [P[i] for i in idx if val(buy[(i, g)])]
        xi = [P[i]["id"] for i in idx if val(x[(i, g)])]
        cap = next((P[i]["id"] for i in idx if val(c[(i, g)])), None)
        # zip() truncates to the shorter list. Sells and buys only balance
        # 1-for-1 within a position when the squad going into this gameweek
        # already has the legal 2/5/5/3 shape - correcting an illegal one
        # (a mismatched squad quota carried in from outside this function)
        # needs an uneven number of sells vs buys in at least one position,
        # and truncating there silently dropped a real, budget-counted leg of
        # the move from what gets reported.
        #
        # The two lists must stay the same length and index-aligned, because
        # callers pair them positionally. So match like-for-like first, then
        # pair the leftovers against each other - that is the move that repairs
        # an illegal shape, and since the model forces total sales to equal
        # total buys the leftover piles are always the same size.
        pairs, spare_o, spare_i = [], [], []
        o_by, i_by = {}, {}
        for o in outs:
            o_by.setdefault(o["pos"], []).append(o)
        for n in ins:
            i_by.setdefault(n["pos"], []).append(n)
        by_price = lambda p: -p["price"]
        for pos in sorted(set(o_by) | set(i_by)):
            o_list = sorted(o_by.get(pos, []), key=by_price)
            i_list = sorted(i_by.get(pos, []), key=by_price)
            k = min(len(o_list), len(i_list))
            pairs += list(zip(o_list[:k], i_list[:k]))
            spare_o += o_list[k:]
            spare_i += i_list[k:]
        pairs += list(zip(sorted(spare_o, key=by_price), sorted(spare_i, key=by_price)))
        route.append(dict(
            gw=g, out=[o for o, _ in pairs], inn=[n for _, n in pairs],
            xi=xi, captain=cap,
            hits=int(hits[g].value() or 0), ft=int(ftv[g].value() or 0),
            ep=round(sum(E(i, g) * (2 if val(c[(i, g)]) else 1)
                         for i in idx if val(x[(i, g)])), 2)))
    return dict(status=pulp.LpStatus[prob.status], route=route,
                total=round(sum(r["ep"] for r in route)
                            - HIT_COST * sum(r["hits"] for r in route), 2))


def plan_transfers(current_ids, pool, ep, bank, free_transfers, gws,
                   decay=0.88, max_hits=2, bench_weight=0.12,
                   ft_value=2.5, locked=None):
    """
    Decide this week's move given what you already own.

    current_ids : {player_id: selling_price}
    bank        : money in the bank, £m
    free_transfers : 1..5

    Objective is horizon EP minus 4 per hit minus the opportunity cost of
    spending a banked free transfer. The solver is allowed to do nothing, and
    frequently should — `ft_value` is what stops it churning for +0.3.
    """
    locked = set(locked or [])
    have = set(current_ids)
    P = list(pool)
    by_id = {p["id"]: p for p in P}
    for pid in have:                       # make sure owned players are in the pool
        if pid not in by_id:
            continue
    idx = range(len(P))
    w = {g: decay ** i for i, g in enumerate(gws)}

    prob = pulp.LpProblem("transfers", pulp.LpMaximize)
    s = {i: pulp.LpVariable(f"s{i}", cat="Binary") for i in idx}
    x = {(i, g): pulp.LpVariable(f"x{i}_{g}", cat="Binary") for i in idx for g in gws}
    c = {(i, g): pulp.LpVariable(f"c{i}_{g}", cat="Binary") for i in idx for g in gws}
    buy = {i: pulp.LpVariable(f"b{i}", cat="Binary") for i in idx}
    sell = {i: pulp.LpVariable(f"d{i}", cat="Binary") for i in idx}
    hits = pulp.LpVariable("hits", lowBound=0, upBound=max_hits, cat="Integer")
    used_ft = pulp.LpVariable("used_ft", lowBound=0, upBound=free_transfers, cat="Integer")

    def E(i, g):
        return ep.get(P[i]["id"], {}).get(g, 0.0)

    prob += (pulp.lpSum(w[g] * (E(i, g) * x[(i, g)] + E(i, g) * c[(i, g)]
                                + bench_weight * E(i, g) * (s[i] - x[(i, g)]))
                        for i in idx for g in gws)
             - HIT_COST * hits
             - ft_value * used_ft)

    # transfer accounting
    for i in idx:
        owned = 1 if P[i]["id"] in have else 0
        prob += s[i] - owned == buy[i] - sell[i]
        prob += buy[i] + sell[i] <= 1
        if P[i]["id"] in locked:
            prob += s[i] == 1
    n_transfers = pulp.lpSum(buy[i] for i in idx)
    prob += n_transfers == pulp.lpSum(sell[i] for i in idx)
    prob += n_transfers <= used_ft + hits

    # budget: money out on buys cannot exceed bank plus money in from sales
    prob += (pulp.lpSum(P[i]["price"] * buy[i] for i in idx)
             <= bank + pulp.lpSum(current_ids.get(P[i]["id"], 0.0) * sell[i] for i in idx))

    prob += pulp.lpSum(s.values()) == 15
    for pos, n in SQUAD_QUOTA.items():
        prob += pulp.lpSum(s[i] for i in idx if P[i]["pos"] == pos) == n
    for club in set(p["club_id"] for p in P):
        prob += pulp.lpSum(s[i] for i in idx if P[i]["club_id"] == club) <= MAX_PER_CLUB

    for g in gws:
        prob += pulp.lpSum(x[(i, g)] for i in idx) == 11
        for pos in SQUAD_QUOTA:
            prob += pulp.lpSum(x[(i, g)] for i in idx if P[i]["pos"] == pos) >= XI_MIN[pos]
            prob += pulp.lpSum(x[(i, g)] for i in idx if P[i]["pos"] == pos) <= XI_MAX[pos]
        prob += pulp.lpSum(c[(i, g)] for i in idx) == 1
        for i in idx:
            prob += x[(i, g)] <= s[i]
            prob += c[(i, g)] <= x[(i, g)]

    status = _solve(prob)
    out = [P[i] for i in idx if sell[i].value() and sell[i].value() > 0.5]
    inn = [P[i] for i in idx if buy[i].value() and buy[i].value() > 0.5]
    squad = [P[i] for i in idx if s[i].value() and s[i].value() > 0.5]
    g0 = gws[0]
    xi0 = [P[i]["id"] for i in idx if x[(i, g0)].value() and x[(i, g0)].value() > 0.5]
    cap0 = next((P[i]["id"] for i in idx if c[(i, g0)].value() and c[(i, g0)].value() > 0.5), None)
    return dict(status=status, out=out, inn=inn, squad=squad, xi=xi0, captain=cap0,
                hits=int(hits.value() or 0), used_ft=int(used_ft.value() or 0),
                objective=pulp.value(prob.objective))


# Every legal FPL formation: one keeper, 3-5 defenders, 2-5 midfielders,
# 1-3 forwards, ten outfield in total. Derived rather than typed out so it
# cannot drift from XI_MIN/XI_MAX above.
FORMATIONS = [
    (d, m, f)
    for d in range(XI_MIN[2], XI_MAX[2] + 1)
    for m in range(XI_MIN[3], XI_MAX[3] + 1)
    for f in range(XI_MIN[4], XI_MAX[4] + 1)
    if d + m + f == 10
]


def formation_name(counts):
    """(4, 4, 2) -> '4-4-2'"""
    return "-".join(str(n) for n in counts)


def parse_formation(s):
    """'4-4-2' -> (4, 4, 2), or None if it is not a legal FPL shape."""
    if not s:
        return None
    try:
        parts = tuple(int(n) for n in str(s).strip().split("-"))
    except ValueError:
        return None
    return parts if parts in FORMATIONS else None


def rank_xi(squad_ids, ep, gw, pool_by_id, formation=None,
            force_start=(), force_bench=()):
    """
    Pick the best XI, captain and bench order from a fixed 15.

    formation   : (def, mid, fwd) to lock the shape, or None to let it choose
    force_start : ids that must start
    force_bench : ids that must be benched

    Returns (xi, cap, vice, bench, dropped). `dropped` names the constraints
    that had to be abandoned to get a legal team - asking for 4-4-2 with only
    two fit defenders, or starting twelve players, has no solution, and a
    silently ignored preference is worse than one reported back.
    """
    P = [pool_by_id[i] for i in squad_ids if i in pool_by_id]
    idx = range(len(P))
    E = lambda i: ep.get(P[i]["id"], {}).get(gw, 0.0)
    want_start = set(force_start or ())
    want_bench = set(force_bench or ())

    def attempt(use_formation, use_roles):
        prob = pulp.LpProblem("xi", pulp.LpMaximize)
        x = {i: pulp.LpVariable(f"x{i}", cat="Binary") for i in idx}
        c = {i: pulp.LpVariable(f"c{i}", cat="Binary") for i in idx}
        prob += pulp.lpSum(E(i) * x[i] + E(i) * c[i] for i in idx)
        prob += pulp.lpSum(x.values()) == 11
        for pos in SQUAD_QUOTA:
            n = pulp.lpSum(x[i] for i in idx if P[i]["pos"] == pos)
            if use_formation and pos != 1:
                prob += n == use_formation[pos - 2]
            else:
                prob += n >= XI_MIN[pos]
                prob += n <= XI_MAX[pos]
        if use_roles:
            for i in idx:
                if P[i]["id"] in want_start:
                    prob += x[i] == 1
                elif P[i]["id"] in want_bench:
                    prob += x[i] == 0
        prob += pulp.lpSum(c.values()) == 1
        for i in idx:
            prob += c[i] <= x[i]
            # never captain someone you have benched
            if use_roles and P[i]["id"] in want_bench:
                prob += c[i] == 0
        status = _solve(prob)
        if status != "Optimal":
            return None
        return x, c

    # Fall back one constraint at a time, most specific first, so a request that
    # cannot be met still yields a legal team instead of an error page.
    fixed = parse_formation(formation) if not isinstance(formation, tuple) else formation
    dropped = []
    for use_formation, use_roles, lost in (
            (fixed, True, None),
            (fixed, False, "your start/bench picks"),
            (None, True, "the formation"),
            (None, False, "the formation and your start/bench picks")):
        got = attempt(use_formation, use_roles)
        if got:
            x, c = got
            if lost:
                dropped.append(lost)
            break
    else:
        raise ValueError("no legal XI from this squad")

    xi = [P[i]["id"] for i in idx if x[i].value() and x[i].value() > 0.5]
    cap = next((P[i]["id"] for i in idx if c[i].value() and c[i].value() > 0.5), None)
    # Bench order is autosub priority: the keeper cannot come on for an outfielder,
    # so they sit apart from the ranking rather than at the top of it.
    off = [P[i] for i in idx if P[i]["id"] not in xi]
    bench = [p["id"] for p in sorted((p for p in off if p["pos"] != 1),
                                     key=lambda p: -ep.get(p["id"], {}).get(gw, 0.0))]
    bench = [p["id"] for p in off if p["pos"] == 1] + bench
    vice = max((p for p in P if p["id"] in xi and p["id"] != cap),
               key=lambda p: ep.get(p["id"], {}).get(gw, 0.0), default=None)
    return xi, cap, (vice["id"] if vice else None), bench, (dropped[0] if dropped else None)
