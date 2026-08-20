"""
The model.

Three layers, each feeding the next:

  1. TeamStrength  - attack/defence indices per club, blending a preseason prior
                     with this season's observed xG. The blend weight shifts
                     automatically as games accumulate, so the model stops
                     trusting last season the moment it has enough of this one.

  2. FixtureModel  - Poisson projection of xG for both sides of every fixture.

  3. PlayerModel   - expected FPL points per player per gameweek, built from the
                     actual scoring rules rather than a black box.

Nothing here needs manual data entry. Every input comes from the FPL API.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

# ---- FPL scoring rules, 2026/27 -------------------------------------------
# Verified against premierleague.com "All you need to know about changes to FPL
# for 2026/27" (Jul 2026). DefCon and the 5-transfer roll both carry over.
GOAL_PTS = {1: 6, 2: 6, 3: 5, 4: 4}      # element_type 1 GK, 2 DEF, 3 MID, 4 FWD
CS_PTS = {1: 4, 2: 4, 3: 1, 4: 0}
ASSIST_PTS = 3
DEFCON_PTS = 2
DEFCON_THRESHOLD = {1: 99, 2: 10, 3: 12, 4: 12}   # DEF: CBIT>=10. MID/FWD: CBIRT>=12.
SAVES_PER_POINT = 3
POS_NAME = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# Shrinkage priors: a player's per-90 rate is unreliable until he has minutes.
# We pull raw rates toward these positional means, with the pull decaying as
# minutes accumulate. PRIOR_MINUTES is the pseudo-sample size.
PRIOR_MINUTES = 450
PRIOR_XG90 = {1: 0.005, 2: 0.05, 3: 0.14, 4: 0.34}
PRIOR_XA90 = {1: 0.01, 2: 0.08, 3: 0.16, 4: 0.13}
PRIOR_DEFCON90 = {1: 0.0, 2: 7.2, 3: 6.4, 4: 3.0}
PRIOR_BPS90 = {1: 18.0, 2: 17.0, 3: 15.0, 4: 14.0}

AVAIL = {"a": 1.0, "d": 0.55, "i": 0.0, "s": 0.0, "u": 0.0, "n": 0.0}

# How often a squad player who is not starting still gets on the pitch. Five
# outfield substitutions a game are shared among the bench; a substitute
# goalkeeper needs the starter to come off injured or be sent off, which is why
# these two numbers are nowhere near each other.
SUB_RATE_OUTFIELD = 0.45
SUB_RATE_GK = 0.02

# A club starts one keeper and ten outfielders, every match. These are the
# budgets calibrate_depth() shares out; they are identities, not estimates.
GK_PLACES = 1
OUTFIELD_PLACES = 10

# Premier League players miss something like an eighth of a season to knocks and
# illness. Last season's start count therefore understates how often a player
# starts when he is actually fit, and current fitness is modelled separately in
# p_avail. This converts "starts per fixture" into "starts per fixture he was
# available for" so the two are not both charged.
MISSED_GAME_UPLIFT = 1.15

# --- how a flag decays over the horizon -------------------------------------
# A status is a statement about the NEXT match, not about the next five, and
# `chance_of_playing_next_round` says so in its own name. Applying either to
# every gameweek in the horizon froze today's bad news across five weeks: a
# one-match suspension projected as zero for over a month, and the planner would
# pay a hit to remove someone back in a week.
#
# So the flag is released over subsequent gameweeks, at a rate that depends on
# what kind of flag it is. Each row is the fraction of the gap back to full
# availability that has closed by that many gameweeks ahead.
#
# ASSUMPTION, not measured. A suspension has a known length and is modelled
# sharply; injuries do not, and are deliberately released slowly so the model
# stays pessimistic about anyone carrying a knock. Revisit with real data.
FLAG_RECOVERY = {
    "d": (0.0, 0.70, 0.90, 1.00, 1.00),   # doubtful: usually resolves within a week
    "s": (0.0, 1.00, 1.00, 1.00, 1.00),   # suspension: nearly always a single match
    "i": (0.0, 0.20, 0.40, 0.60, 0.75),   # injured: open-ended, released slowly
    "u": (0.0, 0.10, 0.20, 0.30, 0.40),   # unavailable
    "n": (0.0, 0.10, 0.20, 0.30, 0.40),   # not in squad
}


def flag_recovery(status, gw_offset):
    """Fraction of a flagged player's availability restored `gw_offset` weeks out."""
    if gw_offset <= 0:
        return 0.0
    row = FLAG_RECOVERY.get(status)
    if not row:
        return 0.0
    return row[min(gw_offset, len(row) - 1)]

# --- set-piece and penalty duty ---------------------------------------------
# The API publishes running order for penalties, direct free kicks and corners.
# These are worth real points and are invisible to raw xG/90 from previous
# seasons, because duty changes when players transfer or a taker leaves.
#
# A Premier League team wins roughly 0.12 penalties a game and converts ~0.79 of
# them, so the designated taker carries about 0.095 extra goals per 90 that his
# historical rate may not reflect at all.
PEN_PER_GAME = 0.12
PEN_CONVERSION = 0.79
PEN_UPLIFT = {1: 0.0, 2: PEN_PER_GAME * PEN_CONVERSION, 3: PEN_PER_GAME * PEN_CONVERSION}
# Corners and indirect free kicks mostly convert into assists.
SP_ASSIST_UPLIFT = {1: 0.055, 2: 0.030, 3: 0.015}


def _order(e, key):
    v = e.get(key)
    try:
        v = int(v)
        return v if v >= 1 else None
    except (TypeError, ValueError):
        return None


def penalty_uplift(e) -> float:
    """Extra xG per 90 from being on penalty duty."""
    o = _order(e, "penalties_order")
    return PEN_UPLIFT.get(o, 0.0) if o else 0.0


def setpiece_uplift(e) -> float:
    """Extra xA per 90 from corners / indirect and direct free kicks."""
    total = 0.0
    for key in ("corners_and_indirect_freekicks_order", "direct_freekicks_order"):
        o = _order(e, key)
        if o:
            total += SP_ASSIST_UPLIFT.get(o, 0.0)
    return total

# --- team name matching ----------------------------------------------------
# FPL's `name` field is not stable across seasons or consistent with how anyone
# else writes club names ("Spurs", "Nott'm Forest", "Man Utd"). Matching priors
# on the raw string is brittle, so everything is canonicalised first.
_TEAM_ALIASES = {
    "man utd": "manchester united", "man united": "manchester united",
    "manchester utd": "manchester united", "mufc": "manchester united",
    "man city": "manchester city", "mcfc": "manchester city",
    "spurs": "tottenham", "tottenham hotspur": "tottenham", "thfc": "tottenham",
    "nottm forest": "nottingham forest", "notts forest": "nottingham forest",
    "nott m forest": "nottingham forest", "forest": "nottingham forest",
    "brighton and hove albion": "brighton", "brighton and hove": "brighton",
    "wolverhampton wanderers": "wolves", "wolverhampton": "wolves",
    "west ham united": "west ham", "newcastle united": "newcastle",
    "leeds united": "leeds", "leicester city": "leicester",
    "hull city": "hull", "ipswich town": "ipswich", "coventry city": "coventry",
    "afc bournemouth": "bournemouth", "sheffield united": "sheffield utd",
    "luton town": "luton", "norwich city": "norwich", "swansea city": "swansea",
    "stoke city": "stoke", "cardiff city": "cardiff", "birmingham city": "birmingham",
}


def canon_team(name: str) -> str:
    """Normalise a club name so priors match whatever the API happens to call it."""
    if not name:
        return ""
    s = str(name).lower().strip().replace("&", "and")
    for ch in ".'\u2019-_":
        s = s.replace(ch, "")
    s = " ".join(s.split())
    return _TEAM_ALIASES.get(s, s)


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def poisson_sf(threshold: int, lam: float) -> float:
    """P(X >= threshold) for X ~ Poisson(lam)."""
    if lam <= 0:
        return 0.0
    if threshold <= 0:
        return 1.0
    cdf = sum(poisson_pmf(k, lam) for k in range(threshold))
    return max(0.0, 1.0 - cdf)


def expected_conceded_penalty(lam: float, cap: int = 12) -> float:
    """FPL docks GK/DEF 1 point per 2 goals conceded. E[floor(X/2)] for X~Poisson."""
    return sum(poisson_pmf(k, lam) * (k // 2) for k in range(cap + 1))


def _logistic(x, L, k, x0):
    try:
        return L / (1.0 + math.exp(-k * (x - x0)))
    except OverflowError:
        return 0.0 if x < x0 else L


# --- empirically fitted response curves -------------------------------------
# Fitted by fit.py on all 2025/26 player-matches (60+ minutes), predicting each
# gameweek from the player's cumulative rate BEFORE it. Walk-forward, no leakage.
#
# DefCon was originally modelled as Poisson. Poisson fits the middle but breaks
# in the tails: at 11-13 actions/90 it predicted a 75.8% hit rate where the real
# figure was 56.1%. That would have systematically over-rated high-volume
# defenders, which is exactly the pool of cheap DefCon picks you'd be shopping in.
# The logistic below fits with SSE 0.0001 across the range.
DEFCON_CURVE = {2: (0.65, 0.540, 8.50),    # DEF, threshold 10 CBIT
                3: (0.60, 0.570, 10.50),   # MID, threshold 12 CBIRT
                4: (0.85, 0.870, 10.00)}   # FWD, same threshold, rarely reached

# Bonus: expected bonus points per match given the player's prior BPS/90.
# The naive assumption (bonus scales fast with BPS) is wrong — even a 20 BPS/90
# player averages only ~0.35 bonus a match, because bonus is a per-match ranking
# contest, not a threshold.
BONUS_CURVE = (1.50, 0.080, 35.0)


def defcon_probability(pos: int, actions_per_90: float, minutes_share: float) -> float:
    if pos not in DEFCON_CURVE:
        return 0.0
    L, k, x0 = DEFCON_CURVE[pos]
    return _logistic(actions_per_90 * min(1.0, minutes_share), L, k, x0)


# ===========================================================================
# 1. TEAM STRENGTH
# ===========================================================================
@dataclass
class TeamStrength:
    """
    attack[t]  ~ 1.00 = league-average goals scored
    defence[t] ~ 1.00 = league-average goals conceded (higher = leakier)

    Blend rule (empirical Bayes):
        index = (games * observed + k * prior) / (games + k)

    `k` is in units of games. At k=6, a team's own season data outweighs the
    prior from about GW7 onward. This is the mechanism that makes the system
    "evolve" — you never have to tell it to stop trusting last season.
    """
    attack: dict = field(default_factory=dict)
    defence: dict = field(default_factory=dict)
    games: dict = field(default_factory=dict)
    base_lambda: float = 1.40
    prior_weight_games: float = 6.0
    observed_xgf: dict = field(default_factory=dict)
    observed_xga: dict = field(default_factory=dict)
    unmatched_priors: list = field(default_factory=list)

    @classmethod
    def build(cls, bootstrap, fixtures, prior_att=None, prior_def=None,
              prior_weight_games=6.0, promoted_att=0.82, promoted_def=1.15):
        teams = {t["id"]: t for t in bootstrap["teams"]}
        elements = bootstrap["elements"]

        played = {tid: 0 for tid in teams}
        for f in fixtures:
            if f.get("finished") and f.get("team_h_score") is not None:
                played[f["team_h"]] += 1
                played[f["team_a"]] += 1

        # --- observed this season -------------------------------------------------
        # xGF: sum every player's expected_goals for the club (each shot has one owner)
        # xGA: minutes-weighted mean of expected_goals_conceded_per_90 over regulars
        xgf, xga = {}, {}
        for tid in teams:
            squad = [e for e in elements if e["team"] == tid]
            g = played[tid]
            if g == 0:
                xgf[tid] = None
                xga[tid] = None
                continue
            xgf[tid] = sum(float(e.get("expected_goals") or 0) for e in squad) / g
            regulars = [e for e in squad if int(e.get("minutes") or 0) >= 0.35 * 90 * g]
            num = sum(float(e.get("expected_goals_conceded_per_90") or 0) * int(e["minutes"])
                      for e in regulars)
            den = sum(int(e["minutes"]) for e in regulars)
            xga[tid] = (num / den) if den > 0 else None

        # --- prior ----------------------------------------------------------------
        # Default prior comes from FPL's own published team strength ratings, which
        # exist before a ball is kicked. If the caller supplies a measured prior
        # (e.g. last season's xG), that is used instead.
        prior_att = {canon_team(k): v for k, v in (prior_att or {}).items()}
        prior_def = {canon_team(k): v for k, v in (prior_def or {}).items()}

        sa = [t.get("strength_attack_home", 0) + t.get("strength_attack_away", 0)
              for t in teams.values()]
        sd = [t.get("strength_defence_home", 0) + t.get("strength_defence_away", 0)
              for t in teams.values()]
        mean_a = (sum(sa) / len(sa)) if sa else 0.0
        mean_d = (sum(sd) / len(sd)) if sd else 0.0
        # Before a season starts FPL leaves every strength rating at 0, so these
        # means are 0 and the ratings carry no information at all.
        fpl_strength_usable = mean_a > 0 and mean_d > 0

        pa, pd = {}, {}
        unmatched = []
        for tid, t in teams.items():
            key = canon_team(t.get("name"))
            alt = canon_team(t.get("short_name"))
            if key in prior_att:
                pa[tid] = prior_att[key]
                pd[tid] = prior_def.get(key, 1.0)
            elif alt in prior_att:
                pa[tid] = prior_att[alt]
                pd[tid] = prior_def.get(alt, 1.0)
            elif fpl_strength_usable:
                pa[tid] = (t["strength_attack_home"] + t["strength_attack_away"]) / mean_a
                d = (t["strength_defence_home"] + t["strength_defence_away"]) / mean_d
                # FPL's defence rating is "how good", we need "how leaky"
                pd[tid] = 2.0 - d if d < 2.0 else 0.5
                unmatched.append(t.get("name"))
            else:
                # No prior and no usable FPL rating: treat as exactly average
                # rather than crashing. The briefing warns when this happens.
                pa[tid] = 1.0
                pd[tid] = 1.0
                unmatched.append(t.get("name"))
        cls._last_unmatched = unmatched

        # --- league average, for normalising --------------------------------------
        obs = [v for v in xgf.values() if v is not None]
        base = (sum(obs) / len(obs)) if len(obs) >= 10 else 1.40

        att, dfn = {}, {}
        for tid in teams:
            g = played[tid]
            w = g / (g + prior_weight_games)
            o_a = (xgf[tid] / base) if xgf[tid] is not None else pa[tid]
            o_d = (xga[tid] / base) if xga[tid] is not None else pd[tid]
            att[tid] = w * o_a + (1 - w) * pa[tid]
            dfn[tid] = w * o_d + (1 - w) * pd[tid]

        # renormalise so the league averages 1.00 and the model neither invents
        # nor destroys goals overall
        ma = sum(att.values()) / len(att)
        md = sum(dfn.values()) / len(dfn)
        att = {k: v / ma for k, v in att.items()}
        dfn = {k: v / md for k, v in dfn.items()}

        return cls(attack=att, defence=dfn, games=played, base_lambda=base,
                   prior_weight_games=prior_weight_games,
                   observed_xgf=xgf, observed_xga=xga,
                   unmatched_priors=unmatched)

    def prior_share(self, tid) -> float:
        g = self.games.get(tid, 0)
        return self.prior_weight_games / (g + self.prior_weight_games)


# ===========================================================================
# 2. FIXTURES
# ===========================================================================
# ---------------------------------------------------------------------------
# Dixon-Coles low-score correction.
#
# Independent Poisson is a decent model of football scorelines except at 0-0,
# 1-0, 0-1 and 1-1, which occur more often than independence implies (teams
# shut up shop, games get cagey). Dixon & Coles (1997) add a correction term
# for exactly those four results.
#
# It matters here because clean-sheet probability is a first-order FPL quantity:
# defenders and keepers live on it. Uncorrected Poisson systematically
# under-rates 0-0 and therefore under-rates cheap defenders in tight fixtures.
# ---------------------------------------------------------------------------
DC_RHO = -0.035          # typical fitted value for the Premier League


def dc_tau(i, j, lam, mu, rho=DC_RHO):
    if i == 0 and j == 0:
        return 1 - lam * mu * rho
    if i == 0 and j == 1:
        return 1 + lam * rho
    if i == 1 and j == 0:
        return 1 + mu * rho
    if i == 1 and j == 1:
        return 1 - rho
    return 1.0


def clean_sheet_prob(lam_against, lam_for, rho=DC_RHO, cap=8):
    """
    P(opponent scores 0), Dixon-Coles corrected and renormalised.

    Plain Poisson would just be exp(-lam_against). The correction shifts mass
    toward 0-0 and 1-1, which raises clean-sheet odds slightly in low-scoring
    fixtures and lowers them in open ones.
    """
    tot = 0.0
    cs = 0.0
    for i in range(cap + 1):
        for j in range(cap + 1):
            p = (poisson_pmf(i, lam_for) * poisson_pmf(j, lam_against)
                 * dc_tau(i, j, lam_for, lam_against, rho))
            p = max(0.0, p)
            tot += p
            if j == 0:
                cs += p
    return (cs / tot) if tot > 0 else math.exp(-lam_against)


@dataclass
class FixtureModel:
    strength: TeamStrength
    home_mult: float = 1.10
    away_mult: float = 0.90
    use_dixon_coles: bool = True

    def project(self, home_id, away_id):
        s = self.strength
        lam = s.base_lambda
        # Home advantage is applied to attack and, in the opposite direction, to
        # the opponent's defence. A side is both more potent and harder to score
        # against at home, and splitting it this way avoids double-counting.
        ha = math.sqrt(self.home_mult)
        aa = math.sqrt(self.away_mult)
        h = lam * ha * s.attack[home_id] * (s.defence[away_id] / aa)
        a = lam * aa * s.attack[away_id] * (s.defence[home_id] / ha)
        return max(0.25, min(4.5, h)), max(0.25, min(4.5, a))

    def cs_prob(self, lam_against, lam_for):
        if self.use_dixon_coles:
            return clean_sheet_prob(lam_against, lam_for)
        return math.exp(-lam_against)

    def team_view(self, fixtures, gw):
        """{team_id: [ {opp, home, xg_for, xg_against, p_cs}, ... ]} for one gameweek.
        A list per team, because double gameweeks exist and blanks must show as []."""
        out = {tid: [] for tid in self.strength.attack}
        for f in fixtures:
            if f.get("event") != gw:
                continue
            h, a = f["team_h"], f["team_a"]
            xh, xa = self.project(h, a)
            out[h].append(dict(opp=a, home=True, xg_for=xh, xg_against=xa,
                               p_cs=self.cs_prob(xa, xh), fixture=f.get("id")))
            out[a].append(dict(opp=h, home=False, xg_for=xa, xg_against=xh,
                               p_cs=self.cs_prob(xh, xa), fixture=f.get("id")))
        return out


# ===========================================================================
# 3. PLAYERS
# ===========================================================================
@dataclass
class PlayerModel:
    strength: TeamStrength
    fixture_model: FixtureModel
    bonus_curve: tuple = BONUS_CURVE         # fitted logistic, see above
    calibration: dict = field(default_factory=dict)   # {pos: multiplier} from calibrate.py
    start_rates: dict = field(default_factory=dict)   # {element_id: recent start rate}
    place_share: dict = field(default_factory=dict)   # {element_id: share of his club's
                                                      # starting places, see
                                                      # calibrate_depth()}
    player_prior: dict = field(default_factory=dict)  # {code: last-season per-90 rates}
    prior_minutes_weight: float = 700.0               # pseudo-minutes of credit given
                                                      # to last season's rates

    # -- minutes ------------------------------------------------------------
    def raw_start_rate(self, e):
        """How often this player starts, before the depth chart is applied.

        Kept separate from minutes_profile because working out who is a club's
        first-choice keeper needs to compare team-mates, and that comparison
        must not recurse back through the availability maths.
        """
        mins = int(e.get("minutes") or 0)
        starts = int(e.get("starts") or 0)
        played_games = self.strength.games.get(e["team"], 0)
        recent = self.start_rates.get(e["id"])
        prior_p = self.player_prior.get(str(e.get("code")))

        if recent is not None:
            return recent                             # last-5 form, best signal
        if played_games > 0:
            observed = min(1.0, starts / played_games)
            # blend with last season until this season has a real sample
            if prior_p and prior_p.get("minutes"):
                prior_rate = min(1.0, prior_p["minutes"] / (38 * 90.0) * 1.25)
                w = played_games / (played_games + 4.0)
                rate = w * observed + (1 - w) * prior_rate
            else:
                rate = observed
            if mins == 0 and played_games >= 3:
                rate = min(rate, 0.15)                # named but never used = bench
            return rate

        # PRE-SEASON: nobody has kicked a ball, so `minutes` and `starts` on the
        # live element are still last season's totals - and crucially they were
        # earned at the club the player is at NOW. That makes them a better guide
        # to his role than the archive prior, which is blended across two seasons
        # and two clubs: it had Arrizabalaga down as a near-ever-present off his
        # Bournemouth campaign while the live feed plainly said one start.
        #
        # The archive is worse here for a second reason. seed.py drops any season
        # under MIN_MINUTES, so the very campaign that shows a player has become
        # a backup is the one thrown away, and only his starting years survive to
        # be averaged. That filter is right for per-90 rates, which are noise at
        # low minutes, and wrong for this question, where low minutes are the
        # entire signal.
        if starts > 0 or mins > 0:
            # Uplifted, for the same reason the archive branch below is. A share
            # of 38 counts every week he was hurt as a week he chose not to
            # start, but whether he is fit *now* is already priced separately in
            # p_avail - so charging last season's absences here bills the same
            # thing twice and quietly deflates every established starter.
            #
            # The uplift is smaller than the archive branch's because it is
            # applied to a share of STARTS. A share of minutes is also dragged
            # down by being substituted on the hour, which is real but belongs
            # to avg_start_mins below, not here.
            return min(1.0, starts / 38.0 * MISSED_GAME_UPLIFT)
        if prior_p and prior_p.get("minutes"):
            # No live history at all - a summer signing from abroad, say. The
            # archive is all there is.
            return min(1.0, prior_p["minutes"] / (38 * 90.0) * 1.25)
        return 0.5 if played_games == 0 else 0.0

    def start_rate(self, e):
        """`raw_start_rate` after the club depth chart, if one has been built."""
        if e["id"] in self.place_share:
            return self.place_share[e["id"]]
        return self.raw_start_rate(e)

    @staticmethod
    def _has_live_record(e):
        return int(e.get("minutes") or 0) > 0 or int(e.get("starts") or 0) > 0

    def _unrated_start_rate(self, e, peers):
        """
        A guess for someone with no record to go on, ranked by price.

        A flat 0.5 for every such player is the worst of both worlds: too high
        for the fringe squad member it usually describes, and identical for
        everyone, so a promoted club's entire squad came out as the same coin
        flip and the model had no way to tell its first eleven from its
        under-21s. FPL's own pricing does distinguish them - it is set from
        expected involvement - so it is used to order players the model
        otherwise knows nothing about. The club budget below then decides how
        many of them can start at once.
        """
        costs = sorted({int(p.get("now_cost") or 0) for p in peers})
        if len(costs) < 2:
            return 0.50
        pct = costs.index(int(e.get("now_cost") or 0)) / (len(costs) - 1)
        return 0.30 + 0.45 * pct

    @staticmethod
    def availability(e, gw_offset=0):
        """Chance this player is fit and permitted to feature, ignoring selection.

        `gw_offset` is how many gameweeks ahead this is being asked about. A flag
        describes the next match only, so further out it is relaxed toward full
        availability - see FLAG_RECOVERY.
        """
        status = e.get("status", "a")
        p_avail = AVAIL.get(status, 0.0)
        chance = e.get("chance_of_playing_next_round")
        if chance is not None:
            p_avail = min(p_avail if status != "a" else 1.0, chance / 100.0)
        if status != "a" and gw_offset > 0:
            back = flag_recovery(status, gw_offset)
            p_avail = p_avail + (1.0 - p_avail) * back
        return p_avail

    def calibrate_depth(self, elements):
        """
        Share out each club's starting places among its players.

        A club starts exactly one goalkeeper and exactly ten outfielders. Those
        are identities, not tendencies, so a club's start probabilities have to
        add up to them - and nothing in the per-player maths knew it. Pre-season
        Arsenal fielded Raya at 1.00, Arrizabalaga at 1.00 and Meslier at 0.73
        at once, and the optimiser quite reasonably bought the cheap "starter",
        because on the numbers in front of it he was one.

        Outfield was inflated the same way for a different reason: `starts` on
        the live element belongs to whichever club the player was at last
        season, so every summer signing brings his old club's record with him
        and both squads count him. Measured across the league that put the
        average club at nearly twelve starting outfielders.

        Places are handed out in order of how likely each player is to start, so
        established starters take what their own record supports and the rest
        divide only what is genuinely left. The players who lose out are the
        ones ranked below a full eleven, which is where the phantom starters
        were; a club whose rates already sum below its allowance is left alone,
        since scaling those up would be inventing evidence.

        The places are shared out in EXPECTED STARTS - each player's start rate
        multiplied by his chance of being fit - not in start rate alone. A man
        who is injured is not occupying a shirt: his absence is precisely what
        frees one for his deputy. Allocating on rate alone had Saliba and Timber,
        both ruled out with no chance of playing, holding 1.76 of Arsenal's ten
        places between them, while White and Mosquera - the two who actually
        deputise for them - were pushed to zero and the club ended up with one
        creditable fit defender.

        What is stored is converted back to a rate conditional on being fit,
        because minutes_profile multiplies by availability itself and would
        otherwise charge it twice.
        """
        by_club = {}
        for e in elements:
            by_club.setdefault((e["team"], e.get("element_type") == 1), []).append(e)
        share = {}
        for (_tid, is_gk), squad in by_club.items():
            unrated = [p for p in squad if not self._has_live_record(p)]
            rated = []
            for p in squad:
                r = (self.raw_start_rate(p) if self._has_live_record(p)
                     else self._unrated_start_rate(p, unrated))
                rated.append((r, self.availability(p), p))
            # rank on expected starts, so an unavailable man cannot outrank a fit one
            rated.sort(key=lambda t: (-(t[0] * t[1]), -int(t[2].get("now_cost") or 0)))
            left = float(GK_PLACES if is_gk else OUTFIELD_PLACES)
            for rate, avail, p in rated:
                take = max(0.0, min(rate * avail, left))
                share[p["id"]] = (take / avail) if avail > 0 else 0.0
                left -= take
        self.place_share = share
        return share

    def minutes_profile(self, e, gw_offset=0):
        """Returns (p_appear, p60, expected_minutes).

        This is the single biggest driver of FPL points and the place most
        models go wrong, so it is deliberately explicit.

        `gw_offset` is how many gameweeks ahead this is being asked about. A
        flag describes the next match only, so further out it is relaxed toward
        full availability - see FLAG_RECOVERY."""
        p_avail = self.availability(e, gw_offset)
        if p_avail <= 0:
            return 0.0, 0.0, 0.0

        mins = int(e.get("minutes") or 0)
        starts = int(e.get("starts") or 0)
        prior_p = self.player_prior.get(str(e.get("code")))

        start_rate = self.start_rate(e)

        p_start = p_avail * start_rate
        # An outfielder who does not start still often comes on. A goalkeeper who
        # does not start almost never does - it takes an injury or a red card to
        # the man in front of him. Charging backup keepers the outfield rate left
        # a No 2 "appearing" nearly half the time, which is where a good part of
        # a cheap keeper's projection was quietly coming from.
        sub_rate = SUB_RATE_GK if e.get("element_type") == 1 else SUB_RATE_OUTFIELD
        p_sub = p_avail * (1 - start_rate) * sub_rate
        p_appear = min(1.0, p_start + p_sub)
        if starts > 0:
            # same trap as the seed file: `mins` includes substitute appearances,
            # so dividing by starts alone overstates how long a start lasts
            avg_start_mins = min(90.0, mins / starts)
        elif prior_p and prior_p.get("mins_per_start"):
            avg_start_mins = prior_p["mins_per_start"]
        else:
            avg_start_mins = 80.0
        avg_start_mins = max(45.0, min(90.0, avg_start_mins))
        p60 = p_start * (0.90 if avg_start_mins >= 70 else 0.55) + p_sub * 0.08
        exp_min = p_start * avg_start_mins + p_sub * 20.0
        return p_appear, p60, exp_min

    # -- shrunk per-90 rates -------------------------------------------------
    @staticmethod
    def _shrink(raw_total, minutes, prior90, k=PRIOR_MINUTES):
        """
        Blend an observed total with a prior rate, weighted by minutes played.

        `minutes <= 0` is not "no data tending to zero" - it means the total
        describes no football at all, and dividing by it credits a whole season
        of actions to nobody's time on the pitch. The FPL feed does emit this:
        players appear with counting stats carried over and minutes reset, which
        sent one midfielder to 37 defensive actions per 90 against a league best
        near 15. With no minutes the prior is all there is.
        """
        if minutes <= 0:
            return prior90
        return ((raw_total * 90) + (prior90 * k)) / (minutes + k)

    def _prior_for(self, e, key, positional):
        """Last season's own rate if we have enough of it, else the positional mean.

        The player prior is itself shrunk toward the positional mean by how many
        minutes it was built on, so a 500-minute sample from last season does not
        get treated like a 3,000-minute one."""
        p = self.player_prior.get(str(e.get("code")))
        if not p:
            return positional
        # Confidence rests on every minute behind the rate, across seasons.
        # `minutes` is deliberately the per-season figure and belongs to the
        # start-rate question, not this one.
        m = p.get("minutes_total") or p.get("minutes", 0)
        if m <= 0:
            return positional
        w = m / (m + PRIOR_MINUTES)
        return w * p.get(key, positional) + (1 - w) * positional

    def rates(self, e):
        pos = e["element_type"]
        mins = int(e.get("minutes") or 0)
        xg = float(e.get("expected_goals") or 0)
        xa = float(e.get("expected_assists") or 0)
        cbit = (float(e.get("clearances_blocks_interceptions") or 0)
                + float(e.get("tackles") or 0))
        if pos in (3, 4):
            cbit += float(e.get("recoveries") or 0)
        bps = float(e.get("bps") or 0)
        K = self.prior_minutes_weight if self.player_prior else PRIOR_MINUTES
        saves_now = float(e.get("saves_per_90") or 0)
        if mins < 180:
            saves_now = self._prior_for(e, "saves90", saves_now)
        return dict(
            xg90=self._shrink(xg, mins, self._prior_for(e, "xg90", PRIOR_XG90[pos]), K),
            xa90=self._shrink(xa, mins, self._prior_for(e, "xa90", PRIOR_XA90[pos]), K),
            dc90=self._shrink(cbit, mins, self._prior_for(e, "dc90", PRIOR_DEFCON90[pos]), K),
            bps90=self._shrink(bps, mins, self._prior_for(e, "bps90", PRIOR_BPS90[pos]), K),
            saves90=saves_now,
        )

    def expected_bonus(self, bps90, exp_min):
        L, k, x0 = self.bonus_curve
        return _logistic(bps90, L, k, x0) * min(1.0, exp_min / 90.0)

    # -- the main event ------------------------------------------------------
    def project(self, e, fixtures_for_team, gw_offset=0):
        """Expected FPL points for one player in one gameweek.
        `fixtures_for_team` is a list, so doubles add and blanks return 0.
        `gw_offset` releases injury/suspension flags further out - see above."""
        pos = e["element_type"]
        p_appear, p60, exp_min = self.minutes_profile(e, gw_offset)
        if p_appear <= 0 or not fixtures_for_team:
            return dict(ep=0.0, p_appear=p_appear, p60=p60, exp_min=0.0,
                        parts={}, n_fix=len(fixtures_for_team))

        r = self.rates(e)
        pen = penalty_uplift(e)
        spa = setpiece_uplift(e)
        total = 0.0
        parts = dict(appearance=0.0, goals=0.0, assists=0.0, cs=0.0,
                     conceded=0.0, saves=0.0, defcon=0.0, bonus=0.0)

        for fx in fixtures_for_team:
            att_mult = fx["xg_for"] / self.strength.base_lambda
            mins_share = exp_min / 90.0

            parts["appearance"] += p60 * 2 + (p_appear - p60) * 1
            parts["goals"] += (r["xg90"] + pen) * mins_share * att_mult * GOAL_PTS[pos]
            parts["assists"] += (r["xa90"] + spa) * mins_share * att_mult * ASSIST_PTS

            if pos in (1, 2, 3):
                parts["cs"] += p60 * fx["p_cs"] * CS_PTS[pos]
            if pos in (1, 2):
                parts["conceded"] -= expected_conceded_penalty(fx["xg_against"]) * p60
            if pos == 1:
                # saves scale with how much shooting the opponent does
                scale = fx["xg_against"] / max(0.4, self.strength.base_lambda)
                parts["saves"] += (r["saves90"] * mins_share * scale) / SAVES_PER_POINT

            if pos in (2, 3, 4):
                # Availability belongs in this term exactly once.
                #
                # `mins_share` is expected minutes over 90, and expected minutes
                # already fold in the chance of not featuring at all - so
                # multiplying the result by p_appear again charged the same
                # discount twice, and only for defensive contribution. Goals and
                # assists above apply it once, through mins_share alone.
                #
                # The simulator gets this right by construction: it draws whether
                # the player featured, then uses the minutes he actually played.
                # Mirroring that here means conditioning on having played, which
                # is expected minutes divided by the chance of playing.
                played_share = mins_share / p_appear if p_appear > 0 else 0.0
                p_hit = defcon_probability(pos, r["dc90"], played_share)
                parts["defcon"] += p_hit * DEFCON_PTS * p_appear

            parts["bonus"] += self.expected_bonus(r["bps90"], exp_min) * att_mult ** 0.5

        total = sum(parts.values())
        total *= self.calibration.get(POS_NAME[pos], 1.0)
        return dict(ep=max(0.0, total), p_appear=p_appear, p60=p60, exp_min=exp_min,
                    parts=parts, rates=r, n_fix=len(fixtures_for_team))

    def project_horizon(self, e, team_views, gws):
        """EP for each gameweek in `gws`. team_views = {gw: fixture_model.team_view(...)}"""
        first = gws[0] if gws else 0
        return {gw: self.project(e, team_views[gw].get(e["team"], []),
                                 gw_offset=gw - first)["ep"] for gw in gws}
