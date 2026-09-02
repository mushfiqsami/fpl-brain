"""
Monte Carlo points engine.

Why this exists: expected points is a mean, and FPL is not won by means.

Captaincy doubles one score, which makes it a convex payoff — you want the fattest
right tail, not the best average. A 5.2 EP striker who returns 2 or 15 is worth
more as a captain than a 5.6 EP defender who returns 4 or 7 every week. Ranking on
EP alone cannot see that, because the two look almost identical.

So instead of computing a mean, we simulate each player's gameweek thousands of
times and keep the whole distribution. That gives:

    mean        - same number as before, for comparison
    median      - what usually happens
    p90         - the ceiling. Captaincy metric.
    p_haul      - P(>= 10 pts). Differential metric.
    p_blank     - P(<= 2 pts). Risk metric.
    sd          - spread, for rank-aware decisions

Everything is drawn from the same fitted components the deterministic model uses,
so the mean of the simulation should track the analytic EP closely. That agreement
is asserted in the tests, because a simulator that silently disagrees with its own
model is worse than no simulator.
"""
from __future__ import annotations
import math, random

from .model import (GOAL_PTS, CS_PTS, ASSIST_PTS, DEFCON_PTS, POS_NAME,
                    defcon_probability, _logistic, BONUS_CURVE)

# Bonus is awarded to the top 3 BPS in a match. Rather than model the whole
# match's BPS race, we use the fitted expected-bonus curve as a probability of
# landing a bonus band, then draw which band.
BONUS_SPLIT = (0.48, 0.32, 0.20)      # given any bonus, P(1), P(2), P(3)

YELLOW_RATE_90 = 0.11                  # league-wide baseline, -1 pt
RED_RATE_90 = 0.004                    # -3 pts


def _pois(lam: float, rng: random.Random) -> int:
    """Knuth for small lambda; normal approximation above 12 (never hit here)."""
    if lam <= 0:
        return 0
    if lam > 12:
        return max(0, int(rng.gauss(lam, math.sqrt(lam)) + 0.5))
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1
        if k > 30:
            return k


class PlayerSim:
    """Simulates one player's gameweek points distribution."""

    __slots__ = ("pos", "rates", "p_start", "p_sub", "avg_start_mins",
                 "fixtures", "base_lambda", "pen_share", "sp_share", "calibration")

    def __init__(self, pos, rates, p_start, p_sub, avg_start_mins,
                 fixtures, base_lambda, pen_share=0.0, sp_share=0.0, calibration=1.0):
        self.pos = pos
        self.rates = rates
        self.p_start = p_start
        self.p_sub = p_sub
        self.avg_start_mins = avg_start_mins
        self.fixtures = fixtures          # list of dicts: xg_for, xg_against
        self.base_lambda = base_lambda
        self.pen_share = pen_share        # extra xG from being the penalty taker
        self.sp_share = sp_share          # extra xA from set pieces
        # The same per-position multiplier PlayerModel.project() applies to its
        # total. Without it the simulated distribution silently stopped
        # tracking the analytic EP the moment calibrate.py started returning
        # anything other than 1.0 for every position - captain_value, ceiling,
        # floor and p_haul all quietly went back to running on the model's
        # pre-calibration numbers while the headline EP used the corrected
        # ones, and the two disagreed by exactly the calibration gap.
        self.calibration = calibration

    def one(self, rng: random.Random) -> int:
        return self.draw(rng)[0]

    def draw(self, rng: random.Random):
        """
        (points, played) for one gameweek.

        Whether the player featured at all is returned alongside the score
        because a team simulation needs it: FPL brings a bench player on for
        anyone who does not play, and a squad that cannot see who blanked cannot
        model that. Scoring zero and not turning up look identical in the total.
        """
        total = 0
        played = False
        for fx in self.fixtures:
            r = rng.random()
            if r < self.p_start:
                mins = self.avg_start_mins * rng.uniform(0.80, 1.0)
                started = True
            elif r < self.p_start + self.p_sub:
                mins = rng.uniform(5, 32)
                started = False
            else:
                continue                                  # did not feature

            played = True
            share = mins / 90.0
            att = fx["xg_for"] / self.base_lambda
            total += 2 if mins >= 60 else 1

            # --- goals ---
            lam_g = (self.rates["xg90"] + self.pen_share) * share * att
            g = _pois(lam_g, rng)
            total += g * GOAL_PTS[self.pos]

            # --- assists ---
            lam_a = (self.rates["xa90"] + self.sp_share) * share * att
            a = _pois(lam_a, rng)
            total += a * ASSIST_PTS

            # --- clean sheet / conceded ---
            if self.pos in (1, 2, 3) and mins >= 60:
                conceded = _pois(fx["xg_against"], rng)
                if conceded == 0:
                    total += CS_PTS[self.pos]
                elif self.pos in (1, 2):
                    total -= conceded // 2
            elif self.pos in (1, 2) and mins > 0:
                conceded = _pois(fx["xg_against"] * share, rng)
                total -= conceded // 2

            # --- saves ---
            if self.pos == 1:
                scale = fx["xg_against"] / max(0.4, self.base_lambda)
                saves = _pois(self.rates.get("saves90", 0) * share * scale, rng)
                total += saves // 3

            # --- defensive contribution ---
            if self.pos in (2, 3, 4):
                if rng.random() < defcon_probability(self.pos, self.rates["dc90"], share):
                    total += DEFCON_PTS

            # --- bonus ---
            L, k, x0 = BONUS_CURVE
            p_any = _logistic(self.rates["bps90"], L, k, x0) * min(1.0, share)
            # attacking returns hugely raise BPS, so condition on what happened
            p_any = min(0.85, p_any * (1.0 + 0.9 * g + 0.5 * a))
            if rng.random() < p_any:
                u = rng.random()
                total += 1 if u < BONUS_SPLIT[0] else (2 if u < BONUS_SPLIT[0] + BONUS_SPLIT[1] else 3)

            # --- cards ---
            if rng.random() < YELLOW_RATE_90 * share:
                total -= 1
            if rng.random() < RED_RATE_90 * share:
                total -= 3

        return total, played

    def run(self, n=4000, seed=None):
        rng = random.Random(seed)
        # Scaled here, not after, so mean/median/percentiles AND the p_haul /
        # p_big / p_blank threshold crossings are all computed on the same
        # calibrated numbers the rest of the app reports - those thresholds are
        # nonlinear, so rescaling the summary stats afterward would not have
        # produced the same p_haul a rescaled sample does.
        out = [self.one(rng) * self.calibration for _ in range(n)]
        out.sort()
        m = sum(out) / n
        var = sum((x - m) ** 2 for x in out) / n
        return dict(
            mean=m,
            median=out[n // 2],
            p90=out[int(n * 0.90)],
            p10=out[int(n * 0.10)],
            sd=math.sqrt(var),
            p_haul=sum(1 for x in out if x >= 10) / n,
            p_big=sum(1 for x in out if x >= 15) / n,
            p_blank=sum(1 for x in out if x <= 2) / n,
            samples=out,
        )


def captain_value(dist_a, n=4000):
    """
    Expected value of captaining, which is just 2x the mean — so on its own it
    cannot separate two players with the same mean. What actually differs is the
    variance you are buying, so we return a blended score.

    score = mean + LAMBDA * (p90 - mean)

    LAMBDA controls appetite for ceiling over consistency. At 0 this reduces to
    ranking on mean; at 0.5 a player with a fat tail overtakes a steady one with
    a slightly better average. 0.35 is deliberately moderate.
    """
    LAMBDA = 0.35
    return dist_a["mean"] + LAMBDA * (dist_a["p90"] - dist_a["mean"])
