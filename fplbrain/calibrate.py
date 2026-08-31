"""
Self-calibration — the part that makes this "evolve" rather than just "update".

Every run writes its predictions to data/predictions.json. Once a gameweek
finishes, `calibrate` pulls the actual scores, measures how wrong the model was,
and stores a per-position correction multiplier that the next run applies.

It is deliberately conservative:
  - corrections are damped (only a fraction of the measured bias is applied)
  - they are clamped to [0.75, 1.30] so one freak gameweek cannot wreck the model
  - only players who were actually projected to play are scored, because
    including 600 zeros would make the error look artificially good

It also records rank correlation, which matters more than absolute error. FPL is
a ranking problem: you do not need to know Haaland will score 8.3, you need to
know he will beat the alternative.
"""
from __future__ import annotations
import json, os, statistics

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_FILE = os.path.join(HERE, "data", "predictions.json")
CAL_FILE = os.path.join(HERE, "data", "calibration.json")

DAMPING = 0.35          # apply only this share of the measured bias
CLAMP = (0.75, 1.30)
MIN_SAMPLE = 25         # per position, before we trust a correction


def _load(path, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


GIST_NAME = "fpl_brain_calibration.json"
PRED_GIST_NAME = "fpl_brain_predictions.json"

# How many gameweeks of predictions to keep. This file exists purely so a
# finished gameweek can be graded once, and once that grading has happened the
# entry is dead weight - kept trimmed so the mirrored copy stays small rather
# than accumulating a whole season.
PRED_KEEP_GWS = 6


def _save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    # What the model has predicted or learned is worth more than the disk it
    # sits on: a hosted box wipes this file on every redeploy, and predictions
    # logged before a finished gameweek is graded are otherwise gone before
    # score_gameweek() ever gets to see them - which is exactly what happened
    # to GW1 here, silently, for weeks, with calibration never noticing there
    # was nothing left to grade.
    gist_name = {CAL_FILE: GIST_NAME, PRED_FILE: PRED_GIST_NAME}.get(path)
    if gist_name:
        try:
            from . import squad
            squad.sync_extra_push(gist_name, obj)
        except Exception:
            pass


def load_calibration():
    """
    Multipliers and the history behind them.

    On a hosted box the file is gone after every redeploy, so an empty local
    copy usually means the disk was wiped rather than that nothing has been
    learned. Pull the mirrored copy back before concluding the model is naive.
    """
    c = _load(CAL_FILE, {})
    if not c.get("multipliers"):
        try:
            from . import squad
            remote = squad.sync_extra_pull(GIST_NAME)
            if remote and remote.get("multipliers"):
                _save(CAL_FILE, remote)
                c = remote
        except Exception:
            pass
    return c.get("multipliers", {"GK": 1.0, "DEF": 1.0, "MID": 1.0, "FWD": 1.0}), c


def _load_predictions():
    """Local copy if there is one, else the mirrored copy - same reasoning as
    load_calibration(): empty here usually means wiped, not never-written."""
    preds = _load(PRED_FILE, {})
    if not preds:
        try:
            from . import squad
            remote = squad.sync_extra_pull(PRED_GIST_NAME)
            if remote:
                preds = remote
        except Exception:
            pass
    return preds


def log_predictions(gw, rows):
    """rows: [{id, name, pos, ep, price, p_appear}]"""
    preds = _load_predictions()
    preds[str(gw)] = {str(r["id"]): dict(name=r["name"], pos=r["pos"], ep=round(r["ep"], 3),
                                         p_appear=round(r.get("p_appear", 0), 3))
                      for r in rows}
    # Trim to the most recent gameweeks by numeric id, not insertion order -
    # dict order does not track gameweek order once a value has been
    # overwritten, which a plain slice would get wrong.
    keep = sorted(preds, key=lambda k: int(k))[-PRED_KEEP_GWS:]
    preds = {k: preds[k] for k in keep}
    _save(PRED_FILE, preds)


def _spearman(a, b):
    n = len(a)
    if n < 3:
        return 0.0
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else 0.0


def score_gameweek(gw, actual_points: dict, min_p_appear=0.30):
    """
    actual_points : {element_id: points scored in that gameweek}
    Returns a report dict and updates the stored calibration multipliers.
    """
    preds = _load_predictions().get(str(gw))
    if not preds:
        return dict(error=f"No logged predictions for GW{gw}. Run `update --gw {gw}` first.")

    by_pos = {}
    for pid, p in preds.items():
        if p["p_appear"] < min_p_appear:
            continue
        act = actual_points.get(int(pid))
        if act is None:
            continue
        by_pos.setdefault(p["pos"], []).append((p["ep"], float(act), p["name"]))

    mult, state = load_calibration()
    report = {"gw": gw, "positions": {}, "overall": {}}
    all_pred, all_act = [], []

    for pos, rows in by_pos.items():
        pr = [r[0] for r in rows]
        ac = [r[1] for r in rows]
        all_pred += pr
        all_act += ac
        mae = statistics.fmean(abs(p - a) for p, a in zip(pr, ac))
        bias = statistics.fmean(a - p for p, a in zip(pr, ac))
        rho = _spearman(pr, ac)
        ratio = (sum(ac) / sum(pr)) if sum(pr) > 0 else 1.0

        old = mult.get(pos, 1.0)
        if len(rows) >= MIN_SAMPLE:
            target = old * ratio
            new = old + DAMPING * (target - old)
            new = max(CLAMP[0], min(CLAMP[1], new))
        else:
            new = old
        mult[pos] = round(new, 4)
        report["positions"][pos] = dict(n=len(rows), mae=round(mae, 3), bias=round(bias, 3),
                                        spearman=round(rho, 3), old_mult=round(old, 3),
                                        new_mult=round(new, 3))

    if all_pred:
        report["overall"] = dict(
            n=len(all_pred),
            mae=round(statistics.fmean(abs(p - a) for p, a in zip(all_pred, all_act)), 3),
            bias=round(statistics.fmean(a - p for p, a in zip(all_pred, all_act)), 3),
            spearman=round(_spearman(all_pred, all_act), 3),
        )

    state["multipliers"] = mult
    state.setdefault("history", []).append(report)
    _save(CAL_FILE, state)
    return report


def calibration_history():
    return _load(CAL_FILE, {}).get("history", [])
