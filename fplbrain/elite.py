"""
What the best managers in the world are actually holding.

Every other number in this project is the model's own opinion. That is a closed
loop: if the projection is wrong about a player, nothing else here disagrees
with it, and a model with no outside reference cannot tell being wrong from
being early. This is the one input that does not come from our own maths.

It is deliberately NOT wired into the projection. Copying the top ten would
just be a slower way of owning the template, and the template is by definition
what everyone already has - you cannot climb past 10 million people by holding
what they hold. What it is good for is the opposite: a short, specific list of
places where the model and the best managers in the game disagree, so the
disagreement can be looked at rather than discovered six gameweeks later.

Two numbers per player, and they answer different questions:

  elite_owned   how many of the top N hold him. High + we do not own him is a
                blind spot worth explaining.
  captaincy     how many of them captained him this week. Captaincy is a much
                sharper signal than ownership, because it is the one pick a
                manager cannot hedge - it is where conviction actually shows.
"""
from __future__ import annotations


def top_entries(cl, n=10, league_id=314, force=False):
    """
    The current top `n` managers overall: [{rank, entry, name, team, total}].

    Page 1 of league 314 returns 50, so anything up to 50 costs one request.
    """
    data = cl.league_standings(league_id, 1, force=force)
    rows = (data.get("standings") or {}).get("results") or []
    return [dict(rank=r.get("rank"), entry=r.get("entry"),
                 name=r.get("player_name"), team=r.get("entry_name"),
                 total=r.get("total"))
            for r in rows[:n]]


def holdings(cl, entries, gw, force=False):
    """
    {element_id: {"owned": k, "captained": k}} across the given managers.

    A manager whose picks cannot be read (a private entry, a network blip) is
    skipped rather than allowed to fail the whole thing - a partial reading of
    nine of ten managers is still worth having, and pretending otherwise would
    mean losing the feature entirely to one bad row.
    """
    counts, seen = {}, 0
    for e in entries:
        try:
            picks = cl.entry_picks(int(e["entry"]), gw, force=force)
        except Exception:
            continue
        seen += 1
        for p in picks.get("picks") or []:
            slot = counts.setdefault(p["element"], {"owned": 0, "captained": 0})
            slot["owned"] += 1
            if p.get("is_captain"):
                slot["captained"] += 1
    return counts, seen


def disagreements(counts, seen, mine, by_id, ep_total, limit=8):
    """
    Where the elite and this squad differ, most owned first.

    `mine` is the set of element ids currently owned. Returns two lists:

      missing  they hold him, we do not. Sorted by how many of them do, because
               nine of ten holding a player is a different claim from three.
      fading   we hold him, almost none of them do. This is the weaker signal of
               the two and is presented as such - a low-owned player is not
               automatically a mistake, it is just the place where this squad is
               taking a position the best managers are not.
    """
    missing, fading = [], []
    for pid, c in counts.items():
        if pid in mine or pid not in by_id:
            continue
        p = by_id[pid]
        missing.append(dict(id=pid, name=p["name"], club=p["club"],
                            pos=p["pos"], price=p["price"],
                            owned=c["owned"], captained=c["captained"],
                            share=round(c["owned"] / max(1, seen), 2),
                            ep=round(ep_total.get(pid, 0.0), 2)))
    for pid in mine:
        if pid not in by_id:
            continue
        c = counts.get(pid, {"owned": 0, "captained": 0})
        if c["owned"] * 2 > seen:          # over half of them hold him too
            continue
        p = by_id[pid]
        fading.append(dict(id=pid, name=p["name"], club=p["club"],
                           pos=p["pos"], price=p["price"],
                           owned=c["owned"], captained=c["captained"],
                           share=round(c["owned"] / max(1, seen), 2),
                           ep=round(ep_total.get(pid, 0.0), 2)))
    missing.sort(key=lambda r: (-r["owned"], -r["ep"]))
    fading.sort(key=lambda r: (r["owned"], r["ep"]))
    return missing[:limit], fading[:limit]
