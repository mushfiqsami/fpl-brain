"""
Your actual squad.

FPL does not publish anyone's picks until the first deadline has passed, and even
after that the API only exposes last gameweek's team. So the dashboard keeps its
own copy of your 15, stored in data/my_squad.json.

Players are stored as name + club rather than as numeric IDs, because FPL element
IDs are reassigned every season while names are not. They get resolved against the
live player list on each run, so a stored squad survives a season rollover and
tells you loudly if someone can no longer be found.
"""
from __future__ import annotations
import json, os, threading, unicodedata, urllib.request, urllib.error

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQUAD_FILE = os.path.join(HERE, "data", "my_squad.json")        # legacy single squad
TEAMS_FILE = os.path.join(HERE, "data", "my_teams.json")        # multi-team store

# ---------------------------------------------------------------------------
# Optional cross-device sync, via a private GitHub Gist.
#
# The hosted filesystem is wiped on every sleep and redeploy, and a browser's
# localStorage only ever knows about the one device that wrote it. Neither can
# make a change on the phone show up on the laptop. A Gist can: it is free,
# private, needs no server, and the token stays in the environment.
#
# Set FPL_SYNC_TOKEN to a GitHub token with the `gist` scope to switch this on.
# With it unset every function here is a no-op and the app behaves exactly as
# before, which is why nothing below is allowed to raise.
# ---------------------------------------------------------------------------
GIST_FILENAME = "fpl_brain_teams.json"
_gist_id = None
_sync_lock = threading.Lock()


def sync_enabled() -> bool:
    return bool(os.environ.get("FPL_SYNC_TOKEN"))


def _gh(url, token, data=None, method=None):
    req = urllib.request.Request(
        url, data=(json.dumps(data).encode() if data is not None else None),
        method=method, headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "fplbrain-sync",
            "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=8) as r:
        return json.loads(r.read().decode())


def _find_gist(token):
    """Locate our gist, or make one. Cached, because the id is not persisted."""
    global _gist_id
    if _gist_id:
        return _gist_id
    forced = os.environ.get("FPL_SYNC_GIST_ID")
    if forced:
        _gist_id = forced
        return _gist_id
    for g in _gh("https://api.github.com/gists?per_page=100", token):
        if GIST_FILENAME in (g.get("files") or {}):
            _gist_id = g["id"]
            return _gist_id
    made = _gh("https://api.github.com/gists", token, data={
        "description": "FPL Brain saved teams", "public": False,
        "files": {GIST_FILENAME: {"content": json.dumps({"active": None, "teams": []})}}},
        method="POST")
    _gist_id = made["id"]
    return _gist_id


def sync_pull():
    """Teams as stored remotely, or None if unavailable."""
    token = os.environ.get("FPL_SYNC_TOKEN")
    if not token:
        return None
    try:
        g = _gh(f"https://api.github.com/gists/{_find_gist(token)}", token)
        raw = (g.get("files") or {}).get(GIST_FILENAME, {}).get("content")
        d = json.loads(raw) if raw else None
        return d if d and d.get("teams") else None
    except Exception:
        return None          # offline, bad token, rate limited - never fatal


def sync_push(data):
    """Mirror the store upward. Fire-and-forget: saving must never wait on GitHub."""
    token = os.environ.get("FPL_SYNC_TOKEN")
    if not token:
        return

    def go():
        with _sync_lock:
            try:
                _gh(f"https://api.github.com/gists/{_find_gist(token)}", token,
                    data={"files": {GIST_FILENAME: {"content": json.dumps(data, indent=1)}}},
                    method="PATCH")
            except Exception:
                pass

    threading.Thread(target=go, daemon=True).start()


def sync_extra_pull(filename):
    """
    Read one more file out of the same gist.

    Calibration is the model's memory of its own errors and it accumulates over
    a season - which makes it exactly the wrong thing to keep on a disk that is
    wiped every redeploy. Teams already ride in the gist; this lets anything
    else that must outlive the filesystem do the same.
    """
    token = os.environ.get("FPL_SYNC_TOKEN")
    if not token:
        return None
    try:
        g = _gh(f"https://api.github.com/gists/{_find_gist(token)}", token)
        raw = (g.get("files") or {}).get(filename, {}).get("content")
        return json.loads(raw) if raw else None
    except Exception:
        return None


def sync_extra_push(filename, data):
    """Mirror one more file upward, fire-and-forget."""
    token = os.environ.get("FPL_SYNC_TOKEN")
    if not token:
        return

    def go():
        with _sync_lock:
            try:
                _gh(f"https://api.github.com/gists/{_find_gist(token)}", token,
                    data={"files": {filename: {"content": json.dumps(data)}}},
                    method="PATCH")
            except Exception:
                pass

    threading.Thread(target=go, daemon=True).start()


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s))
                   if unicodedata.category(c) != "Mn")


def norm(s: str) -> str:
    s = strip_accents(s).lower()
    for ch in ".'-_":
        s = s.replace(ch, "")
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# Multi-team store
#
# Shape: {"active": "<id>", "teams": [{id, name, players[], bank, free_transfers}]}
#
# Several teams are common: a main side, a mini-league side, a cup side. Each is
# optimised independently, because the right transfer for one is regularly wrong
# for another - they have different squads, banks and transfer counts.
# ---------------------------------------------------------------------------

def _read(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_local(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)          # atomic, so a crash mid-write cannot corrupt


def _write(path, obj):
    """Save, and mirror upward. Use _write_local to land a copy pulled FROM the
    remote, so a fetch does not bounce straight back as a write."""
    _write_local(path, obj)
    if path == TEAMS_FILE:
        sync_push(obj)


def _new_id(existing):
    n = 1
    ids = {t.get("id") for t in existing}
    while f"t{n}" in ids:
        n += 1
    return f"t{n}"


def load_all():
    """Every team, plus which one is selected. Migrates a legacy single squad."""
    d = _read(TEAMS_FILE)
    if d and d.get("teams"):
        # guard against a stale active id
        ids = [t["id"] for t in d["teams"]]
        if d.get("active") not in ids:
            d["active"] = ids[0]
        return d

    # Nothing on disk. On a hosted box that usually means the filesystem was
    # wiped rather than that the user has no teams, so try the remote copy
    # before falling back to the legacy single squad.
    remote = sync_pull()
    if remote:
        ids = [t["id"] for t in remote["teams"]]
        if remote.get("active") not in ids:
            remote["active"] = ids[0]
        _write_local(TEAMS_FILE, remote)
        return remote

    legacy = _read(SQUAD_FILE)
    if legacy and legacy.get("players"):
        d = {"active": "t1", "teams": [dict(
            id="t1", name="My team", players=legacy["players"],
            bank=legacy.get("bank", 0.0),
            free_transfers=legacy.get("free_transfers", 1))]}
        _write(TEAMS_FILE, d)
        return d
    return {"active": None, "teams": []}


def get_active():
    d = load_all()
    for t in d["teams"]:
        if t["id"] == d.get("active"):
            return t
    return d["teams"][0] if d["teams"] else None


def set_active(team_id):
    d = load_all()
    if any(t["id"] == team_id for t in d["teams"]):
        d["active"] = team_id
        _write(TEAMS_FILE, d)
    return d


def upsert_team(team_id, name, players, bank=0.0, free_transfers=1, make_active=True,
                formation=None, entry_id=None):
    """
    formation: "4-4-2" etc, or "auto" to let the model pick the shape. None means
    "leave it alone" - callers like rename must not silently reset it.
    Each player may carry role="start"/"bench" to pin them into or out of the XI.

    entry_id: the team's numeric FPL id (visible in the URL of its points page),
    or "" to explicitly clear it back to manual entry. None means "leave it
    alone", same convention as formation - a save that does not mention it must
    not silently unlink a team that was already syncing. When set, resolve_team()
    below pulls this team's real picks from the FPL API on every run instead of
    matching the typed name/club rows, which is the whole point of setting it:
    one edit in the official app is what future runs then see, not a second
    typed copy that can drift out of sync with it.
    """
    d = load_all()
    players = [p for p in (players or []) if (p.get("name") or "").strip()]
    if team_id:
        for t in d["teams"]:
            if t["id"] == team_id:
                eid = t.get("entry_id") if entry_id is None else (entry_id or None)
                t.update(name=name or t.get("name") or "My team", players=players,
                         bank=float(bank), free_transfers=int(free_transfers),
                         formation=formation or t.get("formation") or "auto",
                         entry_id=eid)
                break
        else:
            team_id = None
    if not team_id:
        team_id = _new_id(d["teams"])
        d["teams"].append(dict(id=team_id, name=name or f"Team {len(d['teams']) + 1}",
                               players=players, bank=float(bank),
                               free_transfers=int(free_transfers),
                               formation=formation or "auto",
                               entry_id=(entry_id or None)))
    if make_active or not d.get("active"):
        d["active"] = team_id
    _write(TEAMS_FILE, d)
    return d


def restore_all(data):
    """
    Overwrite the team store from a client-supplied snapshot.

    Hosted storage is ephemeral (Render wipes the disk on sleep/redeploy), so the
    browser keeps its own copy in localStorage and posts it back here whenever the
    server wakes up with nothing on disk. Only called when the server side is
    already empty, so this can never clobber a fresher server-side edit.
    """
    teams = []
    for t in (data or {}).get("teams") or []:
        players = [p for p in (t.get("players") or []) if (p.get("name") or "").strip()]
        entry_id = t.get("entry_id") or None
        # A live-synced team legitimately has no typed rows - only a team with
        # neither is actually empty and worth dropping from the restore.
        if not players and not entry_id:
            continue
        teams.append(dict(id=t.get("id") or _new_id(teams), name=t.get("name") or "Team",
                          players=players, bank=float(t.get("bank", 0.0)),
                          free_transfers=int(t.get("free_transfers", 1)),
                          formation=t.get("formation") or "auto",
                          entry_id=entry_id))
    if not teams:
        return load_all()
    ids = [t["id"] for t in teams]
    active = data.get("active") if data.get("active") in ids else ids[0]
    d = {"active": active, "teams": teams}
    _write(TEAMS_FILE, d)
    return d


def apply_to_all(formation=None, bank=None, free_transfers=None, roles=None):
    """
    Push a setting to every team at once.

    Only one person uses this app, so a decision made on one team is usually
    meant for all of them; editing five squads by hand to say the same thing is
    just a chance to get one of them wrong. Each field is optional and only the
    ones supplied are touched.

    roles: {"<name>|<club>": "start"/"bench"/""} matched case-insensitively, so a
    player pinned in one team is pinned wherever else he is owned. Teams that do
    not own him are simply unaffected.
    """
    d = load_all()
    for t in d["teams"]:
        if formation is not None:
            t["formation"] = formation or "auto"
        if bank is not None:
            t["bank"] = float(bank)
        if free_transfers is not None:
            t["free_transfers"] = int(free_transfers)
        if roles:
            for p in t.get("players") or []:
                key = f"{norm(p.get('name'))}|{norm(p.get('club'))}"
                if key in roles:
                    r = roles[key]
                    if r:
                        p["role"] = r
                    else:
                        p.pop("role", None)
    _write(TEAMS_FILE, d)
    return d


def delete_team(team_id):
    d = load_all()
    d["teams"] = [t for t in d["teams"] if t["id"] != team_id]
    if d.get("active") == team_id:
        d["active"] = d["teams"][0]["id"] if d["teams"] else None
    _write(TEAMS_FILE, d)
    return d


def duplicate_team(team_id):
    d = load_all()
    for t in d["teams"]:
        if t["id"] == team_id:
            nid = _new_id(d["teams"])
            d["teams"].append(dict(id=nid, name=(t.get("name") or "Team") + " copy",
                                   players=list(t["players"]), bank=t.get("bank", 0.0),
                                   free_transfers=t.get("free_transfers", 1),
                                   formation=t.get("formation") or "auto"))
            d["active"] = nid
            _write(TEAMS_FILE, d)
            break
    return d


def load():
    """Backwards-compatible: the active team in the old single-squad shape."""
    t = get_active()
    # A team synced by entry_id has no reason to carry typed players too, so
    # "nothing typed" must not mean "nothing to load" once entry_id is set.
    if not t or not (t.get("players") or t.get("entry_id")):
        return None
    return dict(players=t.get("players") or [], bank=t.get("bank", 0.0),
                free_transfers=t.get("free_transfers", 1),
                formation=t.get("formation") or "auto",
                entry_id=t.get("entry_id"),
                name=t.get("name"), id=t.get("id"))


def load_one(team_id):
    """A specific team in the single-squad shape, regardless of which is active.

    Lets the server work out every team's advice in the background instead of
    only whichever one happens to be open.
    """
    if not team_id:
        return load()
    for t in load_all()["teams"]:
        if t["id"] == team_id and (t.get("players") or t.get("entry_id")):
            return dict(players=t.get("players") or [], bank=t.get("bank", 0.0),
                        free_transfers=t.get("free_transfers", 1),
                        formation=t.get("formation") or "auto",
                        entry_id=t.get("entry_id"),
                        name=t.get("name"), id=t.get("id"))
    return None


def save(players, bank=0.0, free_transfers=1, name=None, team_id=None):
    """Save into the active team (or a named one)."""
    if team_id is None:
        cur = get_active()
        team_id = cur["id"] if cur else None
        if name is None and cur:
            name = cur.get("name")
    return upsert_team(team_id, name or "My team", players, bank, free_transfers)


def clear():
    """Remove every team."""
    for p in (TEAMS_FILE, SQUAD_FILE):
        if os.path.exists(p):
            os.remove(p)


# The My Squad editor lays out its 15 boxes in this fixed shape (2 GK, 5 DEF,
# 5 MID, 3 FWD) and only ever saves a full set of 15, in that order - see
# saveSquad() in the UI, which refuses to save short of 15. So the slot a name
# was typed into is a real claim about position, even though only name/club
# make it into storage. This is that claim, indexed the same way.
SLOT_POS = [1, 1] + [2] * 5 + [3] * 5 + [4] * 3
POS_LABEL = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# When a row's clear first choice is already taken by an earlier row, a stand-in
# has to be at least this good to be used at all. 88 is the weakest exact-name
# reading (a full-name hit); everything below that is a substring of somebody's
# name, which is not evidence of who was meant.
DUPLICATE_FALLBACK_MIN = 88


def resolve(players, elements, teams):
    """
    Match stored name+club entries to live FPL players.

    Returns (resolved, problems) where resolved is a list of
    {name, club, id, price, matched_name, matched_club} and problems lists any
    entry that could not be matched confidently.

    A match is also checked against the slot it was typed into. FPL reclassifies
    a player's position occasionally, and a fuzzy name match can land on the
    wrong namesake; either way, the player's *current* position is what scoring
    must use, but a mismatch means the squad silently no longer has the 2/5/5/3
    shape the editor assumed when it accepted 15 entries as complete - and every
    downstream squad-quota assumption (build_squad, plan_transfers, plan_route)
    is built on that shape holding. Reported rather than left to surface as an
    inexplicable wrong player count somewhere else entirely.
    """
    short = {t["id"]: t.get("short_name", "") for t in teams}
    full = {t["id"]: t.get("name", "") for t in teams}

    index = []
    for e in elements:
        tid = e["team"]
        index.append(dict(
            id=e["id"], web=norm(e.get("web_name", "")),
            second=norm(e.get("second_name", "")),
            first=norm(e.get("first_name", "")),
            whole=norm(f"{e.get('first_name','')} {e.get('second_name','')}"),
            club_s=norm(short.get(tid, "")), club_f=norm(full.get(tid, "")),
            price=e["now_cost"] / 10.0, minutes=int(e.get("minutes") or 0),
            display=e.get("web_name", ""), club_disp=short.get(tid, ""),
            pos=e.get("element_type"),
        ))

    resolved, problems = [], []
    used = set()
    slot_pos = SLOT_POS if len(players) == 15 else None
    for slot_i, p in enumerate(players):
        want_n = norm(p.get("name", ""))
        want_c = norm(p.get("club", ""))
        if not want_n:
            continue

        def name_score(c):
            if c["web"] == want_n:
                return 100
            if c["second"] == want_n:
                return 90
            if c["whole"] == want_n:
                return 88
            if want_n and (want_n in c["web"] or c["web"] in want_n):
                return 60
            if want_n and want_n in c["whole"]:
                return 50
            return -1

        # If exactly one player in the whole pool carries this name at surname
        # tier or better, the name alone already answers "who did you mean" -
        # Donnarumma, Dubravka, Thiaw are each the only one of that name in the
        # game. A club typed for a genuinely unique name is not disambiguating
        # anything, so getting it wrong (a summer transfer, a typo, last
        # season's shirt) must not cost anything either.
        unique_exact = sum(1 for c in index if name_score(c) >= 88) == 1

        # Scored WITHOUT any already-taken penalty, so "who did you mean" is
        # answered on the name and club alone. Whether that player is still
        # free is a separate question, handled below - rolling it into the
        # score let a taken first choice quietly lose to some unrelated player
        # who merely shared a fragment of the surname, and the squad then
        # contained someone the owner had never typed.
        def score(c):
            s = name_score(c)
            if s < 0:
                return -1
            if s >= 88 and unique_exact:
                # Nothing the club field says can add or subtract confidence
                # here - the name alone already identifies exactly one player,
                # so treat it as fully confirmed rather than merely tolerated.
                # Without this a stale or mistyped club (Dubravka moved clubs,
                # a typo, last season's shirt) still kept an unambiguous exact
                # name match from ever reading as more than "weak - check this",
                # on every single row it happened to, for no real reason.
                return 140
            if want_c and (c["club_s"] == want_c or c["club_f"] == want_c):
                s += 40
            elif want_c:
                s -= 25            # name hit but wrong club: probably a namesake
            return s

        ranked = sorted(((score(c), c["minutes"], c) for c in index),
                        key=lambda t: (-t[0], -t[1]))
        ranked = [(s, c) for s, _m, c in ranked if s > 0]
        best, best_s = (ranked[0][1], ranked[0][0]) if ranked else (None, 0)
        taken = None
        if best is not None and best["id"] in used:
            # The player this entry most clearly refers to is already spoken
            # for by an earlier row. That means the same person was entered
            # twice, and silently handing this row to somebody else is how a
            # name nobody typed ends up in the squad - so say so, and take the
            # best still-free candidate only as a fallback.
            taken = best
            # Only stand in a genuine alternative reading of the name - an exact
            # web, surname or full-name hit. Below that the "match" is a
            # fragment: "Mitchell" scores against Dominic Solanke-Mitchell purely
            # because his full name contains it, and with the real Mitchell taken
            # that fragment used to win outright, putting a forward nobody had
            # ever typed into the squad. A row left unresolved is reported and
            # fixable; a wrong player silently taken as read is neither.
            nxt = next(((s, c) for s, c in ranked
                        if c["id"] not in used and s >= DUPLICATE_FALLBACK_MIN), None)
            best, best_s = (nxt[1], nxt[0]) if nxt else (None, 0)
        if best and best_s > 0:
            used.add(best["id"])
            if taken is not None:
                problems.append(dict(
                    name=p.get("name"), club=p.get("club"),
                    reason=(f"Looks like {taken['display']} ({taken['club_disp']}), who is already "
                            f"in an earlier row - entered twice? Using {best['display']} "
                            f"({best['club_disp']}) here instead, which may not be who you meant.")))
            resolved.append(dict(name=p.get("name"), club=p.get("club"),
                                 slot=slot_i,
                                 # "start" / "bench" / None - carried through so the
                                 # XI solver can honour it against a live player id
                                 role=p.get("role"),
                                 id=best["id"], price=best["price"],
                                 sell=float(p.get("sell") or best["price"]),
                                 matched_name=best["display"],
                                 matched_club=best["club_disp"],
                                 pos=best["pos"],
                                 exact=(best_s >= 130)))
            if taken is not None:
                pass                      # already reported as a duplicate above
            elif slot_pos is not None and best["pos"] != slot_pos[slot_i]:
                problems.append(dict(
                    name=p.get("name"), club=p.get("club"),
                    reason=(f"Entered as {POS_LABEL.get(slot_pos[slot_i], '?')} but FPL now lists "
                            f"{best['display']} as {POS_LABEL.get(best['pos'], '?')} - the squad no "
                            "longer has a legal 2/5/5/3 shape until this is fixed.")))
            elif best_s < 130:
                # A weak match (no exact name hit, or the club did not confirm
                # it) still gets used, because refusing it outright would break
                # a squad over a typo. But it is a guess, and a wrong guess
                # here is exactly how a player nobody typed - a namesake, a
                # transferred-out player who still shares a surname - ends up
                # silently in the squad. Reported so it can be checked, not
                # silently trusted.
                problems.append(dict(
                    name=p.get("name"), club=p.get("club"),
                    reason=(f"Matched to {best['display']} ({best['club_disp']}) on a weak name/club "
                            "match - double check this is who you meant.")))
        elif taken is not None:
            problems.append(dict(
                name=p.get("name"), club=p.get("club"),
                reason=(f"Looks like {taken['display']} ({taken['club_disp']}), who is already in an "
                        "earlier row. No other player clearly goes by this name, so this row is "
                        "unresolved - correct it to whoever you actually meant.")))
        else:
            problems.append(dict(name=p.get("name"), club=p.get("club"),
                                 reason="No player of that name at that club."))
    return resolved, problems


# The squad the user is actually running, so the dashboard is useful on first open.
# Overwrite freely from the My Squad tab.
DEFAULT = [
    {"name": "Raya",          "club": "ARS"},
    {"name": "Dubravka",      "club": "TOT"},
    {"name": "Shaw",          "club": "MUN"},
    {"name": "Gabriel",       "club": "ARS"},
    {"name": "Guehi",         "club": "MCI"},
    {"name": "Diop",          "club": "IPS"},
    {"name": "Pedro Porro",   "club": "TOT"},
    {"name": "Anderson",      "club": "MCI"},
    {"name": "B.Fernandes",   "club": "MUN"},
    {"name": "Rogers",        "club": "CHE"},
    {"name": "Rashford",      "club": "MUN"},
    {"name": "Burrowes",      "club": "AVL"},
    {"name": "Haaland",       "club": "MCI"},
    {"name": "Mheuka",        "club": "CHE"},
    {"name": "Kusi-Asare",    "club": "FUL"},
]
