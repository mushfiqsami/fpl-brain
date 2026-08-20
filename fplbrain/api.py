"""
Data access layer.

LiveClient  -> the official FPL API (what you use week to week)
ArchiveClient -> vaastav/Fantasy-Premier-League on GitHub (used only by `backtest`)

Everything is cached to data/cache/ so a re-run inside the TTL costs no requests.
The FPL API is public and needs no key or login.
"""
from __future__ import annotations
import json, os, time, urllib.request, urllib.error, csv, io

BASE = "https://fantasy.premierleague.com/api"
UA = {"User-Agent": "Mozilla/5.0 (fplbrain; personal FPL planning tool)"}
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, "data", "cache")


def _cache_path(key: str) -> str:
    os.makedirs(CACHE, exist_ok=True)
    safe = key.replace("/", "_").replace("?", "_").strip("_")
    return os.path.join(CACHE, f"{safe}.json")


def _get_json(url: str, cache_key: str, ttl: int, force: bool = False):
    p = _cache_path(cache_key)
    if not force and os.path.exists(p) and (time.time() - os.path.getmtime(p)) < ttl:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise FileNotFoundError(f"404 from FPL API: {url}") from e
        raise RuntimeError(f"FPL API returned HTTP {e.code} for {url}") from e
    except Exception as e:
        # fall back to a stale cache rather than dying mid-gameweek
        if os.path.exists(p):
            print(f"  ! network failed ({e}); using cached copy of {cache_key}")
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        raise
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


class LiveClient:
    """Reads the live FPL API."""

    def __init__(self, ttl_minutes: int = 60):
        self.ttl = ttl_minutes * 60

    def bootstrap(self, force=False):
        """Everything static-ish: players, teams, gameweeks, scoring rules."""
        return _get_json(f"{BASE}/bootstrap-static/", "bootstrap", self.ttl, force)

    def fixtures(self, force=False):
        return _get_json(f"{BASE}/fixtures/", "fixtures", self.ttl, force)

    def player_history(self, element_id: int, force=False):
        """Per-gameweek history for one player. Only called for players we shortlist."""
        return _get_json(f"{BASE}/element-summary/{element_id}/",
                         f"element_{element_id}", self.ttl, force)

    def entry(self, entry_id: int, force=False):
        """Your manager record: bank, squad value, chips, transfer count."""
        return _get_json(f"{BASE}/entry/{entry_id}/", f"entry_{entry_id}", self.ttl, force)

    def entry_picks(self, entry_id: int, gw: int, force=False):
        """Your 15 for a given gameweek, including selling_price / purchase_price."""
        return _get_json(f"{BASE}/entry/{entry_id}/event/{gw}/picks/",
                         f"picks_{entry_id}_{gw}", self.ttl, force)

    def entry_history(self, entry_id: int, force=False):
        return _get_json(f"{BASE}/entry/{entry_id}/history/", f"hist_{entry_id}", self.ttl, force)

    def live(self, gw: int, force=False):
        """Live/finished scores for a gameweek. Short TTL — this moves."""
        return _get_json(f"{BASE}/event/{gw}/live/", f"live_{gw}", 300, force)


ARCHIVE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"


class ArchiveClient:
    """Historical season CSVs. Used by `backtest` to validate the model offline."""

    def __init__(self, season="2025-26"):
        self.season = season
        self.dir = os.path.join(HERE, "data", "archive", season)
        os.makedirs(self.dir, exist_ok=True)

    def _csv(self, path, name):
        local = os.path.join(self.dir, name)
        if not os.path.exists(local):
            url = f"{ARCHIVE}/{self.season}/{path}"
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                open(local, "wb").write(r.read())
        with open(local, encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    def players_raw(self):
        return self._csv("players_raw.csv", "players_raw.csv")

    def teams(self):
        return self._csv("teams.csv", "teams.csv")

    def fixtures(self):
        return self._csv("fixtures.csv", "fixtures.csv")

    def gw(self, n: int):
        return self._csv(f"gws/gw{n}.csv", f"gw{n}.csv")
