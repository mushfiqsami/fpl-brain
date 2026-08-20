#!/usr/bin/env python3
"""
Bundle FPL Brain into a single self-contained .py file.

Why: uploading a folder tree from an iPhone is painful. Two flat files is not.
The output embeds every module, the HTML interface and the app icons, so the
whole app deploys by uploading fplbrain_app.py + Dockerfile and nothing else.

Run:  python build_standalone.py
"""
import base64, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
# Discovered from disk rather than hardcoded: a hand-maintained list silently
# drops any module added later, and the failure only shows up in the deployed
# build. Order matters only in that dependencies must exist before use, and the
# import machinery below execs them into one package namespace, so alphabetical
# is fine.
MODULES = sorted(
    f[:-3] for f in os.listdir(os.path.join(HERE, "fplbrain"))
    if f.endswith(".py") and f != "__init__.py"
)
# Where the bundle lands depends on which layout you are in. The repo keeps
# app.py at the root next to this script; the older working tree kept it in a
# phone/ subfolder. Writing blindly to phone/ meant a clone of the repo built a
# second copy nobody deploys and left the real app.py untouched.
OUT = (os.path.join(HERE, "app.py") if os.path.exists(os.path.join(HERE, "app.py"))
       else os.path.join(HERE, "phone", "app.py"))


def read(*p):
    with open(os.path.join(HERE, *p), encoding="utf-8") as f:
        return f.read()


def b64file(*p):
    with open(os.path.join(HERE, *p), "rb") as f:
        return base64.b64encode(f.read()).decode()


# ---- gather sources -------------------------------------------------------
srcs = {m: read("fplbrain", f"{m}.py") for m in MODULES}
ui = read("ui.html")
dash = read("dashboard.py")
icons = {n: b64file("static", n) for n in sorted(os.listdir(os.path.join(HERE, "static")))
         if n.endswith(".png")}

# ---- patch dashboard for the bundled world --------------------------------
# 1. the UI comes from an embedded string, not a file next to the script
dash = dash.replace(
    '''PAGE = ""   # filled in from ui.html at import time
_ui = os.path.join(HERE, "ui.html")
if os.path.exists(_ui):
    with open(_ui, encoding="utf-8") as f:
        PAGE = f.read()''',
    '''PAGE = base64.b64decode(_UI_B64).decode("utf-8")''')

# 2. icons come from the embedded dict, not a static/ folder
dash = dash.replace(
    '''def _ICON_SOURCE():
    """Icon names available. Overridden in the single-file build."""
    d = os.path.join(HERE, "static")
    return [f for f in os.listdir(d) if f.endswith(".png")] if os.path.isdir(d) else []


def _ICON_BYTES(name):
    with open(os.path.join(HERE, "static", name), "rb") as f:
        return f.read()''',
    '''def _ICON_SOURCE():
    return list(_ICONS)


def _ICON_BYTES(name):
    return base64.b64decode(_ICONS[name])''')

# 2a. carry the player baselines inside the bundle, compressed. Without this the
# hosted app re-downloads two seasons from GitHub on every redeploy, because the
# disk is wiped each time.
_prior_path = os.path.join(HERE, "data", "player_prior.json")
if os.path.exists(_prior_path):
    import zlib
    with open(_prior_path, "rb") as f:
        _blob = base64.b64encode(zlib.compress(f.read(), 9)).decode()
    srcs["seed"] = srcs["seed"].replace('EMBEDDED_PRIOR = ""',
                                        f'EMBEDDED_PRIOR = "{_blob}"')
    print(f"  embedded player prior: {len(_blob)/1024:.0f} KB compressed")
else:
    print("  WARNING: data/player_prior.json missing - hosted app will re-download it")

# 2b. carry config.json inside the bundle. It is not one of the published files,
# so without this the hosted app runs with no measured team priors and rates
# every club as league average.
_cfg = read("config.json")
dash = dash.replace("EMBEDDED_CONFIG = {}",
                    "EMBEDDED_CONFIG = json.loads(base64.b64decode(\n    \""
                    + base64.b64encode(_cfg.encode()).decode()
                    + "\").decode(\"utf-8\"))")

# 3. stamp the build so the running app can say which version it is
_stamp = __import__("hashlib").md5(
    (ui + dash + "".join(srcs[m] for m in MODULES)).encode()).hexdigest()[:7]
_when = __import__("datetime").datetime.now().strftime("%d %b %H:%M")
dash = dash.replace('BUILD = "dev"', f'BUILD = "{_stamp} · {_when}"')

# 3b. a missing ui.html can no longer happen
dash = dash.replace('''    if not PAGE:
        print("ERROR: ui.html is missing from this folder.")
        input("Press Enter to close...")
        return
''', "")

dash = dash.replace(
    "import datetime, hashlib, json, os, random, sys, threading, time, traceback, webbrowser, socket",
    "import base64, datetime, hashlib, json, os, random, sys, threading, time, traceback, webbrowser, socket")
dash = re.sub(r'^if __name__ == "__main__":\n    main\(\)\n?', "", dash, flags=re.M)
# a __future__ import is only legal as the very first statement of a file, and in
# the bundle dashboard's code lands two thirds of the way down. Hoist it.
dash = dash.replace("from __future__ import annotations\n", "")

header = '''#!/usr/bin/env python3
from __future__ import annotations
# =============================================================================
#  FPL BRAIN - single-file build
#
#  Everything is in here: the model, the optimiser, the web interface and the
#  icons. Deploy by uploading this file plus the Dockerfile. Nothing else.
#
#  Generated by build_standalone.py - edit the source modules, not this file.
# =============================================================================
import base64, os, sys, types

BASE = os.path.dirname(os.path.abspath(__file__))

_SRC = {}
'''

parts = [header]
for m in MODULES:
    parts.append(f'_SRC["{m}"] = base64.b64decode(\n    "{base64.b64encode(srcs[m].encode()).decode()}"\n).decode("utf-8")\n')

parts.append(f"_ORDER = {MODULES!r}\n")
parts.append('''
# Rebuild the package in memory. Each module gets a __file__ under BASE so that
# the "HERE = dirname(dirname(__file__))" lines in them still resolve to the
# folder this script is sitting in.
_pkg = types.ModuleType("fplbrain")
_pkg.__path__ = [os.path.join(BASE, "fplbrain")]
_pkg.__package__ = "fplbrain"
sys.modules["fplbrain"] = _pkg
for _n in _ORDER:
    _m = types.ModuleType("fplbrain." + _n)
    _m.__package__ = "fplbrain"
    _m.__file__ = os.path.join(BASE, "fplbrain", _n + ".py")
    sys.modules["fplbrain." + _n] = _m
    exec(compile(_SRC[_n], _m.__file__, "exec"), _m.__dict__)
    setattr(_pkg, _n, _m)

''')

parts.append(f'_UI_B64 = "{base64.b64encode(ui.encode()).decode()}"\n\n')
parts.append("_ICONS = {\n")
for n, b in icons.items():
    parts.append(f'    "{n}": "{b}",\n')
parts.append("}\n\n")
parts.append("# " + "=" * 75 + "\n# dashboard\n# " + "=" * 75 + "\n")
parts.append(dash)
parts.append('\n\nif __name__ == "__main__":\n    main()\n')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write("".join(parts))

# --- files that make it a Hugging Face Gradio Space -----------------------
# fastapi/uvicorn are preinstalled on Hugging Face but not on Render, Koyeb etc.
with open(os.path.join(HERE, "phone", "requirements.txt"), "w", encoding="utf-8") as f:
    f.write("pulp\nopenpyxl\nfastapi\nuvicorn\n")

# README.md front-matter is how HF configures a Space
with open(os.path.join(HERE, "phone", "README.md"), "w", encoding="utf-8") as f:
    f.write("""---
title: FPL Brain
emoji: "\u26bd"
colorFrom: green
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
---

Fantasy Premier League squad and transfer advice. Open the Space URL directly.
""")

# kept for anyone on a paid Docker Space
with open(os.path.join(HERE, "phone", "Dockerfile"), "w", encoding="utf-8") as f:
    f.write("""FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir pulp openpyxl fastapi uvicorn
COPY app.py .
ENV PORT=7860 FPL_CLOUD=1
EXPOSE 7860
CMD ["python", "app.py"]
""")

size = os.path.getsize(OUT)
print(f"wrote {OUT}  ({size/1024:.0f} KB)")
print(f"  {len(MODULES)} modules, {len(icons)} icons, UI {len(ui)/1024:.0f} KB")
