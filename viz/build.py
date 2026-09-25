"""Build docs/race.html: the three-board race replay, with the recordings inlined.
(The site's front page, docs/index.html, is a static page and is not built here.)

    uv run python viz/build.py

Crashes are allowed throughout (SNAKE_ALLOW_CRASH=1 when recording).
"""
import json, pathlib, statistics as st

R = pathlib.Path("results")
REPLAY = R / "marathon_crash.json"                 # record_replay.py
# compare.py on boards 100-119, which no training or tuning touched
FAIR = [p for p in (R / "heldout_crash.json", R / "heldout_crash_echo.json", R / "heldout_crash_tree.json") if p.exists()]
EXAMPLE = pathlib.Path("viz/example_move.json")    # one position, for "How they play"


def pack_replay(d):
    # The pause comes once both slow players have finished a game; Echo keeps
    # going a little past it.
    beat = max(d["jev"][0]["t0"] + d["jev"][0]["ms"], d["search"][0]["t0"] + d["search"][0]["ms"])
    horizon = beat * 1.25
    d["jev"], d["search"] = d["jev"][:1], d["search"][:1]
    d["student"] = [g for g in d["student"] if g["t0"] < horizon] or d["student"][:1]
    # An Echo game lasts a few milliseconds -- under a rendered frame -- so a
    # handful of snapshots is all anyone can see.
    for g in d["student"]:
        fr = g["frames"]
        if len(fr) > 3:
            step = (len(fr) - 1) / 2
            g["frames"] = [fr[round(i * step)] for i in range(3)]
    for key in ("jev", "student", "search"):
        for g in d[key]:
            for f in g["frames"]:
                if isinstance(f["b"], list):
                    f["b"] = "".join(f"{i:02d}" for i in f["b"])
    d["horizon_ms"] = min(horizon, max(g["t0"] + g["ms"] for k in ("jev", "student", "search") for g in d[k]))
    return d, beat


def pack_fair(rows):
    by = {}
    for r in rows:
        by.setdefault(r["config"], {})[r["seed"]] = r
    search = next(c for c in by if c.startswith("STUDENT(board)+tree@"))
    names = {"JEV(raw)": "jev", "STUDENT(board)": "echo", search: "search"}
    fair = {"boards": len(by["JEV(raw)"])}
    for cfg, key in names.items():
        sc = [r["score"] for r in by[cfg].values()]
        fair[key] = {"mean": round(st.mean(sc), 1), "se": round(st.stdev(sc) / len(sc) ** 0.5, 2),
                     "ms": round(st.median(r["ms"] for r in by[cfg].values()), 3)}
    for key, cfg in (("echo", "STUDENT(board)"), ("search", search)):
        diff = [by[cfg][s]["score"] - by["JEV(raw)"][s]["score"] for s in sorted(by[cfg])]
        fair["pair_" + key] = {"d": round(st.mean(diff), 2),
                               "se": round(st.stdev(diff) / len(diff) ** 0.5, 2),
                               "w": sum(x > 0 for x in diff), "l": sum(x < 0 for x in diff),
                               "t": sum(x == 0 for x in diff)}
    # Jev in search mode: the tree's Echo calls per move, asked of Jev instead.
    jev, tree = by["JEV(raw)"].values(), by[search].values()
    calls = sum(r["calls"] for r in tree) / sum(r["steps"] for r in tree)
    moves = st.mean(r["steps"] for r in tree)
    usd = sum(r["usd"] for r in jev) / sum(r["calls"] for r in jev)
    fair["jev_search"] = {"calls": round(calls), "hours": round(moves * calls * fair["jev"]["ms"] / 3.6e6, 1),
                          "usd": round(moves * calls * usd, 2)}
    return fair


D, beat = pack_replay(json.loads(REPLAY.read_text()))
D["fair"] = F = pack_fair([r for p in FAIR for r in json.loads(p.read_text())])
D["example"] = json.loads(EXAMPLE.read_text())
speed = float(f"{F['jev']['ms'] / F['echo']['ms']:.2g}")
times = round(F["search"]["mean"] / F["jev"]["mean"])
print(f"replay: jev {D['jev'][0]['score']}, search {D['search'][0]['score']}, "
      f"echo {len(D['student'])} games to {D['horizon_ms']/1000:.0f}s (pause {beat/1000:.1f}s)")
J = F["jev_search"]
print(f"{F['boards']} held-out boards: jev {F['jev']['mean']} echo {F['echo']['mean']} "
      f"search {F['search']['mean']}  |  {speed:,.0f}x faster, {times}x the score  |  "
      f"Jev in search mode: {J['calls']} calls/move, {J['hours']} h, ${J['usd']:.2f} a game")

tpl = pathlib.Path("viz/marathon_tpl.html").read_text()
out = (tpl.replace("/*__DATA__*/", json.dumps(D, separators=(",", ":")))
          .replace("/*__CORE__*/", pathlib.Path("viz/snake_core.js").read_text())
          .replace("__SPEED__", f"{speed:,.0f}×").replace("__TIMES__", f"{times}×")
          .replace("__CALLS__", str(J["calls"])).replace("__HOURS__", f"{J['hours']:g}")
          .replace("__USD__", f"${J['usd']:.2f}"))
pathlib.Path("docs").mkdir(exist_ok=True)
pathlib.Path("docs/race.html").write_text(out)
print(f"page {len(out)/1024:.0f} KB")
