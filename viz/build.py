"""Build docs/index.html: the page, with both modes' recordings inlined.

    uv run python viz/build.py
"""
import json, pathlib, statistics as st

R = pathlib.Path("results")
MODES = {
    "net":   {"replay": R / "marathon.json",       "fair": R / "fair_uncapped.json"},
    "crash": {"replay": R / "marathon_crash.json", "fair": R / "fair_crash_uncapped.json"},
}


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
    search = next(c for c in by if c.startswith("STUDENT(board)+search"))
    names = {"JEV(raw)": "jev", "STUDENT(board)": "echo", search: "search"}
    fair = {}
    for cfg, key in names.items():
        sc = [r["score"] for r in by[cfg].values()]
        fair[key] = {"mean": round(st.mean(sc), 1), "se": round(st.stdev(sc) / len(sc) ** 0.5, 2),
                     "ms": round(st.median(r["ms"] for r in by[cfg].values()), 2)}
    for key, cfg in (("echo", "STUDENT(board)"), ("search", search)):
        diff = [by[cfg][s]["score"] - by["JEV(raw)"][s]["score"] for s in sorted(by[cfg])]
        fair["pair_" + key] = {"d": round(st.mean(diff), 2),
                               "se": round(st.stdev(diff) / len(diff) ** 0.5, 2),
                               "w": sum(x > 0 for x in diff), "l": sum(x < 0 for x in diff),
                               "t": sum(x == 0 for x in diff)}
    return fair


ALL = {}
for mode, src in MODES.items():
    d, beat = pack_replay(json.loads(src["replay"].read_text()))
    d["fair"] = pack_fair(json.loads(src["fair"].read_text()))
    ALL[mode] = d
    F = d["fair"]
    print(f"{mode:<6} replay: jev {d['jev'][0]['score']}, search {d['search'][0]['score']}, "
          f"echo {len(d['student'])} games to {d['horizon_ms']/1000:.0f}s (pause {beat/1000:.1f}s)  |  "
          f"ten games: jev {F['jev']['mean']} echo {F['echo']['mean']} search {F['search']['mean']}")

tpl = pathlib.Path("viz/marathon_tpl.html").read_text()
out = tpl.replace("/*__DATA__*/", json.dumps(ALL, separators=(",", ":")))
out = out.replace("/*__CORE__*/", pathlib.Path("viz/snake_core.js").read_text())
pathlib.Path("docs").mkdir(exist_ok=True)
pathlib.Path("docs/index.html").write_text(out)
print(f"page {len(out)/1024:.0f} KB")
