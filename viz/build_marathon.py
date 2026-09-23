"""Pack the marathon recording into the page."""
import json, os, pathlib
CRASH = os.environ.get("SNAKE_ALLOW_CRASH") == "1"
SUFFIX = "_crash" if CRASH else ""

d = json.loads(pathlib.Path(f"results/marathon{SUFFIX}.json").read_text())

# The demo pauses when BOTH slow players have finished a game, so the horizon
# has to clear that moment with room for Echo to keep running afterwards.
# Jev's game is not always the longer of the two.
BEAT = max(d["budget_ms"], d["search"][0]["t0"] + d["search"][0]["ms"])
HORIZON = BEAT * 1.25
# Jev and Echo + search play exactly one game each and then hold their final
# board; only Echo keeps playing.
d["jev"] = d["jev"][:1]
d["search"] = d["search"][:1]
d["student"] = [g for g in d["student"] if g["t0"] < HORIZON] or d["student"][:1]

# A student run lasts ~62 ms, which is under four rendered frames at 60 fps.
# Storing more than that is storing pixels nobody can see.
KEEP = 3
for g in d["student"]:
    fr = g["frames"]
    if len(fr) > KEEP:
        step = (len(fr) - 1) / (KEEP - 1)
        g["frames"] = [fr[round(i * step)] for i in range(KEEP)]

# Body cells as a packed 2-char-per-index string rather than a JSON array.
for key in ("jev", "student", "search"):
    for g in d[key]:
        for f in g["frames"]:
            if isinstance(f["b"], list):
                f["b"] = "".join(f"{i:02d}" for i in f["b"])

d["horizon_ms"] = min(HORIZON,
    max(g["t0"] + g["ms"] for k in ("jev", "student", "search") for g in d[k]))

# Ten-seed comparison: the fair test of score, since one replayed game is
# one sample. Paired by seed, so board luck cancels out.
import statistics as st
if CRASH:
    # Jev from the first crash run; Echo and Echo + search from the retrained model.
    rows = [r for r in json.loads(pathlib.Path("results/fair_crash.json").read_text())
            if r["config"] == "JEV(raw)"]
    rows += json.loads(pathlib.Path("results/fair_crash_echo.json").read_text())
else:
    rows = json.loads(pathlib.Path("results/fair.json").read_text())
    # Echo + search was re-run with its budget matched to Jev's per-move time.
    matched = pathlib.Path("results/fair_search197.json")
    if matched.exists():
        rows = [r for r in rows if not r["config"].startswith("STUDENT(board)+search")]
        rows += json.loads(matched.read_text())
SEARCH = next(r["config"] for r in rows if r["config"].startswith("STUDENT(board)+search"))
by = {}
for r in rows:
    by.setdefault(r["config"], {})[r["seed"]] = r
names = {"JEV(raw)": "jev", "STUDENT(board)": "echo", SEARCH: "search"}
fair = {}
for cfg, key in names.items():
    sc = [r["score"] for r in by[cfg].values()]
    fair[key] = {"mean": round(st.mean(sc), 1),
                 "sem": round(st.pstdev(sc) / len(sc) ** 0.5, 2),
                 "ms": round(st.median(r["ms"] for r in by[cfg].values()), 2),
                 "n": len(sc)}
for key, cfg in (("echo", "STUDENT(board)"), ("search", SEARCH)):
    seeds = sorted(set(by[cfg]) & set(by["JEV(raw)"]))
    diff = [by[cfg][s]["score"] - by["JEV(raw)"][s]["score"] for s in seeds]
    fair["pair_" + key] = {"d": round(st.mean(diff), 2),
                           "se": round(st.stdev(diff) / len(diff) ** 0.5, 2),
                           "w": sum(x > 0 for x in diff), "l": sum(x < 0 for x in diff),
                           "t": sum(x == 0 for x in diff)}
d["fair"] = fair

tpl = pathlib.Path("viz/marathon_tpl.html").read_text()
out = tpl.replace("/*__DATA__*/", json.dumps(d, separators=(",", ":")))
out = out.replace("/*__CORE__*/", pathlib.Path("viz/snake_core.js").read_text())
pathlib.Path("viz/marathon.html").write_text(out)
site = pathlib.Path("site"); site.mkdir(exist_ok=True)
(site / "index.html").write_text(out)

at = [g for g in d["student"] if g["t0"] + g["ms"] <= BEAT]
print(f"jev game        {d['budget_ms']/1000:>6.1f}s   score {d['jev'][0]['score']}")
print(f"echo+search g1  {(d['search'][0]['t0']+d['search'][0]['ms'])/1000:>6.1f}s   "
      f"score {d['search'][0]['score']}")
print(f"beat (pause)    {BEAT/1000:>6.1f}s")
print(f"echo at beat    {len(at):>6} runs, mean "
      f"{sum(g['score'] for g in at)/len(at):.1f}, best {max(g['score'] for g in at)}")
print(f"student recorded {len(d['student']):>4} runs to "
      f"{(d['student'][-1]['t0']+d['student'][-1]['ms'])/1000:.0f}s")
print(f"horizon         {d['horizon_ms']/1000:>6.1f}s")
print(f"page            {len(out)/1024:>6.0f} KB")
print(f"ten-game: jev {fair['jev']['mean']} ({fair['jev']['ms']} ms)  "
      f"echo {fair['echo']['mean']}  search {fair['search']['mean']} "
      f"({fair['search']['ms']} ms)  [{SEARCH}]")
