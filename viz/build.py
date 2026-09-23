"""Emit the page: frontier metric + multi-seed replay."""
import json, pathlib, statistics as st

SEEDS = list(range(10))
CFG = [("jev_raw","Jev","cloud"),
       ("board_student","Student 87KB","local"),
       ("board_budget_208","Student + search","local")]

runs = {k: {s: json.loads((pathlib.Path("results/seeds")/f"{k}__s{s}.json").read_text())
            for s in SEEDS} for k,_,_ in CFG}

def ttf(r, n):
    for f in r["frames"]:
        if f["score"] >= n: return f["t"]
    return None

agg, curves = {}, {}
for k,_,_ in CFG:
    rs = [runs[k][s] for s in SEEDS]
    agg[k] = dict(
        score=round(st.mean(r["score"] for r in rs), 1),
        sem=round(st.pstdev([r["score"] for r in rs])/len(rs)**.5, 2),
        sec=round(st.mean(r["total_ms"] for r in rs)/1000, 2),
        ms=round(st.mean(r["ms_per_move"] for r in rs), 2),
        usd=round(st.mean(r["usd"] for r in rs), 5))
    c = []
    for n in range(1, 40):
        hit = [t for t in (ttf(r, n) for r in rs) if t is not None]
        if len(hit) < 5: break
        c.append([round(st.median(hit), 1), n])
    curves[k] = c

target = round(agg["jev_raw"]["score"])
tt = {}
for k,_,_ in CFG:
    hit = [t for t in (ttf(runs[k][s], target) for s in SEEDS) if t is not None]
    tt[k] = {"ms": round(st.median(hit), 1) if hit else None, "n": len(hit)}

def pack(frames):
    """Cell indices instead of 110-char board strings -- ~4x smaller."""
    out = []
    for f in frames:
        flat = f["board"].replace("\n", "")
        out.append({"t": f["t"], "s": f["score"], "l": f["len"],
                    "m": f["move"], "c": f["calls"],
                    "h": flat.find("@"), "f": flat.find("*"),
                    "b": [i for i, ch in enumerate(flat) if ch == "o"]})
    return out

replay = {k: {str(s): {**{kk: runs[k][s][kk] for kk in
                          ("score","moves","alive","calls","usd",
                           "total_ms","ms_per_move")},
                       "frames": pack(runs[k][s]["frames"])}
              for s in SEEDS} for k,_,_ in CFG}

sims = {}
rec = pathlib.Path("results/recordings")
for f in rec.glob("self_*.json"):
    _, who, n = f.stem.split("_")
    r = json.loads(f.read_text())
    sims.setdefault(n, {})[who] = {kk: r[kk] for kk in
        ("score","moves","calls","usd","total_ms","ms_per_move")}

data = {"agg": agg, "curves": curves, "target": target, "ttf": tt,
        "replay": replay, "sims": sims, "seeds": SEEDS,
        "cfg": [{"key":k,"name":n,"where":w} for k,n,w in CFG]}

tpl = pathlib.Path("viz/template.html").read_text()
out = tpl.replace("/*__DATA__*/", json.dumps(data, separators=(",",":")))
pathlib.Path("viz/index.html").write_text(out)
print(f"target score {target}")
for k,n,_ in CFG:
    a = agg[k]
    print(f"  {n:<9} score {a['score']:>5} +/-{a['sem']:<5} "
          f"{a['sec']:>6}s/game  {a['ms']:>7}ms/move  ${a['usd']}  "
          f"to-{target}: {tt[k]['ms']} ms ({tt[k]['n']}/10)")
print(f"wrote viz/index.html ({len(out)/1024:.0f} KB)")
