"""Phase 6 step 6.1f: the C++ ray cast against the Python one (D56).

  rays OUT      per ray: every frame of two flights (seed 2, draw 0, TRN,
                epoch 0, gaps 0 and +168 h, flown on the Python cast) cast
                again in C++ on the same inputs, and frames aimed across
                tile edges and across the region's edge; per frame the
                largest range difference over rays both casts range, the
                rays one casts and the other does not (before and after the
                150 to 1,300 m limits), and each cast's time with its tiles
                already made. Writes work/step6_1/OUT.json.
  flights A B   whole flights: the saved records of folders A (Python cast)
                and B (C++ cast) of work/step6_1, field by field: the
                outcome class, the site, the verdict and the fix outcomes
                must agree; every continuous quantity's largest difference
                is reported. Writes work/step6_1/flights_A_B.json.

Seed 2 only (a development seed). Nothing is scored. Needs LLG_DATA.
Usage: python examples/phase6_cast_agreement.py rays OUT | flights A B
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

import phase6_flights as P  # noqa: E402
from lunar_landing_gnc import estflight as EF  # noqa: E402
from lunar_landing_gnc import estlanding as EL  # noqa: E402
from lunar_landing_gnc import lidar as L  # noqa: E402

SEED, DRAW = 2, 0
GAPS = (0.0, 168.0)
TOL = 1e-9
RUN_KEYS = ("seconds", "shared_seconds", "maxrss_mb", "tiles_held")
DISCRETE = ("outcome", "kind", "site", "safe", "gear_safe", "criterion2", "triggers", "frames", "lost_lock",
            "crosses_tile_edge")


def frame_stats(label, r, c, tp, tc):
    both = np.isfinite(r) & np.isfinite(c)
    lim = lambda v: np.isfinite(v) & (v >= L.RANGE_MIN) & (v <= L.RANGE_MAX)  # noqa: E731
    return dict(label=label, rays=int(r.size), ranged=int(both.sum()),
                max_diff=float(np.max(np.abs(r[both] - c[both]))) if both.any() else 0.0,
                identical=int(np.sum((r == c) | (np.isnan(r) & np.isnan(c)))),
                hit_miss_differ=int(np.sum(np.isfinite(r) != np.isfinite(c))),
                returned_differ=int(np.sum(lim(r) != lim(c))), python_s=tp, cpp_s=tc)


def both_casts(label, ground, o, d, near, rec, py_cast):
    """C++ first (it makes any tile a ray reaches beyond near), then Python
    on the same tiles; then each timed again with every tile made."""
    c = EL.ray_cast_ground_cpp(ground, o, d, near)
    r = py_cast(ground, o, d, near)
    a = time.perf_counter()
    py_cast(ground, o, d, near)
    tp = time.perf_counter() - a
    a = time.perf_counter()
    EL.ray_cast_ground_cpp(ground, o, d, near)
    tc = time.perf_counter() - a
    rec.append(frame_stats(label, r, c, tp, tc))
    return r


def edge_frames(ctx, ground):
    """Frames aimed across tile edges and across the region's edge, from 500
    m back and 400 m up: (label, origin, dirs, near)."""
    cx, cy, _ = ctx["target"]
    lo, hi = ground.bounds()
    s = ground.size / 2
    aims = [("tile edge x", cx + s, cy + 10.0), ("tile edge y", cx + 10.0, cy + s),
            ("tile corner ne", cx + s, cy + s), ("tile corner sw", cx - s, cy - s),
            ("region edge east", hi[0], cy + 20.0), ("region edge north", cx + 30.0, hi[1]),
            ("region edge west", lo[0], cy - 20.0), ("region edge south", cx - 30.0, lo[1]),
            ("region corner ne", hi[0], hi[1]), ("region corner sw", lo[0], lo[1])]
    looks = {"west": (-500.0, 0.0), "south": (0.0, -500.0), "east": (500.0, 0.0), "north": (0.0, 500.0)}
    out = []
    for name, ax, ay in aims:
        px, py = min(max(ax, lo[0] + 1.0), hi[0] - 1.0), min(max(ay, lo[1] + 1.0), hi[1] - 1.0)
        az = float(ground.height(np.array([px]), np.array([py]), strict=False)[0])
        for look, (bx, by) in looks.items():
            o = np.array([ax + bx, ay + by, az + 400.0])
            aim = np.array([ax, ay, az])
            d = L.frame_dirs(o, aim)
            hw = float(np.linalg.norm(aim - o)) * L.FOV + 30.0
            out.append((f"{name}, from the {look}", o, d, (ax - hw, ax + hw, ay - hw, ay + hw)))
    return out


def cmd_rays(name):
    ctx = P.landing_context()
    py_cast = EL.ray_cast_ground
    rec = []
    label = ["shared start"]

    def wrapped(ground, origin, dirs, near, step=L.STEP, tol=L.TOL):
        return both_casts(label[0], ground, origin, dirs, near, rec, py_cast)
    EF.EL.ray_cast_ground = wrapped
    flights = {}
    try:
        snap = EF.shared_start(ctx, SEED, DRAW, until="touchdown")
        for g in GAPS:
            label[0] = f"flight seed {SEED} draw {DRAW} epoch 0 gap {g:+g} h"
            out = EF.from_shared(ctx, snap, 0, g)
            flights[label[0]] = dict(outcome=out.get("outcome"), frames=sum(r["label"] == label[0] for r in rec))
    finally:
        EF.EL.ray_cast_ground = py_cast
    ground = ctx["ground"](SEED, DRAW)
    for lab, o, d, near in edge_frames(ctx, ground):
        both_casts(lab, ground, o, d, near, rec, py_cast)
    groups = {}
    for r in rec:
        k = r["label"].split(", from")[0]
        g = groups.setdefault(k, dict(frames=0, rays=0, ranged=0, max_diff=0.0, identical=0, hit_miss_differ=0,
                                      returned_differ=0, python_s=0.0, cpp_s=0.0))
        g["frames"] += 1
        for f in ("rays", "ranged", "identical", "hit_miss_differ", "returned_differ", "python_s", "cpp_s"):
            g[f] += r[f]
        g["max_diff"] = max(g["max_diff"], r["max_diff"])
    tot = dict(frames=len(rec), rays=sum(r["rays"] for r in rec), ranged=sum(r["ranged"] for r in rec),
               max_diff=max(r["max_diff"] for r in rec), identical=sum(r["identical"] for r in rec),
               hit_miss_differ=sum(r["hit_miss_differ"] for r in rec),
               returned_differ=sum(r["returned_differ"] for r in rec),
               python_s=sum(r["python_s"] for r in rec), cpp_s=sum(r["cpp_s"] for r in rec))
    tot["speedup"] = tot["python_s"] / tot["cpp_s"]
    tot["agree"] = bool(tot["max_diff"] <= TOL and tot["hit_miss_differ"] == 0)
    res = dict(tol=TOL, total=tot, groups=groups, flights=flights, frames=rec)
    (P.OUT / f"{name}.json").write_text(json.dumps(res, indent=1, default=float))
    for k, g in groups.items():
        print(f"{k:45s} frames {g['frames']:4d} rays {g['rays']:8d} ranged {g['ranged']:8d} "
              f"max diff {g['max_diff']:.3e} identical {g['identical']:8d} hit/miss differ {g['hit_miss_differ']} "
              f"returned differ {g['returned_differ']} python {g['python_s']:.2f} s C++ {g['cpp_s']:.2f} s")
    print(json.dumps(tot, indent=1))


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in RUN_KEYS:
                continue
            walk(a.get(k), b.get(k), f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (u, v) in enumerate(zip(a, b)):
            walk(u, v, f"{path}[]" if not isinstance(u, (dict, list)) else f"{path}[{i}]", out)
    elif isinstance(a, float) and isinstance(b, float):
        key = path
        d = abs(a - b) if np.isfinite(a) and np.isfinite(b) else (0.0 if (a == b or (a != a and b != b)) else np.inf)
        out["continuous"][key] = max(out["continuous"].get(key, 0.0), d)
    elif a != b:
        out["discrete"].append((path, a, b))


def cmd_flights(a, b):
    fa, fb = P.OUT / a, P.OUT / b
    res = dict(jobs={}, discrete=[], largest={})
    for p in sorted(fa.glob("d*.json")):
        q = fb / p.name
        if not q.exists():
            res["jobs"][p.name] = "missing"
            continue
        ra, rb = json.loads(p.read_text()), json.loads(q.read_text())
        out = dict(continuous={}, discrete=[])
        walk(ra.get("rows"), rb.get("rows"), "rows", out)
        keyed = [d for d in out["discrete"]]
        res["jobs"][p.name] = dict(discrete=len(keyed), continuous_nonzero=sum(v > 0 for v in out["continuous"].values()))
        res["discrete"] += [(p.name, *d) for d in keyed]
        for k, v in out["continuous"].items():
            kk = k.split(".", 1)[1] if "." in k else k
            res["largest"][kk] = max(res["largest"].get(kk, 0.0), v)
    rows = [res["largest"][k] for k in res["largest"]]
    res["registered"] = dict(
        class_site_verdict_fixes_agree=not any(any(f".{f}" in d[1] for f in DISCRETE + ("status",)) for d in res["discrete"]),
        any_discrete_difference=bool(res["discrete"]), largest_continuous=max(rows) if rows else 0.0)
    (P.OUT / f"flights_{a}_{b}.json".replace("/", "_")).write_text(json.dumps(res, indent=1, default=str))
    for n, v in res["jobs"].items():
        print(n, v)
    for d in res["discrete"]:
        print("DIFFERS", d)
    for k, v in sorted(res["largest"].items(), key=lambda kv: -kv[1])[:25]:
        print(f"{k:50s} {v:.3e}")
    print(json.dumps(res["registered"]))


if __name__ == "__main__":
    if sys.argv[1] == "rays":
        cmd_rays(sys.argv[2])
    else:
        cmd_flights(sys.argv[2], sys.argv[3])
