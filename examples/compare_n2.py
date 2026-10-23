"""Step 4.4 of PLAN.md phase 4: the C++ TRN fix (lunar_landing_gnc._flight.trn)
against the Python reference on the seed 20261040 golden traces.

Every recorded N2 call is repeated in C++ with the Python call's own inputs:
the image and the truth camera from the run's inputs file, the DEM window,
the reference map (each checked against the hash recorded with the call),
and for the inner calls the Python's own intermediate values: a template
rebuilt from its recorded crop (checked against its recorded hash), a
surface from the correlation that made it, and the estimated camera that
fix_pass builds, recomputed with the Python's own projection functions from
the pass's recorded position and height. Guard (PLAN.md bar 1) per function
and quantity: the largest C++ minus Python difference over all calls at most
1,000 times the largest Python one-ulp spread (work/step4_2/summary.json); a
quantity whose spread is zero must agree bit for bit. Discrete outcomes
(integer peak, edge, register, iterations, status, dark, cells, the second
peak's place, the dither registers) must agree except at the Python's own
recorded ties.

Usage: python examples/compare_n2.py [compare | bench-input]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import golden_traces as GT  # noqa: E402
import tie_amendment as TA  # noqa: E402

OUT = ROOT / "work" / "step4_4"
GUARD = 1000.0


def _cam(v):
    v = np.asarray(v, float)
    return dict(position=v[0:3], attitude=v[3:12].reshape(3, 3), focal_px=float(v[12]), width=int(v[13]),
                height=int(v[14]))


def _template(row):
    """A template rebuilt from its recorded crop."""
    t = np.full((800, 800), np.nan)
    r0, r1, c0, c1 = (int(v) for v in row)
    return t, (r0, r1, c0, c1)


class Run:
    """One recorded run's N2 inputs."""

    def __init__(self, path):
        import nav_baselines as NB
        from lunar_landing_gnc import _flight as F
        self.F = F
        self.t, self.meta = GT.load(path)
        inp = np.load(GT.OUT / self.meta["inputs"])
        self.images, self.cams = {}, {}
        for k in inp.files:
            if k.startswith("img_t"):
                self.images[GT._arr_id(inp[k])] = (inp[k], _cam(inp["cam_t" + k[5:]]))
        st = NB._static()
        self.heights = st["dem"].h
        self.ref = np.load(ROOT / self.meta["map"]).astype(np.float64)
        g = st["grid"]
        self.grid = dict(x0=float(g.x0), y1=float(g.y1), cell_m=float(g.cell_m), ny=int(g.ny), nx=int(g.nx))
        self.map = F.trn.MapContext(self.grid, self.heights, self.ref)
        self.h_sha, self.r_sha = GT._arr_id(self.heights), GT._arr_id(self.ref)
        o = self.t.get("trn.orthorectify", {})
        self.templates = {}
        for j in range(len(o.get("parent", []))):
            tmpl = np.full((800, 800), np.nan)
            r0, r1, c0, c1 = (int(v) for v in o["out.bbox"][j])
            tmpl[r0:r1, c0:c1] = np.asarray(o["out.crop"][j], float)
            assert GT._arr_id(tmpl) == o["out.template_sha"][j], "template not rebuilt exactly"
            self.templates[o["out.template_sha"][j]] = tmpl
        c = self.t.get("trn.correlate", {})
        self.surfaces = {c["out.surface_sha"][j]: np.asarray(c["out.surface"][j], float)
                         for j in range(len(c.get("parent", [])))}

    def image(self, sha):
        return self.images[sha]

    def check(self, row):
        assert row.get("in.heights_sha", self.h_sha) == self.h_sha and row.get("in.reference_sha", self.r_sha) == self.r_sha


def rows(table):
    n = len(table.get("parent", []))
    return [{k: v[j] for k, v in table.items()} for j in range(n)]


def est_camera(cam, xy, z):
    """fix_pass's estimated camera, by the Python's own functions."""
    from lunar_landing_gnc.projection import MOON_RADIUS_M, map_to_latlon, unit_vector
    lat, lon = map_to_latlon(xy[0], xy[1])
    return dict(cam, position=(MOON_RADIUS_M + z) * unit_vector(lat, lon))


def nd(a, b):
    """Largest difference of two values (NaN where both NaN is no difference;
    a changed NaN pattern or shape is inf)."""
    absent = lambda v: v is None or np.all(np.isnan(np.asarray(v, float)))
    if absent(a) and absent(b):      # no value on either side (None here, NaN in the stored table)
        return 0.0
    a = np.asarray(np.nan if a is None else a, float)
    b = np.asarray(np.nan if b is None else b, float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return np.inf
    d = np.where(np.isnan(a), 0.0, np.abs(a - b))
    return float(d.max()) if d.size else 0.0


PASS_Q = ("fix", "ncc", "second", "offset")
DITHER_Q = PASS_Q + ("dithers.0", "dithers.1", "dithers.2")


def pass_fields(d):
    out = dict(fix=d.get("fix"), ncc=d.get("ncc"), second=d.get("second"), offset=d.get("offset"))
    for k in range(3):
        if "dithers" in d:
            out[f"dithers.{k}"] = d["dithers"][k]
    disc = dict(status=d.get("status"), cells=d.get("cells"), edge=d.get("edge"), register=d.get("register"),
                iterations=d.get("iterations"), radius=d.get("radius"))
    if "dither_registers" in d:
        disc.update({f"dither_registers.{k}": d["dither_registers"][k] for k in range(3)})
    return out, disc


def rec_pass(r, dither=False):
    out = {q: r.get("out." + q) for q in (DITHER_Q if dither else PASS_Q)}
    if out.get("second") is not None and np.ndim(out["second"]) == 0 and not np.isfinite(out["second"]):
        pass
    disc = dict(status=r.get("out.status"), cells=r.get("out.cells"), edge=r.get("out.edge"),
                register=r.get("out.register"), iterations=r.get("out.iterations"), radius=r.get("out.radius"))
    if dither:
        disc.update({f"dither_registers.{k}": r.get(f"out.dither_registers.{k}") for k in range(3)})
    return out, disc


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None or (_isnan(a) and b is None) or (_isnan(b) and a is None)
    try:
        return float(a) == float(b) if not isinstance(a, str) else a == b
    except (TypeError, ValueError):
        return a == b


def _isnan(v):
    try:
        return v is not None and not isinstance(v, str) and np.isnan(float(v))
    except (TypeError, ValueError):
        return False


def compare_run(path):
    R = Run(path)
    F = R.F.trn
    worst, disc, secs = {}, [], {}
    ties = set()

    # the amendment of 22 Oct 2026: a call in a tie's family is held back,
    # every other call goes straight to item 2's worst (worst2)
    fams = TA.n2_families(R.t)
    fam_calls = {c for f in fams for c in f["calls"]}
    stash, worst2 = {}, {}

    def w(name, q, v, j):
        worst.setdefault(name, {})
        worst[name][q] = max(worst[name].get(q, 0.0), v)
        if (("navtrn.candidate", j) if name == "margin" else (name, j)) in fam_calls:
            stash.setdefault((name, j), {})
            stash[(name, j)][q] = max(stash[(name, j)].get(q, 0.0), v)
        else:
            worst2.setdefault(name, {})
            worst2[name][q] = max(worst2[name].get(q, 0.0), v)

    def d(name, j, what, py, cpp):
        disc.append(dict(file=path.name, function=name, call=j, what=what, python=py, cpp=cpp))

    def timed(name, f):
        t0 = time.perf_counter()
        out = f()
        secs.setdefault(name, []).append(time.perf_counter() - t0)
        return out
    T = R.t
    for name in ("trn.correlate",):
        for j, r in enumerate(rows(T.get(name, {}))):
            if r.get("disagree#dx_int") or r.get("disagree#dy_int"):
                ties.add((name, j))
    # orthorectify: the Python's estimated camera, from the parent pass
    fp = rows(T.get("navtrn.fix_pass", {}))
    for j, r in enumerate(rows(T.get("trn.orthorectify", {}))):
        par = json.loads(r["parent"]) if isinstance(r["parent"], str) else r["parent"]
        p = fp[par[1]]
        img, cam = R.image(r["in.image_sha"])
        cam_e = est_camera(cam, np.asarray(p["in.xy"], float), float(p["in.z"]))
        out = timed("orthorectify", lambda: F.orthorectify(img, cam_e, R.map))
        rec = R.templates[r["out.template_sha"]]
        if not np.array_equal(np.isnan(out), np.isnan(rec)):
            d("trn.orthorectify", j, "cells in view", int(np.isfinite(rec).sum()), int(np.isfinite(out).sum()))
        w("trn.orthorectify", "crop", nd(np.where(np.isnan(rec), np.nan, out), rec), j)
    for j, r in enumerate(rows(T.get("trn.correlate", {}))):
        tmpl = R.templates[r["in.template_sha"]]
        out = timed(f"correlate r={int(r['in.radius'])}", lambda: F.correlate(tmpl, R.ref, int(r["in.radius"])))
        for q in ("dx", "dy", "ncc", "surface"):
            w("trn.correlate", q, nd(out[q], r["out." + q]), j)
        for q in ("dx_int", "dy_int", "edge"):
            if not _same(out[q], r["out." + q]):
                d("trn.correlate", j, q, r["out." + q], out[q])
    for j, r in enumerate(rows(T.get("trn.register_peak", {}))):
        tmpl = R.templates[r["in.template_sha"]]
        coarse = dict(dx_int=int(r["in.coarse.dx_int"]), dy_int=int(r["in.coarse.dy_int"]), dx=float(r["in.coarse.dx"]),
                      dy=float(r["in.coarse.dy"]), ncc=float(r["in.coarse.ncc"]), edge=bool(r["in.coarse.edge"]))
        out = timed("register_peak", lambda: F.register_peak(tmpl, R.ref, coarse, int(r["in.radius"])))
        for q in ("dx", "dy", "ncc", "gain", "bias"):
            w("trn.register_peak", q, nd(out.get(q), r.get("out." + q)), j)
        for q in ("register", "iterations", "edge"):
            if not _same(out.get(q), r.get("out." + q)):
                d("trn.register_peak", j, q, r.get("out." + q), out.get(q))
    for j, r in enumerate(rows(T.get("navtrn.second_peak", {}))):
        s = R.surfaces[r["in.surface_sha"]]
        out = F.second_peak(s, int(r["in.dx"]), int(r["in.dy"]))
        whole = r.get("out.")
        if whole is not None and np.ndim(whole) == 1 and not np.all(np.isnan(np.asarray(whole, float))):
            rec = (float(whole[0]), int(whole[1]), int(whole[2]))      # stored as one array when found
        else:
            rec = (r.get("out.0"), r.get("out.1"), r.get("out.2"))    # (nan, None, None): none found
        w("navtrn.second_peak", "value", nd(out[0], rec[0]), j)
        for k in (1, 2):
            if not _same(out[k], rec[k]):
                d("navtrn.second_peak", j, f"place {k}", rec[k], out[k])
    for j, r in enumerate(rows(T.get("navtrn.dark", {}))):
        if bool(F.dark(R.templates[r["in.template_sha"]])) != bool(r["out."]):
            d("navtrn.dark", j, "dark", r["out."], not r["out."])
    corr = rows(T.get("trn.correlate", {}))

    def tie_under(j):
        """Whether fix_pass call j's correlation is a recorded Python tie."""
        for k, c in enumerate(corr):
            par = json.loads(c["parent"]) if isinstance(c["parent"], str) else c["parent"]
            if par == ["navtrn.fix_pass", j]:
                return ("trn.correlate", k) in ties
        return False
    for j, r in enumerate(fp):
        R.check(r)
        img, cam = R.image(r["in.image_sha"])
        xy = tuple(float(v) for v in r["in.xy"])
        out = timed(f"fix_pass r={int(r['in.radius'])} {r['in.subcell']}",
                    lambda: F.fix_pass(img, cam, xy, float(r["in.z"]), R.map, int(r["in.radius"]), r["in.subcell"]))
        a, da = pass_fields(out)
        b, db = rec_pass(r)
        for q in PASS_Q:
            w("navtrn.fix_pass", q, nd(a[q], b[q]), j)
        if nd(a["offset"], b["offset"]) >= 0.5:
            d("navtrn.fix_pass", j, "integer peak" + (" (recorded tie)" if tie_under(j) else ""), b["offset"], a["offset"])
        for q in da:
            if not _same(da[q], db[q]):
                d("navtrn.fix_pass", j, q, db[q], da[q])
    for name, fn in (("navtrn._dither3", "dither3"), ("navtrn.second_pass", "second_pass")):
        for j, r in enumerate(rows(T.get(name, {}))):
            R.check(r)
            img, cam = R.image(r["in.image_sha"])
            xy = tuple(float(v) for v in r["in.xy"])
            if fn == "dither3":
                out = timed("dither3", lambda: F.dither3(img, cam, xy, float(r["in.z"]), R.map))
            else:
                out = timed("second pass (R5D)", lambda: F.second_pass(r["in.method"], img, cam, xy, float(r["in.z"]), R.map))
            a, da = pass_fields(out)
            b, db = rec_pass(r, dither=True)
            for q in DITHER_Q:
                w(name, q, nd(a.get(q), b.get(q)), j)
            for q in da:
                if not _same(da[q], db.get(q)):
                    d(name, j, q, db.get(q), da[q])
    # the margin each first pass gives the filter
    fx = T["N1.fix"]
    for j, r in enumerate(rows(T.get("navtrn.candidate", {}))):
        img, cam = R.image(r["in.image_sha"])
        x = np.asarray(r["in.x"], float)
        out = F.fix_pass(img, cam, (float(x[0]), float(x[1])), float(x[2]), R.map, int(r["in.radius"]))
        a, da = pass_fields(out)
        b, db = rec_pass(r)
        for q in PASS_Q:
            w("navtrn.candidate", q, nd(a[q], b[q]), j)
        if out.get("fix") is not None:
            m_c = F.peak_margin(out["ncc"], np.nan if out["second"] is None else out["second"])
            k = [i for i in range(len(fx["t"])) if fx["status"][i] not in ("no fix",)][j] if False else j
            w("margin", "margin", nd(m_c, fx["margin"][k]), j)
    # the C++'s side of each tie; what follows from it left out where it
    # took the other side; the first amendment (the four quantities, every
    # call at an integer-peak tie) left out whatever the C++ did
    keys = [(x["function"], x["call"], x["what"]) for x in disc if "call" in x]
    excluded, amendment = set(), []
    for f in fams:
        ev = TA.n2_flipped(f, keys, stash)
        fol = TA.n2_follows(f)
        if ev:
            excluded |= {(fn, jj, q) for (fn, jj), qs in fol.items() for q in qs}
        amendment.append(dict(file=path.name, tie=list(f["tie"]), kind=f["kind"], cpp_other_side=bool(ev),
                              evidence=ev, left_out={f"{fn} {jj}": {q: stash[(fn, jj)][q] for q in qs
                                                                     if q in stash.get((fn, jj), {})}
                                                     for (fn, jj), qs in fol.items()} if ev else {}))
    tie_fp = {f["calls"][1][1] for f in fams if f["kind"] == "integer peak" and len(f["calls"]) > 1}
    tie_cand = {f["calls"][2][1] for f in fams if f["kind"] == "integer peak" and len(f["calls"]) > 2
                and f["calls"][2][0] == "navtrn.candidate"}
    first = {("navtrn.fix_pass", jj, q) for jj in tie_fp for q in ("fix", "offset")} | \
            {("navtrn.candidate", jj, q) for jj in tie_cand for q in ("fix", "offset")}
    TA.fold(worst2, stash, excluded | first)
    return dict(file=path.name, worst=worst, discrete=disc, ties=sorted(ties), seconds=secs, worst2=worst2,
                amendment=amendment, first_amendment=[dict(file=path.name, call=[fn, jj], quantity=q)
                                                      for fn, jj, q in sorted(first)])


def cmd_compare():
    OUT.mkdir(parents=True, exist_ok=True)
    summ = json.loads((GT.OUT.parent / "summary.json").read_text())
    gate = json.loads((ROOT / "work" / "step4_3" / "gate_spread.json").read_text())
    t0 = time.time()
    runs = [compare_run(p) for p in sorted(GT.OUT.glob("nav_e*_k*_g*.npz"))]
    took = time.time() - t0
    worst = {}
    for r in runs:
        for name, qs in r["worst"].items():
            for q, v in qs.items():
                worst.setdefault(name, {})
                worst[name][q] = max(worst[name].get(q, 0.0), v)
    table = []
    for name, qs in worst.items():
        for q, v in qs.items():
            if name == "margin":
                sp = max(f["margin"] for rr in gate for f in rr["fixes"])
            elif name == "navtrn.candidate":
                sp = summ["spread"]["navtrn.fix_pass"].get(q, {}).get("max")
            elif name == "navtrn.second_peak":
                sp = max(v["max"] for v in summ["spread"][name].values())
            else:
                sp = summ["spread"].get(name, {}).get(q, {}).get("max")
            if sp is None:
                ok, rule = v == 0.0, "no spread recorded: bit for bit"
            elif sp == 0.0:
                ok, rule = v == 0.0, "spread zero: bit for bit"
            else:
                ok, rule = v <= GUARD * sp, f"<= {GUARD * sp:.3g}"
            table.append(dict(function=name, quantity=q, cpp_minus_python=v, python_spread=sp, guard=rule, ok=bool(ok)))
    discrete = [x for r in runs for x in r["discrete"]]
    ties = [(r["file"], t) for r in runs for t in r["ties"]]
    secs = {}
    for r in runs:
        for k, v in r["seconds"].items():
            secs.setdefault(k, []).extend(v)
    times = {k: dict(calls=len(v), median_ms=float(np.median(v) * 1e3), max_ms=float(np.max(v) * 1e3))
             for k, v in secs.items()}
    worst2 = {}
    for r in runs:
        for name, qs in r["worst2"].items():
            for q, v in qs.items():
                worst2.setdefault(name, {})
                worst2[name][q] = max(worst2[name].get(q, 0.0), v)
    item2 = [dict(function=name, quantity=q, cpp_minus_python=v) for name, qs in worst2.items() for q, v in qs.items()]
    res = dict(table=table, discrete=discrete, recorded_ties=ties, times=times, seconds=took, item2=item2,
               amendment=[a for r in runs for a in r["amendment"]],
               first_amendment=[a for r in runs for a in r["first_amendment"]])
    (OUT / "compare.json").write_text(json.dumps(res, indent=1, default=str))
    for row in table:
        print(f"{row['function']:20s} {row['quantity']:10s} C++-Python {row['cpp_minus_python']:.3g}  spread "
              f"{row['python_spread']}  {row['guard']}  {'ok' if row['ok'] else 'OVER'}")
    print("discrete disagreements:", len(discrete), "of which at recorded ties:",
          sum("recorded tie" in x["what"] for x in discrete))
    for x in discrete[:40]:
        print("  ", x)
    print("recorded Python ties:", ties)
    for k, v in sorted(times.items()):
        print(f"time {k}: {v['calls']} calls, median {v['median_ms']:.2f} ms, max {v['max_ms']:.2f} ms")
    return res


def bench_input(paths, out_file):
    """Every recorded fix of one flight's runs (paths, one per gap), for
    llg_flight_bench trn: the heights, the maps, the flight's images and
    cameras, and per fix the map, image, estimate (x, y, z), search radius and
    whether the filter asked for a second pass, as a flat binary of doubles.
    Returns the number of fixes."""
    runs = [Run(p) for p in paths]
    imgs = list(runs[0].images.items())
    shas = [k for k, _ in imgs]
    recs = []
    for m, R in enumerate(runs):
        cands = rows(R.t.get("navtrn.candidate", {}))
        seconds = {json.dumps(list(np.asarray(r["in.xy"], float))): r for r in rows(R.t.get("navtrn.second_pass", {}))}
        for r in cands:
            x = np.asarray(r["in.x"], float)
            fixed = r.get("out.fix") is not None and np.all(np.isfinite(np.asarray(r["out.fix"], float)))
            asked = fixed and json.dumps(list(np.asarray(r["out.fix"], float))) in seconds
            recs += [m, shas.index(r["in.image_sha"]), x[0], x[1], x[2], int(r["in.radius"]), 1.0 if asked else 0.0]
    g = runs[0].grid
    head = [len(runs), len(imgs), len(recs) // 7, g["x0"], g["y1"], g["cell_m"], g["ny"], g["nx"], 512, 512]
    parts = [np.asarray(head, float), runs[0].heights.ravel()] + [R.ref.ravel() for R in runs]
    for _, (img, cam) in imgs:
        parts += [img.ravel(), cam["position"], cam["attitude"].ravel(), [cam["focal_px"], cam["width"], cam["height"]]]
    parts.append(np.asarray(recs, float))
    np.concatenate([np.asarray(p, float) for p in parts]).astype("<f8").tofile(out_file)
    return len(recs) // 7


def cmd_bench_input():
    """Flight e1 k1 at each gap (bench_input)."""
    OUT.mkdir(parents=True, exist_ok=True)
    n = bench_input([GT.OUT / f"nav_e1_k1_g{g}.npz" for g in ("+0", "+24", "+168")], OUT / "bench_trn.bin")
    print("wrote", OUT / "bench_trn.bin", n, "fixes")


if __name__ == "__main__":
    {"compare": cmd_compare, "bench-input": cmd_bench_input}[sys.argv[1] if len(sys.argv) > 1 else "compare"]()
