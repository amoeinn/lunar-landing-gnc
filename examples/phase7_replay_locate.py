"""Phase 7 step 7.3: where in each flight a violation falls. Read only: for
every FAIL in OUT/summary.json it walks the flight's items with
replay_check's own functions, exactly as the check does, asserts that the
smallest margin over tolerance it finds equals the one the checker wrote
(bit for bit), and reports where the items beyond tolerance lie. Nothing is
judged again: a mismatch is an error, not a new verdict.

    python examples/phase7_replay_locate.py work/step7_2 work/step7_3
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "examples")]

import numpy as np  # noqa: E402

import replay_check as C  # noqa: E402


def terrain_of(h, m, row):
    """The checker's ground for the flight, built as check_flight builds it."""
    P, _ = C.samples(h, m)
    ts = row.get("true_site")
    xs, ys = list(P[:, 0]) + ([ts[0]] if ts else []), list(P[:, 1]) + ([ts[1]] if ts else [])
    return C.Terrain(xs, ys)


def glide_items(h, m):
    """check_glide_slope's judged items: per guided tick, its absolute time,
    step, the estimate's height above and range from the site in force, the
    margin, tolerance and margin over tolerance."""
    hs = C.holds(h, m)
    t_step = {int(st): float(t) for st, t in zip(h["step_index"], h["step_t"])}
    est = {int(round(float(t) / C.TICK)): (np.asarray(q, float), np.asarray(w, float))
           for t, q, w in zip(h["est_t"], h["est_pos"], h["est_vel"])}
    out = []
    for k, tt, p in zip(h["tick_hold"], h["tick_t"], h["tick_pos"]):
        x = hs[int(k)]
        if x["phase"] != "guided" or x["rhs"] != "_truth_rhs" or not np.all(np.isfinite(p)):
            continue
        st = x["step"]
        ta = t_step[st] + float(tt)
        q, w = est[int(round(ta / C.TICK))]
        o = x["origin"]
        up, dx, dy = q[2] - o[2], q[0] - o[0], q[1] - o[1]
        mg, tol = C.elevation_margin(up, dx, dy), C.glide_tolerance(up, dx, dy, w)
        out.append(dict(t=ta, step=st, up=float(up), range=float(np.hypot(dx, dy)), margin=mg, tolerance=tol,
                        over=mg + tol))
    return out


def clearance_items(h, m, row):
    """check_clearance's items: per sample, its phase, index, clearance,
    tolerance and margin over tolerance (judged samples only)."""
    terrain = terrain_of(h, m, row)
    P, ph, V, DT = C.samples(h, m, full=True)
    c = P[:, 2] - terrain.height(P[:, 0], P[:, 1])
    ok = np.isfinite(c)
    above = ~ok & (P[:, 2] > C.HIGHEST_POINT_M)
    c = np.where(above, P[:, 2] - C.HIGHEST_POINT_M, c)
    ok = ok | above
    tol = np.where(np.isfinite(V).all(1), C.clearance_tolerance(terrain, P, np.nan_to_num(V), DT), 0.0)
    over = np.where(ok, c + tol, np.inf)
    n_desc = sum(1 for x in ph if x == "descent")
    out, k_desc = [], 0
    for j in range(len(P)):
        if ph[j] == "descent":
            k_desc += 1
        out.append(dict(index=j, phase=ph[j], x=float(P[j, 0]), y=float(P[j, 1]), z=float(P[j, 2]),
                        clearance=float(c[j]), tolerance=float(tol[j]), over=float(over[j]),
                        descent_ticks_before_contact=(n_desc - k_desc + 1) if ph[j] == "descent" else None))
    return out


def locate(src, v):
    seed, job, ep = v["flight"].split()
    ep = int(ep[1:])
    _, flights = C.load_job(Path(src) / f"seed_{seed}", job)
    row, h, m = next((r, hh, mm) for r, hh, mm in flights if mm["epoch"] == ep)
    name = v["check"]
    if name.startswith("3"):
        it = glide_items(h, m)
        assert min(x["over"] for x in it) == v["over"], "check 3: smallest over does not match the checker"
        bad = [x for x in it if x["over"] < 0]
        w = min(it, key=lambda x: x["over"])
        return dict(ticks_beyond=len(bad), of_ticks=len(it), first_t=bad[0]["t"], last_t=bad[-1]["t"],
                    steps=sorted({x["step"] for x in bad}), worst=dict(t=w["t"], step=w["step"], up=w["up"],
                                                                        range=w["range"]),
                    guided_end_t=it[-1]["t"])
    if name.startswith("4"):
        it = clearance_items(h, m, row)
        assert min(x["over"] for x in it) == v["over"], "check 4: smallest over does not match the checker"
        bad = [x for x in it if x["over"] < 0]
        return dict(samples_beyond=len(bad), items=[{k: x[k] for k in ("phase", "index", "x", "y", "z", "clearance",
                                                                       "tolerance", "over",
                                                                       "descent_ticks_before_contact")}
                                                    for x in bad])
    if name.startswith("6"):
        return dict(touchdown_map=m["touchdown_map"], epoch=m["epoch"], t_touchdown=m["t_touchdown"],
                    **v["where"])
    return dict(v["where"])


def main(src, out):
    out = Path(out)
    sm = json.loads((out / "summary.json").read_text())
    res = []
    for v in sm["violations"]:
        res.append(dict(v, located=locate(src, v)))
    (out / "violations_located.json").write_text(json.dumps(res, indent=1, default=float))
    L = ["", "## Where each violation falls (examples/phase7_replay_locate.py)", ""]
    for v in res:
        lc = v["located"]
        if v["check"].startswith("3"):
            w = lc["worst"]
            s = (f"{lc['ticks_beyond']} of {lc['of_ticks']} guided ticks beyond tolerance, t = {lc['first_t']:.2f} to "
                 f"{lc['last_t']:.2f} s (guidance steps {lc['steps']}; guided flight ends at {lc['guided_end_t']:.2f} s); "
                 f"worst at t = {w['t']:.2f} s, the estimate {w['up']:.2f} m above and {w['range']:.2f} m from the "
                 f"site in force")
        elif v["check"].startswith("4"):
            s = "; ".join(f"{x['phase']} sample {x['index']}"
                          + (f" ({x['descent_ticks_before_contact']} tick(s) before contact)"
                             if x["descent_ticks_before_contact"] else "")
                          + f", clearance {x['clearance']:.4g} m against tolerance {x['tolerance']:.4g} m"
                          for x in lc["items"])
        elif v["check"].startswith("6"):
            s = (f"touchdown at map ({lc['touchdown_map'][0]:.1f}, {lc['touchdown_map'][1]:.1f}) m, epoch "
                 f"{lc['epoch']}, t = {lc['t_touchdown']:.2f} s; sun elevation {lc.get('sun_elevation', float('nan')):.4f} "
                 f"deg against horizon {lc.get('horizon', float('nan')):.4f} deg")
        else:
            s = json.dumps(lc, default=float)
        L.append(f"- {v['flight']} ({v['outcome']}), {v['check']}: margin {v['margin']:.4g} against tolerance "
                 f"{v['tolerance']:.4g} {v['unit']}; {s}")
    with open(out / "summary.md", "a") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"{len(res)} violations located; every smallest margin over tolerance matched the checker's")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
