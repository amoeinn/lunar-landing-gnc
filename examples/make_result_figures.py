"""Step 8.2: the result figures (PLAN.md "Phase 8: closeout").

    python examples/make_result_figures.py [--out DIR] [--only payoff,split,premise,phase5,replay]

Every plotted number is read from a committed result file; nothing is flown,
scored or typed here. One function per figure returns the data it plots
(payoff_data, split_data, premise_data, phase5_data, replay_data), and
tests/test_result_figures.py checks each against the file it reads.

Writes DIR (default docs/): fig_payoff.png, fig_split.png, fig_premise.png,
fig_phase5.png, fig_replay.png and fig_captions.md. A figure whose source file
is absent from the checkout is skipped and named.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"

# phase 6: the scored run and its two reference seeds (work/step6_3, the scorer's output)
P6_SEED, P6_REFS = 20261062, (20261060, 20261061)
P6_SCORE = WORK / "step6_3" / "score_{}.json"
# phase 3: the premise score, scored seed and reference seeds in one file (work/step3_3)
P3_SCORE = WORK / "step3_3" / "score.json"
# phase 5: the scored hazard run and the rock-field baselines
P5_SCORED = WORK / "step5_3" / "score_scored_20261052.json"
P5_BASE = WORK / "step5_2" / "baselines.json"
P5_ROCK_SEED = 20261050
ROCK_FIELDS = ("minimum", "nominal", "maximum")
# phase 7: the registered replay (7.3), its diagnosis (7.4), the ground-fix comparison (7.5c)
P7_SUMMARY = WORK / "step7_3" / "summary.json"
P7_DIAG = WORK / "step7_4" / "diagnosis.json"
P7_COMPARE = WORK / "step7_5" / "compare" / "summary.json"
P7_SEED = 20261062

SPLIT = ("safe landing", "abort to a safe orbit", "unsafe outcome")
SPLIT_COLOUR = ("#55a868", "#4c72b0", "#c44e52")
SEED_COLOUR = {P6_SEED: "#d62728", P6_REFS[0]: "#1f77b4", P6_REFS[1]: "#2ca02c"}


def load(path):
    return json.loads(Path(path).read_text())


def gap_hours(key):
    """'+24 h', '-168 h', '+0 h' (phase 6) or '+24', '-168' (phase 3) as hours."""
    return float(key.split()[0])


def gap_keys(d):
    return sorted((k for k in d if k != "no-TRN"), key=gap_hours)


# ---------------------------------------------------------------- data

def payoff_data():
    """Phase 6, per seed: the gaps, the pooled safe-landing fraction and the
    pooled median Q (over flights that touched down) at each gap, and the
    no-TRN lander's pooled safe-landing fraction and median Q; read from each
    seed's summary_6_2."""
    out = {}
    for s in (P6_SEED,) + P6_REFS:
        sm = load(str(P6_SCORE).format(s))["summary_6_2"]
        ks = gap_keys(sm["safe_fraction"])
        out[s] = dict(gaps=[gap_hours(k) for k in ks],
                      safe=[sm["safe_fraction"][k]["pooled"] for k in ks],
                      q_median=[sm["q"][k]["pooled"]["median"] for k in ks],
                      q_n=[sm["q"][k]["pooled"]["n"] for k in ks],
                      notrn_safe=sm["safe_fraction"]["no-TRN"]["pooled"],
                      notrn_q=sm["q"]["no-TRN"]["pooled"]["median"])
    return out


def split_data():
    """Phase 6, scored seed: the three-way split (D69) of the pooled TRN
    flights at each gap and of the no-TRN flights, as counts."""
    tw = load(str(P6_SCORE).format(P6_SEED))["three_way"]
    ks = gap_keys(tw) + ["no-TRN"]
    return dict(labels=ks, counts={c: [tw[k]["pooled"][c] for k in ks] for c in SPLIT},
                flights=[tw[k]["pooled"]["flights"] for k in ks])


def premise_data():
    """Phase 3: the fix hazard rate H against gap, both signs, for the scored
    seed (its registered H) and the two reference seeds (the all-fix-times
    rate in each seed's table, the same quantity), with the fixes behind each
    value; gaps with no fix are left out."""
    d = load(P3_SCORE)
    out = {}
    for s in [d["seed"]] + list(d["references"]):
        tab = d["tables"][str(s)]["all"]
        ks = [k for k in sorted(tab, key=gap_hours) if tab[k]["fixes"] > 0]
        out[s] = dict(gaps=[gap_hours(k) for k in ks], h=[tab[k]["worse"] for k in ks],
                      fixes=[tab[k]["fixes"] for k in ks])
    reg = d["criteria"][str(d["seed"])]["H"]
    out["registered_h"] = {float(k): v for k, v in reg.items() if v == v}
    out["scored"], out["references"] = d["seed"], list(d["references"])
    return out


def phase5_data():
    """Phase 5: safe landings of the system and of the blind lander by
    landing epoch on the scored seed, and pooled over epochs for the three
    rock fields on the reference seed's baselines."""
    r = load(P5_SCORED)["reported"]
    eps = sorted(k for k in r if k.startswith("epoch_"))
    b = load(P5_BASE)
    rk = [f"seed_{P5_ROCK_SEED}_{f}" for f in ROCK_FIELDS]
    return dict(epochs=[int(k.split("_")[1]) for k in eps],
                system=[r[k]["safe"]["system"] for k in eps], blind=[r[k]["safe"]["blind"] for k in eps],
                flights=[r[k]["flights"] for k in eps],
                rock=dict(fields=list(ROCK_FIELDS), system=[b[k]["pooled"]["safe"]["system"] for k in rk],
                          blind=[b[k]["pooled"]["safe"]["blind"] for k in rk],
                          flights=[b[k]["pooled"]["flights"] for k in rk]))


def replay_data():
    """Phase 7: violating flights per check for the registered run (three
    seeds, 7.3), for its seed 20261062 alone and for the ground-fix
    sensitivity run of that seed (7.5c); and violating flights by primary
    cause for checks 3 (first violation), 4 and 6, for the registered run
    (7.4), for its seed 20261062 alone (7.5c's restriction of 7.4, which
    holds check 4's causes by sample only, so that entry is None) and for
    the sensitivity run (7.5c)."""
    reg = load(P7_SUMMARY)["bar"]["violations_per_check"]
    cmp_ = load(P7_COMPARE)
    pc = cmp_["replay"]["per_check"]
    checks = list(reg)
    one = {c: pc[c]["registered"].get("FAIL", 0) for c in checks}
    sens = {c: pc[c]["sensitivity"].get("FAIL", 0) for c in checks}
    dg, ds = load(P7_DIAG), cmp_["causes"]["sensitivity"]

    def causes(dd):
        return {"3 glide slope": dd["glide"]["first"]["primary"], "4 terrain clearance":
                dd["clearance"]["flight_causes"], "6 site lit": dd["lit"]["primary"]}
    r1 = cmp_["causes"]["registered_7_4"]
    seed_causes = {"3 glide slope": r1["glide_first"], "4 terrain clearance": None, "6 site lit": r1["lit_primary"]}
    return dict(checks=checks, registered=[reg[c] for c in checks], registered_seed=[one[c] for c in checks],
                sensitivity=[sens[c] for c in checks], causes_registered=causes(dg), causes_seed=seed_causes,
                causes_sensitivity=causes(ds))


# ---------------------------------------------------------------- figures

def symlog_gap_axis(ax, gaps, linthresh=24):
    ax.set_xscale("symlog", linthresh=linthresh)
    ax.set_xticks(gaps)
    ax.set_xticklabels([f"{g:+.0f}" if g else "0" for g in gaps], fontsize=8)
    ax.set_xlabel("map gap, hours (map epoch minus landing epoch; symmetric log axis)", fontsize=9)


def fig_payoff(out):
    import matplotlib.pyplot as plt
    D = payoff_data()
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5))
    for s, d in D.items():
        tag = "scored" if s == P6_SEED else "reference"
        lw = 2.2 if s == P6_SEED else 1.2
        a.plot(d["gaps"], d["safe"], "o-", color=SEED_COLOUR[s], lw=lw, label=f"TRN, seed {s} ({tag})")
        a.axhline(d["notrn_safe"], color=SEED_COLOUR[s], ls="--", lw=1.0, label=f"no-TRN, seed {s}")
        b.plot(d["gaps"], d["q_median"], "o-", color=SEED_COLOUR[s], lw=lw, label=f"TRN, seed {s} ({tag})")
        if s == P6_SEED:
            for g, q, n in zip(d["gaps"], d["q_median"], d["q_n"]):
                b.annotate(f"n {n}", (g, q), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=7.5,
                           color=SEED_COLOUR[s], bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                                                            alpha=0.8))
        b.axhline(d["notrn_q"], color=SEED_COLOUR[s], ls="--", lw=1.0, label=f"no-TRN, seed {s}")
    gaps = D[P6_SEED]["gaps"]
    for ax in (a, b):
        symlog_gap_axis(ax, gaps)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    a.set_ylim(0, 1.03)
    a.set_ylabel("safe-landing fraction (400 flights per gap)", fontsize=9)
    a.set_title("Safe landings against map gap", fontsize=10)
    b.set_yscale("log")
    b.set_ylabel("median touchdown error Q, m (flights that touched down)", fontsize=9)
    b.set_title(f"Median Q against map gap (n: touchdowns of 400 behind each median, seed {P6_SEED})", fontsize=9)
    b.set_ylim(top=max(max(d["q_median"]) for d in D.values()) * 6)
    fig.suptitle("Phase 6 payoff: the TRN lander against map gap, scored seed and both reference seeds, with the "
                 "no-TRN lander's level", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig_payoff.png", dpi=110)
    plt.close(fig)


def fig_split(out):
    import matplotlib.pyplot as plt
    D = split_data()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(D["labels"]))
    bottom = np.zeros(len(x))
    for c, col in zip(SPLIT, SPLIT_COLOUR):
        v = np.array(D["counts"][c], float)
        ax.bar(x, v, bottom=bottom, color=col, label=c, width=0.7)
        for i, n in enumerate(v):
            if n >= 12:
                ax.text(x[i], bottom[i] + n / 2, f"{int(n)}", ha="center", va="center", fontsize=8, color="white")
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels([("TRN " + k) if k != "no-TRN" else "no-TRN" for k in D["labels"]], fontsize=8)
    ax.set_ylabel("flights", fontsize=9)
    ax.set_ylim(0, max(D["flights"]) * 1.12)
    ax.legend(fontsize=8, ncol=3, loc="upper center")
    ax.set_title(f"Phase 6, scored seed {P6_SEED}: outcome of every flight by map gap, three way split", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_split.png", dpi=110)
    plt.close(fig)


def fig_premise(out):
    import matplotlib.pyplot as plt
    D = premise_data()
    col = {D["scored"]: "#d62728", D["references"][0]: "#1f77b4", D["references"][1]: "#2ca02c"}
    fig, ax = plt.subplots(figsize=(10, 5))
    for s in [D["scored"]] + D["references"]:
        d = D[s]
        tag = "scored" if s == D["scored"] else "reference"
        ax.plot(d["gaps"], d["h"], "o-", color=col[s], lw=2.2 if tag == "scored" else 1.2,
                label=f"seed {s} ({tag})")
    symlog_gap_axis(ax, D[D["scored"]]["gaps"], linthresh=3)
    ax.set_ylim(-0.02, 1.0)
    ax.set_ylabel("fix hazard rate H: fraction of fixes worse than the prior alone", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Phase 3 premise: a stale map makes TRN fixes hazardous, both signs of gap", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_premise.png", dpi=110)
    plt.close(fig)


def fig_phase5(out):
    import matplotlib.pyplot as plt
    D = phase5_data()
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8))
    w = 0.38
    for ax, xs, sy, bl, n, lab in ((a, [f"epoch {e}" for e in D["epochs"]], D["system"], D["blind"], D["flights"],
                                    "landing epoch"),
                                   (b, D["rock"]["fields"], D["rock"]["system"], D["rock"]["blind"],
                                    D["rock"]["flights"], "rock field")):
        x = np.arange(len(xs))
        ax.bar(x - w / 2, sy, w, color="#55a868", label="system (hazard detection and divert)")
        ax.bar(x + w / 2, bl, w, color="#8c8c8c", label="blind lander (lands at the target)")
        for i in range(len(x)):
            ax.text(x[i] - w / 2, sy[i], f"{sy[i]}", ha="center", va="bottom", fontsize=8)
            ax.text(x[i] + w / 2, bl[i], f"{bl[i]}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(xs, fontsize=9)
        ax.set_ylim(0, max(n) * 1.15)
        ax.set_ylabel(f"safe landings (of {n[0]} flights per {lab})", fontsize=9)
        ax.legend(fontsize=8, loc="lower right")
    a.set_title("Scored seed 20261052: safe landings by landing epoch", fontsize=10)
    b.set_title(f"Rock-field sensitivity, reference seed {P5_ROCK_SEED}, all epochs", fontsize=10)
    fig.suptitle("Phase 5: hazard detection and divert against the blind lander", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig_phase5.png", dpi=110)
    plt.close(fig)


def fig_replay(out):
    import matplotlib.pyplot as plt
    D = replay_data()
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw=dict(width_ratios=[1, 1.25]))
    x = np.arange(len(D["checks"]))
    w = 0.27
    for i, (k, lab, col) in enumerate((("registered", "registered run, three seeds (7.3)", "#4c72b0"),
                                       ("registered_seed", f"registered run, seed {P7_SEED} alone", "#8172b2"),
                                       ("sensitivity", f"ground fix (D109), seed {P7_SEED} (7.5c)", "#dd8452"))):
        v = D[k]
        a.bar(x + (i - 1) * w, v, w, color=col, label=lab)
        for j, n in enumerate(v):
            if n:
                a.text(x[j] + (i - 1) * w, n, str(n), ha="center", va="bottom", fontsize=7)
    a.set_xticks(x)
    import textwrap
    a.set_xticklabels([textwrap.fill(c, 11) for c in D["checks"]], fontsize=8)
    a.set_ylabel("flights violating the check", fontsize=9)
    a.legend(fontsize=8)
    a.set_title("Violations by check", fontsize=10)
    rows = [(c, k) for c in D["causes_registered"] for k in sorted(set(D["causes_registered"][c])
                                                                   | set(D["causes_sensitivity"][c]))]
    y = np.arange(len(rows))
    h = 0.27
    for i, (key, lab, col) in enumerate((("causes_registered", "registered run, three seeds (7.4)", "#4c72b0"),
                                         ("causes_seed", f"registered run, seed {P7_SEED} alone", "#8172b2"),
                                         ("causes_sensitivity", f"ground fix, seed {P7_SEED} (7.5c)", "#dd8452"))):
        v = [None if D[key][c] is None else D[key][c].get(k, 0) for c, k in rows]
        b.barh(y + (i - 1) * h, [0 if n is None else n for n in v], h, color=col, label=lab)
        for j, n in enumerate(v):
            b.text(0 if n is None else n, y[j] + (i - 1) * h, " n/a (by sample only)" if n is None else f" {n}",
                   va="center", fontsize=7)
    b.set_yticks(y)
    b.set_yticklabels([f"check {c.split()[0]}: {k}" for c, k in rows], fontsize=8)
    b.invert_yaxis()
    b.set_xlabel("violating flights by primary cause", fontsize=9)
    b.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=3)
    b.set_title("Violations by cause (check 3: the first violation of each flight)", fontsize=10)
    fig.suptitle("Phase 7 replay checker: the registered run and the ground-fix sensitivity run", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig_replay.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- captions

def captions():
    """Captions with every number read through the data functions."""
    L = ["# Result figures", "",
         "Made by `examples/make_result_figures.py`; every number is read from the committed result file named "
         "under each figure, and `tests/test_result_figures.py` checks each figure's data against that file.", ""]
    if FIGS["payoff"][1]():
        D = payoff_data()
        s = D[P6_SEED]
        i168, i0 = s["gaps"].index(168.0), s["gaps"].index(0.0)
        L += ["## Payoff (`fig_payoff.png`)", "",
              f"The TRN lander's safe-landing fraction and median touchdown error Q against map gap, for the scored "
              f"seed {P6_SEED} and the reference seeds {P6_REFS[0]} and {P6_REFS[1]}, with each seed's no-TRN "
              f"lander as a dashed level. On the scored seed the safe fraction is {s['safe'][i0]:.3f} at gap 0 and "
              f"{s['safe'][i168]:.3f} at +168 h against {s['notrn_safe']:.3f} for the no-TRN lander; median Q is "
              f"{s['q_median'][i0]:.2f} m at gap 0 and {s['q_median'][i168]:.0f} m at +168 h against "
              f"{s['notrn_q']:.0f} m. Source: `work/step6_3/score_<seed>.json`, `summary_6_2`. The number beside each scored seed "
              "point is the touchdowns behind its median. Limit: median Q is "
              f"over the flights that touched down only, {s['q_n'][i168]} of 400 at +168 h on the scored seed, so "
              "at large gaps it describes a shrinking and selected subset; aborts and errors are not in it.", ""]
    if FIGS["split"][1]():
        D = split_data()
        k = D["labels"].index("+168 h")
        L += ["## Three way split (`fig_split.png`)", "",
              f"Every flight on the scored seed {P6_SEED}, by map gap and for the no-TRN lander, as safe landing, "
              f"abort to a safe orbit or unsafe outcome (D69). At +168 h: {D['counts']['safe landing'][k]} safe, "
              f"{D['counts']['abort to a safe orbit'][k]} aborts to a safe orbit and "
              f"{D['counts']['unsafe outcome'][k]} unsafe outcomes of {D['flights'][k]}. Source: "
              f"`work/step6_3/score_{P6_SEED}.json`, `three_way`. Limit: unsafe outcome holds unsafe landings, "
              "aborts that missed a safe orbit and flights that ended in an error together.", ""]
    if FIGS["premise"][1]():
        D = premise_data()
        s = D[D["scored"]]
        h = dict(zip(s["gaps"], s["h"]))
        L += ["## Premise (`fig_premise.png`)", "",
              f"The fix hazard rate H, the fraction of TRN fixes whose error exceeds the prior's alone, against map "
              f"gap of both signs, for the scored premise seed {D['scored']} and the reference seeds "
              f"{D['references'][0]} and {D['references'][1]}. On the scored seed H is {h[0.0]:.4f} at gap 0, "
              f"{h[-24.0]:.3f} and {h[24.0]:.3f} at -24 and +24 h, and {h[-168.0]:.3f} and {h[168.0]:.3f} at -168 "
              f"and +168 h. Source: `work/step3_3/score.json` (`tables`, all fix times; the scored seed's values "
              "equal its registered `criteria` H). Limit: gaps with no fix (half a lunation, the map almost "
              "dark) are left out, and the fixes behind each point fall as the gap grows.", ""]
    if FIGS["phase5"][1]():
        D = phase5_data()
        r = D["rock"]
        L += ["## Phase 5 (`fig_phase5.png`)", "",
              f"Safe landings of the hazard detection and divert system against the blind lander, which lands at "
              f"the target, on the scored seed 20261052 by landing epoch (system {', '.join(map(str, D['system']))}"
              f"; blind {', '.join(map(str, D['blind']))} of {D['flights'][0]} each), and the rock-field "
              f"sensitivity on the reference seed {P5_ROCK_SEED} (system "
              f"{', '.join(f'{f} {n}' for f, n in zip(r['fields'], r['system']))}; blind "
              f"{', '.join(f'{f} {n}' for f, n in zip(r['fields'], r['blind']))} of {r['flights'][0]}). Source: "
              "`work/step5_3/score_scored_20261052.json` and `work/step5_2/baselines.json`. The blind lander's count "
              "is the same at every epoch because the epochs fly the same draws, with the same dispersions and "
              "rocks, and the blind lander lands at the target, so it never uses the lit test that differs between "
              "epochs; its failures are on the ground's span. Limit: phase 5 "
              "navigates on the truth, so this is the hazard system alone, before any TRN error.", ""]
    if FIGS["replay"][1]():
        D = replay_data()
        L += ["## Replay checker (`fig_replay.png`)", "",
              "Flights violating each of the seven replay checks, for the registered run on three seeds "
              f"({', '.join(f'{c} {n}' for c, n in zip(D['checks'], D['registered']) if n)}), for its seed "
              f"{P7_SEED} alone ({', '.join(f'{c} {n}' for c, n in zip(D['checks'], D['registered_seed']) if n)}) "
              f"and for the ground-fix sensitivity run of that seed "
              f"({', '.join(f'{c} {n}' for c, n in zip(D['checks'], D['sensitivity']) if n)}); and the violating "
              "flights by primary cause. Source: `work/step7_3/summary.json`, `work/step7_4/diagnosis.json` and "
              "`work/step7_5/compare/summary.json`. Limit: the cause panel sets the registered run's three seeds "
              f"beside its seed {P7_SEED} alone and the sensitivity run of that seed; for check 4 the registered "
              f"record of seed {P7_SEED} alone gives causes by sample, not by flight, so that bar is marked n/a. "
              "Check 3 is judged by each flight's first violation.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- main

FIGS = dict(payoff=(fig_payoff, lambda: all(Path(str(P6_SCORE).format(s)).exists() for s in (P6_SEED,) + P6_REFS)),
            split=(fig_split, lambda: Path(str(P6_SCORE).format(P6_SEED)).exists()),
            premise=(fig_premise, lambda: P3_SCORE.exists()),
            phase5=(fig_phase5, lambda: P5_SCORED.exists() and P5_BASE.exists()),
            replay=(fig_replay, lambda: P7_SUMMARY.exists() and P7_DIAG.exists() and P7_COMPARE.exists()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs"))
    ap.add_argument("--only", default=",".join(FIGS))
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    done, skipped = [], []
    for name in a.only.split(","):
        fn, ok = FIGS[name]
        if not ok():
            skipped.append(name)
            print(f"{name}: skipped, its source file is not in this checkout")
            continue
        t0 = time.time()
        fn(out)
        done.append(name)
        print(f"{name}: {time.time() - t0:.1f} s")
    if a.only == ",".join(FIGS):
        (out / "fig_captions.md").write_text(captions())
    return dict(done=done, skipped=skipped)


if __name__ == "__main__":
    main()
    sys.exit(0)
