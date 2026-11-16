# Result figures

Made by `examples/make_result_figures.py`; every number is read from the committed result file named under each figure, and `tests/test_result_figures.py` checks each figure's data against that file.

## Payoff (`fig_payoff.png`)

The TRN lander's safe-landing fraction and median navigation error at touchdown Q against map gap, for the scored seed 20261062 and the reference seeds 20261060 and 20261061, with each seed's no-TRN lander as a dashed level. On the scored seed the safe fraction is 1.000 at gap 0 and 0.275 at +168 h against 0.620 for the no-TRN lander; median Q is 0.12 m at gap 0 and 608 m at +168 h against 343 m. Source: `work/step6_3/score_<seed>.json`, `summary_6_2`. The number beside each scored seed point is the touchdowns behind its median. Limit: median Q is over the flights that touched down only, 138 of 400 at +168 h on the scored seed, so at large gaps it describes a shrinking and selected subset; aborts and errors are not in it.

## Three way split (`fig_split.png`)

Every flight on the scored seed 20261062, by map gap and for the no-TRN lander, as safe landing, abort to a safe orbit or unsafe outcome (D69). At +168 h: 110 safe, 261 aborts to a safe orbit and 29 unsafe outcomes of 400. Source: `work/step6_3/score_20261062.json`, `three_way`. Limit: unsafe outcome holds unsafe landings, aborts that missed a safe orbit and flights that ended in an error together.

## Premise (`fig_premise.png`)

The fix hazard rate H, the fraction of TRN fixes whose error exceeds the prior's alone, against map gap of both signs, for the scored premise seed 20261022 and the reference seeds 20261020 and 20261021. On the scored seed H is 0.0063 at gap 0, 0.063 and 0.166 at -24 and +24 h, and 0.814 and 0.753 at -168 and +168 h. Source: `work/step3_3/score.json` (`tables`, all fix times; the scored seed's values equal its registered `criteria` H). Limit: gaps with no fix (half a lunation, the map almost dark) are left out, and the fixes behind each point fall as the gap grows.

## Phase 5 (`fig_phase5.png`)

Safe landings of the hazard detection and divert system against the blind lander, which lands at the target, on the scored seed 20261052 by landing epoch (system 100, 100, 100, 100; blind 91, 91, 91, 91 of 100 each), and the rock-field sensitivity on the reference seed 20261050 (system minimum 400, nominal 400, maximum 400; blind minimum 388, nominal 368, maximum 344 of 400). Source: `work/step5_3/score_scored_20261052.json` and `work/step5_2/baselines.json`. The blind lander's count is the same at every epoch because the epochs fly the same draws, with the same dispersions and rocks, and the blind lander lands at the target, so it never uses the lit test that differs between epochs; its failures are on the ground's span. Limit: phase 5 navigates on the truth, so this is the hazard system alone, before any TRN error.

## Replay checker (`fig_replay.png`)

Flights violating each of the seven replay checks, for the registered run on three seeds (3 glide slope 130, 4 terrain clearance 52, 6 site lit 198), for its seed 20261062 alone (3 glide slope 51, 4 terrain clearance 17, 6 site lit 59) and for the ground-fix sensitivity run of that seed (3 glide slope 54, 6 site lit 62); and the violating flights by primary cause. Source: `work/step7_3/summary.json`, `work/step7_4/diagnosis.json` and `work/step7_5/compare/summary.json`. Limit: the cause panel sets the registered run's three seeds beside its seed 20261062 alone and the sensitivity run of that seed; for check 4 the registered record of seed 20261062 alone gives causes by sample, not by flight, so that bar is marked n/a. Check 3 is judged by each flight's first violation.
