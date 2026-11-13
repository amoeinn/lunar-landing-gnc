# Step 7.4: diagnosis of the 380 replay violations

Read only. The 7.3 verdict stays FAIL; nothing is rescored, and no checker,
tolerance or flight code is changed. Every number below is from
`work/step7_4/diagnosis.json`, built by `diag_summary.py` from the outputs of
`diag_lit.py` (check 6), `diag_clearance.py` and `diag_ground_shift.py`
(check 4) and `diag_glide.py` (check 3). Nothing new was flown. Check 3
rebuilds guidance plans by calling `closed_loop.resolve` on logged states;
that is a solve, not a flight.

## Reproduction checks (each diagnosis against the log it reads)

| Check | What was rebuilt | Against | Largest difference |
|---|---|---|---|
| 6 | the checker's lit margin at the touchdown point and time | the margin 7.3 wrote | 0.0 deg, all 198 |
| 6 | the flight's lit test on the chosen point alone | the flight's own recorded `decision.lit[index]` | 198 of 198 equal |
| 4 | the flight's D5 ground, rebuilt as `phase6_flights` makes it | the flight's recorded `min_clearance` | 0.0 m, all 50 with one (the 2 strikes record none) |
| 3 | the plan in force, from `step_nav` and `step_remaining` | the hold's logged command, and the next step's remaining count | 0.0 N; 0 mismatches |
| 3 | the plan's path between nodes (constant `u_k + g`) | the plan's next node | 9.6e-10 m |
| 3 | the estimate at the solve | the plan's node 0 | 0.0 m |

## Breakdowns

| | Check 3 glide slope | Check 4 terrain clearance | Check 6 site lit |
|---|---|---|---|
| Violations | 130 | 52 | 198 |
| Seed 20261060 / 61 / 62 | 41 / 38 / 51 | 18 / 17 / 17 | 76 / 63 / 59 |
| TRN / no-TRN | 126 / 4 | 43 / 9 | 181 / 17 |
| Gap -168 / -72 / -24 / 0 / 24 / 72 / 168 h | 11 / 65 / 23 / 0 / 7 / 10 / 10 | 2 / 2 / 6 / 9 / 16 / 7 / 1 | 0 / 0 / 32 / 38 / 19 / 16 / 76 |
| No-TRN (no gap) | 4 | 9 | 17 |
| Epoch 0 / 1 / 2 / 3 | 33 / 39 / 37 / 21 | 11 / 9 / 10 / 22 | 193 / 2 / 0 / 3 |
| At gap 0 | 0 | 9 | 38 |
| Outcome | 91 unsafe, 37 safe, 2 abort | 46 safe, 4 unsafe, 2 strike | 198 safe |

## A. Check 6, site lit (198)

Four points per flight:
- the chosen site, in the frame the flight estimated;
- the centre of the 20 m cell holding it, which is what the flight's test evaluates;
- the site's true position (D26);
- the true touchdown point.

Each point is evaluated twice, at the epoch (t = 0) and at the touchdown time,
by two methods: the checker's (E's ray march with SPICE, and D91's tolerance)
and the flight's (`landing.lit_at`).

| Point | Lit by the checker (t0 / td) | Lit by the flight's test (t0 / td) |
|---|---|---|
| chosen site | 131 / 131 | 198 / 198 |
| chosen cell centre | 198 / 198 | 198 / 198 |
| true site | 8 / 8 | 58 / 58 |
| true touchdown | 0 / 0 | 59 / 59 |

Distances, m (min / quartiles / max):

| Pair | min | 25 % | median | 75 % | max |
|---|---|---|---|---|---|
| chosen site to true site | 0.02 | 5.30 | 25.24 | 74.28 | 1705.90 |
| chosen site to touchdown | 0.006 | 5.34 | 24.70 | 73.31 | 1705.87 |
| true site to touchdown | 0.004 | 0.44 | 0.68 | 1.07 | 2.06 |
| chosen cell centre to touchdown | 5.22 | 12.76 | 30.15 | 70.35 | 1707.10 |

How each cause was tested:
- **1, navigation:** the touchdown is in another 20 m cell. The chosen cell centre is lit by the checker at the touchdown time, and the touchdown point is not.
- **2, method:** the same point (the chosen cell centre) at the time the flight tests it. The flight's test says lit and the checker's says not.
- **3, cell versus point:** the touchdown is in the chosen cell, the cell centre is lit by the checker, and the touchdown point is not.
- **4, time:** the touchdown point is lit at t0 and not at the touchdown time.

| Cause | Flights | TRN / no-TRN | Chosen site to touchdown, m (min / median / max) |
|---|---|---|---|
| 1 navigation displacement | 147 | 130 / 17 | 1.62 / 60.55 / 1705.87 |
| 2 method disagreement | 0 | | |
| 3 cell versus point | 51 | 51 / 0 | 0.006 / 0.32 / 15.44 |
| 4 time | 0 | | |
| fits none | 0 | | |

The flags never overlap: each of the 198 flights carries exactly one.

By gap, navigation displacement accounts for:
- 76 of 76 at 168 h;
- 16 of 16 at 72 h;
- 11 of 19 at 24 h;
- 26 of 32 at -24 h;
- 1 of 38 at 0 h;
- all 17 no-TRN flights.

Cell versus point accounts for 37 of the 38 flights at gap 0.

Why causes 2 and 4 are zero, and whether those tests could fire:
- **Method:** both tests agree on every chosen cell centre. They are computed independently and their horizons there differ by up to 1.36 deg, but in every case the sun was above both.
- **Time:** from t0 to the touchdown time the checker's margin at the touchdown point moves by only -0.0003 to 0.0049 deg. Time can only decide a point whose margin lies within that band of zero, and none does.

The true touchdown lies within 2.06 m of the true site in every flight. So the
lander flies to its chosen site well, and the true site sits where the map
placement error puts it.

What it says:
- **The checker is not wrong at any point:** at the same point and time the two lit tests agree.
- **The lander's lit test is wrong for the point it lands on, in two ways:**
  - it tests the chosen site in the estimated frame, while the lander lands at the true site, up to 1706 m away (147 flights);
  - it tests a 20 m cell centre, while the touchdown point in that same cell is dark (51 flights). In 125 of the 147 navigation flights the chosen point itself was lit by the checker; in the other 22, and in 45 of the 51, the chosen point itself was already dark at its exact position.

## B. Check 4, terrain clearance (52 flights, 139 samples)

| Cause | Samples | Flights |
|---|---|---|
| 1 terrain strike, confirmed (the flight recorded it; the flight's own ground is above the sample) | 59 | 2 |
| 2 ground-model difference | 80 | 50 |
| 3 real pass below the flight's ground, not recorded | 0 | 0 |
| 4 other | 0 | 0 |

The two strikes:

| Flight | Samples | Flight's clearance there, m | Checker ground less flight ground, m |
|---|---|---|---|
| 20261060 d71_trn_e1_g168.0 e1 | 36 guided | -7.99 to -1.13 | -0.82 to -0.40 |
| 20261061 d43_trn_e0_g-24.0 e0 | 23 guided | -1.35 to -0.23 | -0.105 to -0.090 |

The ground-model difference (80 samples, all of them descent samples):

| Quantity | min | 25 % | median | 75 % | max |
|---|---|---|---|---|---|
| checker ground less flight ground, m | 0.028 | 0.050 | 0.097 | 0.145 | 0.308 |
| checker ground less flight ground without rocks, m | 0.028 | 0.051 | 0.100 | 0.145 | 0.308 |
| rock height, m | 0 | 0 | 0 | 0 | 0.123 (one sample) |
| the flight's own clearance, m | 0.0002 | 0.007 | 0.022 | 0.055 | 0.240 |
| the checker's clearance, m | -0.305 | -0.091 | -0.049 | -0.035 | -0.022 |

Ticks before contact: 50 samples are 1 tick out, 17 are 2, 7 are 3, 4 are 4, one is 5 and one is 6.

Rocks contribute nothing (one sample), so the difference was measured
(`diag_ground_shift.py`). It is a placement offset of the flight's ground, not
a modelling difference: the flight's D5 ground at (x, y) equals the checker's
bilinear 5 m mosaic read at (x + 2.5, y - 2.5) m.

| Tile | Window `mosaic_base` reads (col_off, row_off) | Its transform origin (x0, y1) | Best shift | Largest difference there |
|---|---|---|---|---|
| (0, 0) | 20319.5, 18674.5 | 1597.5, 6627.5 | +2.5, -2.5 | 5.7e-14 m |
| (1, 0) | 20349.5, 18674.5 | 1747.5, 6627.5 | +2.5, -2.5 | 5.7e-14 m |
| (-3, 2) | 20229.5, 18614.5 | 1147.5, 6927.5 | +2.5, -2.5 | 5.7e-14 m |
| (5, -4) | 20469.5, 18794.5 | 2347.5, 6027.5 | +2.5, -2.5 | 1.7e-13 m |
| (-8, 0) | 20079.5, 18674.5 | 397.5, 6627.5 | +2.5, -2.5 | 2.8e-14 m |
| (0, 8) | 20319.5, 18434.5 | 1597.5, 7827.5 | +2.5, -2.5 | 2.8e-14 m |

Why the ground is offset: `truthsurface.mosaic_base` builds its window from
bounds centred on the target (1682.5, 6542.5), which lies half a 5 m cell off
the mosaic's grid. Every tile centre is a multiple of 150 m from the target, so
every tile inherits the same half-pixel offset. The window's transform keeps
the fractional origin, but the data read come from whole-pixel offsets.

Read at the shifted point, the checker's ground puts none of the 80 samples
beyond tolerance. The 59 strike samples stay beyond it.

What it says:
- **The checker reads the mosaic correctly.** The flight's truth ground (D5) is the mosaic displaced 2.5 m west and 2.5 m north, so its height at a point is the mosaic's height 2.5 m east and 2.5 m south of it.
- **The flight's physics are self-consistent.** Lidar, contact and clearance all use that one ground, so this is a fault in how the flight builds its truth, not in what it flies.
- **The two strikes are real** on either ground.
- **No unrecorded pass below ground** was found.

## C. Check 3, glide slope (130)

At each flight's first violating tick and its worst tick:
- the plan in force is rebuilt;
- its nodes are judged against guidance's own cone, from node 1 on;
- its own path at the tick is computed;
- the estimate's movement since that solve is split into two parts:
  - the true lander's departure from the plan, which the filter's propagation follows;
  - the change in the estimate's error (estimate less truth), which comes from the updates and the propagation's own drift.

How each cause was tested:
- **Plan nodes outside:** some node margin is below -1e-6 deg.
- **2, between-node:** the nodes are inside, and the plan's own path at the tick is outside the cone.
- **3, divert switch:** the plan in force is the divert's and the tick is in the switch step.
- **1, estimate jump:** the plan path is inside the cone, and the estimate with its error held at the solve's value would pass. The movement is then due to the change in error.
- **Dynamics:** as cause 1, but the error held at the solve's value would still fail.
- **4, near the apex:** range below `ALT_END` = 45.72 m. This is a flag only; the threshold is mine, and is one of the questions below.

| Cause | First tick | Worst tick |
|---|---|---|
| plan nodes outside the cone | 0 | 0 |
| 2 between-node path | 105 | 128 |
| of which the plan's own path is beyond the tick's tolerance | 24 | 97 |
| of which the plan is below the cone within tolerance, the estimate's change taking it beyond | 81 | 31 |
| 3 divert switch | 0 | 0 |
| 1 estimate jump after the solve | 25 | 2 |
| true lander's departure from the plan | 0 | 0 |
| 4 near the apex (flag, co-occurring) | 78 | 70 |
| 5 other / fits none | 0 | 0 |

| At the worst tick | min | 25 % | median | 75 % | max |
|---|---|---|---|---|---|
| time since the solve, s | 0.05 | 0.90 | 1.23 | 1.40 | 2.00 |
| plan's least node margin, deg | -3.9e-08 | 7.2e-10 | 5.4e-09 | 3.7e-08 | 14.55 |
| plan's own path margin at the tick, deg | -4.26 | -1.28 | -0.64 | -0.33 | 0.90 |
| node 0 (the measured state) margin, deg | -3.16 | -0.27 | 0.53 | 1.79 | 16.63 |
| range to the site in force, m | 7.6 | 23.7 | 42.9 | 65.6 | 1273.7 |
| height above it, m | 0.70 | 3.71 | 6.77 | 10.88 | 224.25 |
| truth's departure from the plan's motion, m | 0.002 | 0.058 | 0.092 | 0.119 | 0.288 |
| change in the estimate's up error, m (0, 10, 50, 90, 100 %) | -5.04 | -0.278 | -0.098 | -0.009 | 0.033 |

More findings:
- **The plan in force:**
  - it is a re-solve 4 to 14 steps after the divert (the 4 no-TRN flights have no divert);
  - it is never the divert's own plan;
  - no TRN fix falls between a solve and its tick.
- **First ticks classed as estimate jumps (25):**
  - the change in the estimate's up error was -4.41 to -0.195 m (median -0.35);
  - the time since the solve was 1.1 to 2.0 s.
- **The lowest point of the estimated path above the site in force, after the decision:**
  - median 6.17 m, range 0.67 to 173.3 m, and below 10 m in 95 of the 130;
  - it falls a median 17.3 s before guided flight ends;
  - guided flight then ends a median 45.5 m above the site;
  - median lowest height by gap: -72 h 5.48 m, -24 h 5.17 m, 24 h 4.80 m, 72 h 10.15 m, 168 h 21.07 m, -168 h 46.28 m, no-TRN 13.07 m.

What it says:
- **Guidance enforces the cone at its 2 s nodes, from node 1 on, and the plan meets it there.** The least node margin is -3.9e-08 deg, and in at least three quarters of the worst ticks the plan's least node margin is within 4e-8 deg of zero: a node sits on the cone's edge.
- **The plan's own path between nodes 0 and 1 bulges below the cone by up to 4.26 deg.** That alone takes 97 of the 130 worst ticks beyond tolerance.
- **The check judges every 20 Hz tick, so it is stricter than the constraint the lander flies.** The checker is not in error.
- **The plans that bulge dive low near the site and climb back to `ALT_END`.** The estimated path comes within a few metres of the site's height some 17 s before guided flight ends, and the violations sit there.
- **The estimate's updates move the estimate down relative to the truth between solves.** That decides the 25 first ticks and adds to the 31 worst ticks where the plan is within tolerance.
- **The altimeter and Doppler parts of that change cannot be separated from the logs.** Both update at every 20 Hz tick, and the history keeps only the estimate after each tick.

## Run times and memory (each under `systemd-run --user --scope -p MemoryMax=8G`, 30 minute watchdog)

| Script | Wall time | Workers | Peak per worker |
|---|---|---|---|
| diag_lit.py | 683 s | 3 | 602 MB |
| diag_clearance.py | 670 s | 1 | 1,905 MB |
| diag_ground_shift.py | 2.5 s | 1 | |
| diag_glide.py | 6 s | 4 | 107 MB |

## Questions (design questions, listed and not decided)

1. **The D5 ground is offset half a 5 m cell from the mosaic, from `truthsurface.mosaic_base`'s fractional window.** Is D5's ground defined as the mosaic, which would make this a flight-code fault whose fix changes every history and needs a re-fly? Or should the checker take the flight's placement? 80 of check 4's 139 samples hang on this.
2. **Is check 3's requirement continuous, judged at every 20 Hz tick, or nodal, the way guidance imposes it?** If continuous, the lander has to change: denser nodes, a path constraint between nodes, or a constraint on node 0. If nodal, does the checker need a registered term between nodes? Both are design choices.
3. **Should the lit requirement apply to the true touchdown point, as check 6 judges it, or to the flight's design, the chosen 20 m cell centre in the estimated frame?** If the former, the lander's options include a lit test over the placement-uncertainty footprint and a lit test at points finer than 20 m cells.
4. **Is the low pass over the site before the climb to `ALT_END` an accepted trajectory?** It occurs in 95 of 130 check 3 flights, with the estimated path below 10 m above the site.
5. **Do you want the altimeter and Doppler parts of the estimate's change separated?** That needs one flight re-flown with the filter's per-update corrections logged. That is flying something new, so I did not do it.
6. **What should "near the apex" mean?** I flagged ticks with range below `ALT_END` (45.72 m). The threshold is mine and is not registered.
7. **Do you want the 25 first-tick estimate jumps looked into?** The estimate settles 0.2 to 4.4 m downward relative to the truth within 2 s of a solve, at ranges from 10 m to 1109 m.
