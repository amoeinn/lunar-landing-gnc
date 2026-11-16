# Descent trajectory visualization

Made by `examples/make_traj_viz.py` from the six flight histories in `docs/viz_data/` and the two reference maps in `docs/viz_data/maps/`; `tests/test_traj_viz.py` checks every plotted value the flight record also holds against the record.

Three draws of the scored seed 20261062, the TRN lander, landing epoch 0, each chosen by a rule registered before it was drawn, among the draws whose fresh map flight is a safe landing (flights that ended in an error are passed over): the lowest numbered draw whose week-old map (+168 h) flight aborts to a safe orbit (D126, draw 0), touches down (D121, draw 2) and ends in an unsafe landing (D126, draw 18). Each is shown beside the same draw flown with a fresh map; both flights of a draw share their start up to the first fix. Each is labelled with its outcome's count among the seed's week-old map TRN flights in the registered score: aborts to a safe orbit and safe landings from its three way split (D69), unsafe landings alone from its outcome counts, with the 1 flight that ended in an error stated separately. None of the three is typical of anything but its own outcome.

**Draw 0 (D126): week-old map abort to a safe orbit.** Registered score, abort to a safe orbit: 261 of 400 of this seed's +168 h TRN flights (52 of 100 at epoch 0). Week-old map: abort (hazard trigger at 72 s) to a safe orbit; TRN fixes 10 s accepted; 20 s rejected by nis; 30 s rejected by nis; 40 s rejected by margin; 0.5 s after the first fix the estimate was 767.4 m from the truth (fresh map: 0.0 m). Its landing decision found 0 eligible sites, so it aborted at 72 s. The record gives the outcome, a safe orbit, and the orbit, perilune 16.7 km and apolune 83.4 km. The drawn rise and burn end where the log ends, at t = 485.6 s, 338.9 km east and 17.0 km up. The same draw with a fresh map: safe landing, Q 0.17 m, 28.1 m from the target.

**Draw 2 (D121): week-old map safe landing.** Registered score, safe landing: 110 of 400 of this seed's +168 h TRN flights (41 of 100 at epoch 0). Week-old map: safe landing, Q 96.25 m, 119.6 m from the target; TRN fixes 10 s accepted; 20 s rejected by nis; 30 s rejected by margin; 40 s rejected by margin; 0.5 s after the first fix the estimate was 96.2 m from the truth (fresh map: 2.8 m). It touched down 1.573 m from its chosen site's true position and 95.8 m from where it believed that site was. It passed every landing check. The same draw with a fresh map: safe landing, Q 0.09 m, 8.5 m from the target.

**Draw 18 (D126): week-old map unsafe landing.** Registered score, unsafe landing: 28 of 400 of this seed's +168 h TRN flights (6 of 100 at epoch 0); besides these, 1 ended in an error (1 at epoch 0). Week-old map: unsafe landing, Q 517.43 m, 531.9 m from the target; TRN fixes 10 s accepted; 20 s rejected by nis; 30 s rejected by nis; 40 s rejected by nis; 0.5 s after the first fix the estimate was 517.6 m from the truth (fresh map: 1.1 m). It touched down 2.137 m from its chosen site's true position and 518.9 m from where it believed that site was. It failed criterion 2 by 0.045 m: touchdown 2.137 m from the chosen site's true position, against the 2.092 m radius. Its ground was gear safe. Of the 28 unsafe landings at +168 h on this seed, 27 touched down on gear-unsafe ground; this is the single one that did not, the registered score's one case off the verified site. The same draw with a fresh map: safe landing, Q 0.13 m, 8.4 m from the target.

## Plan view (`traj_plan.png`)

One row per draw: the final approach of each flight, the true path (logged map positions; an abort's rise and burn in purple, followed until it leaves the view) and the estimated path (the filter's logged estimate), with the TRN fixes marked on the true path at their times, the chosen site in the lander's frame, its true position and the touchdown. The backdrop is a hillshade of the 5 m LOLA mosaic itself (D128).

## Map against camera (`traj_maps.png`)

One row per draw: the descent image the camera saw at the first fix beside the reference map each lander carried (the lit disc fraction at the landing epoch plus its gap, 20 m cells), cropped to the image's ground footprint. The camera image is in camera pixels.

## Side view (`traj_side.png`)

One row per draw, two linear panels (D129): the whole path, and the last 300 m with the 80 degree glide-slope cone drawn straight. Horizontal distance to the site in force against height above it: true paths about the site's true position in force (D81's form; the pre-mission target before the landing decision, and throughout a flight that chose no site), estimated paths about the site guidance flew to (the replay checker's check 3), an abort's rise and burn about the pre-mission target. Heights are relative to the site's logged height.

## Landing close-up (`traj_hazard.png`)

The landing decision's hazard map for each week-old map flight, in the lander's estimated frame where the map lives: cells detected safe and lit, detected safe but dark, and not detected safe, with the chosen site and the believed touchdown (the estimate at contact). An arrow points from the believed touchdown to the true touchdown, Q away in the map frame, and stops at the panel's edge when the true touchdown is off the panel.

## Animation (`traj_anim.mp4`, `traj_anim.gif`)

The three draws side by side, each with its fresh map and week-old map flights, time-synchronised from high gate: one second of flight per frame to 60 s, a quarter second per frame after, to the last touchdown.

## Limits

- No scan frames are shown, nor the search over offsets inside a TRN match; a fix is drawn only as its time and outcome.
- The terrain backdrop is the 5 m mosaic without the rock field and without the half-cell ground offset of D109, which is invisible at this scale.
- The descent image is rendered from the gap-0 map of the landing epoch, so the fresh map matches it by construction; the week-old map's shadows are a week later.
- Which of the hazard map's not-safe cells were unsafe and which unverified is not separable: the log keeps one flag.
- The abort's burn is drawn as far as the history holds it; the orbit it reaches is the record's, not drawn.
- Each draw is one flight per map, not a distribution: the shares are the distribution.
