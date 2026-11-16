# Descent trajectory visualization

Made by `examples/make_descent_viz.py` from the two flight histories in `docs/viz_data/`; `tests/test_descent_viz.py` checks every plotted value the flight record also holds against the record.

Draw 2 of the scored seed 20261062, the TRN lander, landing epoch 0, chosen by a rule registered before any draw was drawn (D121): the lowest numbered draw whose fresh map flight is a safe landing and whose week-old map flight touches down. It is not a typical week-old map flight. Of the 400 TRN flights flown on this seed with a +168 h map, the registered score counts 261 of 400 (65.2 percent) aborting to a safe orbit, the most common outcome, 110 of 400 (27.5 percent) landing safely and 29 of 400 (7.2 percent) ending unsafely (at landing epoch 0 alone: 52 aborts, 41 safe and 7 unsafe of 100; a flight that ended in an error counts as not a safe landing). Both flights here share their start up to the first fix.

Fresh map (gap 0 h): safe landing, Q 0.09 m, touchdown 8.5 m from the pre-mission target; TRN fixes: 10 s accepted; 20 s accepted; 30 s accepted; 40 s accepted.

Week-old map (gap +168 h): safe landing, Q 96.25 m, touchdown 119.6 m from the pre-mission target; TRN fixes: 10 s accepted; 20 s rejected by nis; 30 s rejected by margin; 40 s rejected by margin. Its first fix was accepted against the week-old map and moved the estimate to 96.2 m from the truth, where it stayed: the next three fixes were rejected. The lander landed where it believed its chosen site to be, and the touchdown lies 95.8 m from that believed site; the ground at the touchdown was safe for the gear.

## Plan view (`descent_plan.png`)

The true path (logged map positions) and the estimated path (the filter's logged estimate) over a hillshade of the 5 m LOLA mosaic, from high gate to touchdown, with the final approach enlarged. The estimate is 686 m from the truth just before the first fix at 10 s; just after it, 2.8 m with the fresh map and 96.2 m with the week-old one. Not shown: the scan frames, the rocks (the hillshade is the mosaic alone), and the half-cell offset of the ground the flights flew over (D109: the flights' ground is the mosaic read half a 5 m cell away; the hillshade is the mosaic in its own place).

## Map against camera (`descent_maps.png`)

At the first fix (10 s), the descent image the camera saw, rendered at the landing epoch, beside the reference map each lander carried (the lit disc fraction at the landing epoch plus its gap, 20 m cells), cropped to the image's ground footprint. The week-old map's shadows differ from what the camera saw. Both fixes were accepted here, the fresh map's leaving the estimate 2.8 m from the truth and the week-old map's 96.2 m. Not shown: the later fixes, where the week-old map's fixes were rejected, and the match itself (the search over offsets). The camera image is in camera pixels, so its orientation is the camera's, not the map's.

## Side view (`descent_side.png`)

Horizontal distance to the site in force against height above it, with the 80 degree glide-slope cone. The estimated path is drawn about the site guidance flew to, in the lander's frame (the replay checker's check 3); the true path about the site's true position in force (D81's form; the pre-mission target before the landing decision). Both axes are linear near zero and logarithmic beyond. Not shown: the terrain under the path; heights are relative to the site's logged height.

## Landing close-up (`descent_hazard.png`)

The landing decision's hazard map in the lander's estimated frame: cells detected safe and lit, detected safe but dark, and not detected safe, with the chosen site, its true position and the touchdown. In the week-old map flight the map is drawn where the lander believed it was, about 96 m from where it truly was. Not shown: which of the not-safe cells were unsafe and which unverified (the log keeps one flag), the rocks, and the scan frames that built the map.

## Animation (`descent_anim.mp4`, `descent_anim.gif`)

Both flights time-synchronised from high gate: the whole path in plan view, the final approach at a fixed scale and height above the target against time; one second of flight per frame to 100 s, a quarter second per frame after. Not shown: the scan frames, the rocks, the hazard map and the D109 offset.
