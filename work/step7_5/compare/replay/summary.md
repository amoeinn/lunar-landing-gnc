# Step 7.3: replay checker over the re-flown histories

Bar: zero violations beyond the registered tolerances on checks 1, 2, 4, 5, 6 and 7, and on check 3 as judged (D97); an unjudged judged check counts against the bar (D105).

**Verdict: FAIL**

## Integrity

| Seed | Jobs | Histories | Flights judged | Error flights (D107) |
|---|---|---|---|---|
| 20261060 | 0 | 0 | 0 | 0 |
| 20261061 | 0 | 0 | 0 | 0 |
| 20261062 | 2900 | 2892 | 3189 | 8 |

Checker errors: 0. Load errors: 0.

Error flights, excluded (D107):

- 20261062 d28_trn_e1_g-24.0: ValueError: scan plan reaches slant 144.4 m at t = 74.35 s
- 20261062 d28_trn_e1_g72.0: ValueError: scan plan reaches slant 128.5 m at t = 74.65 s
- 20261062 d34_trn_e0_g168.0: ValueError: scan plan reaches slant 129.8 m at t = 94.95 s
- 20261062 d37_trn_e3_g72.0: ValueError: scan plan reaches slant 146.5 m at t = 78.85 s
- 20261062 d61_trn_e3_g24.0: ValueError: scan plan reaches slant 126.7 m at t = 72.25 s
- 20261062 d63_trn_e3_g24.0: ValueError: scan plan reaches slant 120.3 m at t = 82.25 s
- 20261062 d83_notrn_eNone_gNone: ValueError: scan plan reaches slant 142.3 m at t = 76.25 s
- 20261062 d96_trn_e3_g24.0: ValueError: scan plan reaches slant 137.7 m at t = 76.05 s

## Per check

| Check | Seed | PASS | FAIL | Other | Smallest margin over tolerance | Its tolerance | Flight |
|---|---|---|---|---|---|---|---|
| 1 mass flow | 20261060 |  |  |  |  |  |  |
| 1 mass flow | 20261061 |  |  |  |  |  |  |
| 1 mass flow | 20261062 | 3189 |  |  | 3.229e-12 | 4.093e-12 | 20261062 d70_trn_e2_g-168.0 e2 |
| 1 mass flow | total | 3189 |  |  | 3.229e-12 | 4.093e-12 | 20261062 d70_trn_e2_g-168.0 e2 |
| 2 thrust and pointing | 20261060 |  |  |  |  |  |  |
| 2 thrust and pointing | 20261061 |  |  |  |  |  |  |
| 2 thrust and pointing | 20261062 | 3189 |  |  | 4.404e-12 | 6.223e-12 | 20261062 d05_notrn_eNone_gNone e0 |
| 2 thrust and pointing | total | 3189 |  |  | 4.404e-12 | 6.223e-12 | 20261062 d05_notrn_eNone_gNone e0 |
| 3 glide slope | 20261060 |  |  |  |  |  |  |
| 3 glide slope | 20261061 |  |  |  |  |  |  |
| 3 glide slope | 20261062 | 3135 | 54 |  | -3.11 | 1.906 | 20261062 d59_trn_e0_g24.0 e0 |
| 3 glide slope | total | 3135 | 54 |  | -3.11 | 1.906 | 20261062 d59_trn_e0_g24.0 e0 |
| 4 terrain clearance | 20261060 |  |  |  |  |  |  |
| 4 terrain clearance | 20261061 |  |  |  |  |  |  |
| 4 terrain clearance | 20261062 | 3189 |  |  | 0.02171 | 0.02165 | 20261062 d04_trn_e1_g72.0 e1 |
| 4 terrain clearance | total | 3189 |  |  | 0.02171 | 0.02165 | 20261062 d04_trn_e1_g72.0 e1 |
| 5 touchdown velocity | 20261060 |  |  |  |  |  |  |
| 5 touchdown velocity | 20261061 |  |  |  |  |  |  |
| 5 touchdown velocity | 20261062 | 3189 |  |  | 1.105 | 1.009e-08 | 20261062 d30_trn_e1_g-72.0 e1 |
| 5 touchdown velocity | total | 3189 |  |  | 1.105 | 1.009e-08 | 20261062 d30_trn_e1_g-72.0 e1 |
| 6 site lit | 20261060 |  |  |  |  |  |  |
| 6 site lit | 20261061 |  |  |  |  |  |  |
| 6 site lit | 20261062 | 3127 | 62 |  | -3.303 | 0.0006199 | 20261062 d92_trn_e0_g168.0 e0 |
| 6 site lit | total | 3127 | 62 |  | -3.303 | 0.0006199 | 20261062 d92_trn_e0_g168.0 e0 |
| 7 triggers | 20261060 |  |  |  |  |  |  |
| 7 triggers | 20261061 |  |  |  |  |  |  |
| 7 triggers | 20261062 | 3189 |  |  |  |  |  |
| 7 triggers | total | 3189 |  |  |  |  |  |
| 2 tilt (deg) | 20261060 |  |  |  |  |  |  |
| 2 tilt (deg) | 20261061 |  |  |  |  |  |  |
| 2 tilt (deg) | 20261062 |  |  |  | 3.544e-05 | 3.481e-05 | 20261062 d47_trn_e1_g-72.0 e1 |
| 2 tilt (deg) | total |  |  |  | 3.544e-05 | 3.481e-05 | 20261062 d47_trn_e1_g-72.0 e1 |
| 2 abort engines (relative) | 20261060 |  |  |  |  |  |  |
| 2 abort engines (relative) | 20261061 |  |  |  |  |  |  |
| 2 abort engines (relative) | 20261062 |  |  |  | 1.332e-15 | 1.332e-15 | 20261062 d00_notrn_eNone_gNone e0 |
| 2 abort engines (relative) | total |  |  |  | 1.332e-15 | 1.332e-15 | 20261062 d00_notrn_eNone_gNone e0 |
| 3 D81 form, reported (deg) | 20261060 |  |  |  |  |  |  |
| 3 D81 form, reported (deg) | 20261061 |  |  |  |  |  |  |
| 3 D81 form, reported (deg) | 20261062 |  |  |  | -6.85 | 0.03915 | 20261062 d53_notrn_eNone_gNone e0 |
| 3 D81 form, reported (deg) | total |  |  |  | -6.85 | 0.03915 | 20261062 d53_notrn_eNone_gNone e0 |

## Check 3: D81's form, reported, below the cone

- 20261060: 0 flights; by outcome {}
- 20261061: 0 flights; by outcome {}
- 20261062: 70 flights; by outcome {'safe landing': 30, 'abort': 40}

## Unjudged items

- 20261060: 
- 20261061: 
- 20261062: 1 holds without a step count: 0; 2 guided holds without a step estimate: 0; 3 guided ticks without an estimate (D105): 0; 4 off-DEM samples below 10,786 m (D87): 0; 4 off-DEM samples above 10,786 m, passed by the bound (D87, not unjudged): 447197

Flights with a check neither PASS nor FAIL: 0


## Violations (116)

| Flight | Outcome | Check | Margin | Tolerance | Over | Items beyond | Where |
|---|---|---|---|---|---|---|---|
| 20261062 d01_trn_e0_g-24.0 e0 | safe landing | 6 site lit | -0.4601 | 0.03921 | -0.4208 |  | sun_elevation 1.704; horizon 2.164; sun_azimuth 15.34; margin_lower_limb -0.727; lit_disc_fraction 0; ray_spacing_deg 0.0001536; march_step_deg 0.03906; samples_outside_grid 0 |
| 20261062 d03_trn_e0_g168.0 e0 | safe landing | 6 site lit | -2.422 | 0.1513 | -2.271 |  | sun_elevation 1.706; horizon 4.128; sun_azimuth 15.12; margin_lower_limb -2.689; lit_disc_fraction 0; ray_spacing_deg 4.027e-06; march_step_deg 0.1513; samples_outside_grid 0 |
| 20261062 d03_trn_e1_g-24.0 e1 | unsafe landing | 3 glide slope | -1.639 | 0.5639 | -1.075 | 33 | lowest_elevation_margin_deg -1.639; d81_over_deg 1.658 |
| 20261062 d03_trn_e3_g-72.0 e3 | safe landing | 3 glide slope | -0.8014 | 0.4006 | -0.4009 | 25 | lowest_elevation_margin_deg -0.8032; d81_over_deg 0.6796 |
| 20261062 d05_trn_e0_g0.0 e0 | safe landing | 6 site lit | -0.2311 | 0.04446 | -0.1867 |  | sun_elevation 1.704; horizon 1.935; sun_azimuth 15.38; margin_lower_limb -0.4981; lit_disc_fraction 0.03046; ray_spacing_deg 0.0001617; march_step_deg 0.04429; samples_outside_grid 0 |
| 20261062 d06_trn_e0_g-24.0 e0 | safe landing | 6 site lit | -0.2594 | 0.08601 | -0.1734 |  | sun_elevation 1.705; horizon 1.964; sun_azimuth 15.4; margin_lower_limb -0.5263; lit_disc_fraction 0.003837; ray_spacing_deg 0.0005158; march_step_deg 0.0855; samples_outside_grid 0 |
| 20261062 d06_trn_e0_g168.0 e0 | safe landing | 6 site lit | -1.344 | 0.08827 | -1.255 |  | sun_elevation 1.705; horizon 3.049; sun_azimuth 15.23; margin_lower_limb -1.611; lit_disc_fraction 0; ray_spacing_deg 0.0002655; march_step_deg 0.088; samples_outside_grid 0 |
| 20261062 d07_trn_e0_g168.0 e0 | safe landing | 6 site lit | -2.493 | 0.09695 | -2.396 |  | sun_elevation 1.707; horizon 4.199; sun_azimuth 15.19; margin_lower_limb -2.76; lit_disc_fraction 0; ray_spacing_deg 6.033e-05; march_step_deg 0.09689; samples_outside_grid 0 |
| 20261062 d08_trn_e0_g0.0 e0 | safe landing | 6 site lit | -0.1455 | 0.06733 | -0.0782 |  | sun_elevation 1.705; horizon 1.85; sun_azimuth 15.42; margin_lower_limb -0.4125; lit_disc_fraction 0.1729; ray_spacing_deg 0.000343; march_step_deg 0.06698; samples_outside_grid 0 |
| 20261062 d10_trn_e0_g168.0 e0 | safe landing | 6 site lit | -1.081 | 0.1771 | -0.904 |  | sun_elevation 1.706; horizon 2.787; sun_azimuth 15.27; margin_lower_limb -1.348; lit_disc_fraction 0; ray_spacing_deg 0.0001076; march_step_deg 0.177; samples_outside_grid 0 |
| 20261062 d12_trn_e0_g-24.0 e0 | safe landing | 6 site lit | -0.9821 | 0.01183 | -0.9703 |  | sun_elevation 1.705; horizon 2.687; sun_azimuth 15.28; margin_lower_limb -1.249; lit_disc_fraction 0; ray_spacing_deg 0.008452; march_step_deg 0.003379; samples_outside_grid 0 |
| 20261062 d12_trn_e0_g0.0 e0 | safe landing | 6 site lit | -0.2238 | 0.01338 | -0.2104 |  | sun_elevation 1.705; horizon 1.929; sun_azimuth 15.59; margin_lower_limb -0.4908; lit_disc_fraction 0.03811; ray_spacing_deg 6.22e-07; march_step_deg 0.01338; samples_outside_grid 0 |
| 20261062 d13_trn_e0_g-24.0 e0 | unsafe landing | 3 glide slope | -0.4248 | 0.2677 | -0.1571 | 19 | lowest_elevation_margin_deg -0.4259; d81_over_deg 1.019 |
| 20261062 d13_trn_e0_g0.0 e0 | safe landing | 6 site lit | -0.2803 | 0.01898 | -0.2614 |  | sun_elevation 1.704; horizon 1.984; sun_azimuth 15.36; margin_lower_limb -0.5473; lit_disc_fraction 0; ray_spacing_deg 0.0001539; march_step_deg 0.01883; samples_outside_grid 0 |
| 20261062 d13_trn_e0_g72.0 e0 | unsafe landing | 3 glide slope | -0.3132 | 0.2635 | -0.04971 | 10 | lowest_elevation_margin_deg -0.3132; d81_over_deg 0.1789 |
| 20261062 d13_trn_e1_g-72.0 e1 | unsafe landing | 3 glide slope | -0.4977 | 0.238 | -0.2597 | 27 | lowest_elevation_margin_deg -0.4977; d81_over_deg 0.9461 |
| 20261062 d13_trn_e1_g72.0 e1 | unsafe landing | 3 glide slope | -0.4324 | 0.2574 | -0.1749 | 20 | lowest_elevation_margin_deg -0.4324; d81_over_deg 0.2661 |
| 20261062 d13_trn_e3_g-72.0 e3 | safe landing | 3 glide slope | -0.4133 | 0.3142 | -0.09912 | 14 | lowest_elevation_margin_deg -0.4133; d81_over_deg 0.2274 |
| 20261062 d14_trn_e1_g-72.0 e1 | unsafe landing | 3 glide slope | -1.185 | 0.3479 | -0.837 | 39 | lowest_elevation_margin_deg -1.185; d81_over_deg 0.7605 |
| 20261062 d14_trn_e3_g-72.0 e3 | safe landing | 3 glide slope | -0.4796 | 0.3108 | -0.1688 | 17 | lowest_elevation_margin_deg -0.4796; d81_over_deg 0.659 |
| 20261062 d15_trn_e1_g72.0 e1 | unsafe landing | 3 glide slope | -0.6588 | 0.6081 | -0.05071 | 5 | lowest_elevation_margin_deg -0.6588; d81_over_deg 3.069 |
| 20261062 d18_trn_e0_g-24.0 e0 | safe landing | 6 site lit | -0.2171 | 0.01219 | -0.2049 |  | sun_elevation 1.705; horizon 1.922; sun_azimuth 15.59; margin_lower_limb -0.4841; lit_disc_fraction 0.04707; ray_spacing_deg 2.204e-06; march_step_deg 0.01219; samples_outside_grid 0 |
| 20261062 d19_trn_e0_g-24.0 e0 | safe landing | 6 site lit | -0.3051 | 0.06687 | -0.2383 |  | sun_elevation 1.705; horizon 2.011; sun_azimuth 15.4; margin_lower_limb -0.5721; lit_disc_fraction 0; ray_spacing_deg 0.0002761; march_step_deg 0.06659; samples_outside_grid 0 |
| 20261062 d19_trn_e0_g168.0 e0 | safe landing | 6 site lit | -0.7692 | 0.08323 | -0.6859 |  | sun_elevation 1.706; horizon 2.475; sun_azimuth 15.37; margin_lower_limb -1.036; lit_disc_fraction 0; ray_spacing_deg 0.0002218; march_step_deg 0.08301; samples_outside_grid 0 |
| 20261062 d19_trn_e0_g24.0 e0 | safe landing | 6 site lit | -0.08605 | 0.009064 | -0.07698 |  | sun_elevation 1.705; horizon 1.791; sun_azimuth 15.44; margin_lower_limb -0.353; lit_disc_fraction 0.2989; ray_spacing_deg 0.0002727; march_step_deg 0.008789; samples_outside_grid 0 |
| 20261062 d20_notrn_eNone_gNone e0 | safe landing | 6 site lit | -0.4147 | 0.2283 | -0.1864 |  | sun_elevation 1.706; horizon 2.121; sun_azimuth 15.39; margin_lower_limb -0.6817; lit_disc_fraction 0; ray_spacing_deg 0.000312; march_step_deg 0.228; samples_outside_grid 0 |
| 20261062 d20_trn_e0_g-24.0 e0 | safe landing | 6 site lit | -0.1615 | 0.004463 | -0.157 |  | sun_elevation 1.705; horizon 1.867; sun_azimuth 15.43; margin_lower_limb -0.4284; lit_disc_fraction 0.1416; ray_spacing_deg 0.004461; march_step_deg 0; samples_outside_grid 0 |
| 20261062 d25_trn_e0_g0.0 e0 | safe landing | 6 site lit | -0.07149 | 0.04737 | -0.02412 |  | sun_elevation 1.705; horizon 1.776; sun_azimuth 15.58; margin_lower_limb -0.3385; lit_disc_fraction 0.3316; ray_spacing_deg 1.599e-06; march_step_deg 0.04737; samples_outside_grid 0 |
| 20261062 d25_trn_e0_g168.0 e0 | safe landing | 6 site lit | -2.102 | 0.3836 | -1.718 |  | sun_elevation 1.707; horizon 3.808; sun_azimuth 15.2; margin_lower_limb -2.369; lit_disc_fraction 0; ray_spacing_deg 6.501e-05; march_step_deg 0.3836; samples_outside_grid 0 |
| 20261062 d25_trn_e0_g24.0 e0 | safe landing | 6 site lit | -0.1096 | 0.01825 | -0.09132 |  | sun_elevation 1.705; horizon 1.814; sun_azimuth 15.63; margin_lower_limb -0.3765; lit_disc_fraction 0.2465; ray_spacing_deg 7.887e-05; march_step_deg 0.01817; samples_outside_grid 0 |
| 20261062 d26_trn_e0_g168.0 e0 | safe landing | 6 site lit | -3.168 | 0.1269 | -3.042 |  | sun_elevation 1.707; horizon 4.875; sun_azimuth 14.7; margin_lower_limb -3.435; lit_disc_fraction 0; ray_spacing_deg 0.0001583; march_step_deg 0.1267; samples_outside_grid 0 |
| 20261062 d27_trn_e0_g-24.0 e0 | unsafe landing | 3 glide slope | -3.904 | 2.324 | -1.581 | 21 | lowest_elevation_margin_deg -3.932; d81_over_deg 6.323 |
| 20261062 d27_trn_e1_g-72.0 e1 | unsafe landing | 3 glide slope | -4.25 | 1.522 | -2.728 | 54 | lowest_elevation_margin_deg -4.256; d81_over_deg 6.323 |
| 20261062 d28_notrn_eNone_gNone e0 | safe landing | 3 glide slope | -0.3782 | 0.234 | -0.1442 | 18 | lowest_elevation_margin_deg -0.3782; d81_over_deg 1.736 |
| 20261062 d28_notrn_eNone_gNone e1 | safe landing | 3 glide slope | -0.3782 | 0.234 | -0.1442 | 18 | lowest_elevation_margin_deg -0.3782; d81_over_deg 1.736 |
| 20261062 d28_notrn_eNone_gNone e2 | safe landing | 3 glide slope | -0.3782 | 0.234 | -0.1442 | 18 | lowest_elevation_margin_deg -0.3782; d81_over_deg 1.736 |
| 20261062 d28_notrn_eNone_gNone e3 | safe landing | 3 glide slope | -0.3782 | 0.234 | -0.1442 | 18 | lowest_elevation_margin_deg -0.3782; d81_over_deg 1.736 |
| 20261062 d28_trn_e0_g-24.0 e0 | unsafe landing | 3 glide slope | -1.107 | 0.4495 | -0.6579 | 28 | lowest_elevation_margin_deg -1.107; d81_over_deg 0.8135 |
| 20261062 d28_trn_e0_g168.0 e0 | safe landing | 3 glide slope | -2.155 | 0.7445 | -1.41 | 36 | lowest_elevation_margin_deg -2.16; d81_over_deg 3.412 |
| 20261062 d28_trn_e1_g-72.0 e1 | unsafe landing | 3 glide slope | -0.9362 | 0.3065 | -0.6296 | 37 | lowest_elevation_margin_deg -0.9362; d81_over_deg 0.9354 |
| 20261062 d28_trn_e1_g24.0 e1 | safe landing | 3 glide slope | -1.468 | 1.149 | -0.3192 | 19 | lowest_elevation_margin_deg -1.512; d81_over_deg -0.1769 |
| 20261062 d28_trn_e2_g-72.0 e2 | unsafe landing | 3 glide slope | -1.415 | 0.5203 | -0.8951 | 38 | lowest_elevation_margin_deg -1.415; d81_over_deg 0.2641 |
| 20261062 d28_trn_e3_g-72.0 e3 | unsafe landing | 3 glide slope | -0.3204 | 0.2359 | -0.0845 | 16 | lowest_elevation_margin_deg -0.3204; d81_over_deg 0.7854 |
| 20261062 d29_trn_e0_g0.0 e0 | safe landing | 6 site lit | -0.2547 | 0.008438 | -0.2462 |  | sun_elevation 1.705; horizon 1.96; sun_azimuth 15.42; margin_lower_limb -0.5216; lit_disc_fraction 0.006601; ray_spacing_deg 0.008436; march_step_deg 0; samples_outside_grid 0 |
| 20261062 d33_trn_e0_g-24.0 e0 | safe landing | 6 site lit | -0.778 | 0.03299 | -0.7451 |  | sun_elevation 1.705; horizon 2.483; sun_azimuth 15.32; margin_lower_limb -1.045; lit_disc_fraction 0; ray_spacing_deg 0.0001297; march_step_deg 0.03286; samples_outside_grid 0 |
| 20261062 d40_trn_e0_g24.0 e0 | safe landing | 6 site lit | -0.1161 | 0.01344 | -0.1027 |  | sun_elevation 1.705; horizon 1.822; sun_azimuth 15.43; margin_lower_limb -0.3831; lit_disc_fraction 0.2326; ray_spacing_deg 0.0002733; march_step_deg 0.01317; samples_outside_grid 0 |
| 20261062 d41_trn_e0_g168.0 e0 | safe landing | 6 site lit | -3.312 | 0.2135 | -3.098 |  | sun_elevation 1.707; horizon 5.019; sun_azimuth 14.78; margin_lower_limb -3.579; lit_disc_fraction 0; ray_spacing_deg 9.416e-06; march_step_deg 0.2135; samples_outside_grid 0 |
| 20261062 d41_trn_e0_g24.0 e0 | safe landing | 6 site lit | -0.1368 | 0.1146 | -0.02224 |  | sun_elevation 1.705; horizon 1.841; sun_azimuth 15.42; margin_lower_limb -0.4038; lit_disc_fraction 0.1906; ray_spacing_deg 0.0004795; march_step_deg 0.1141; samples_outside_grid 0 |
| 20261062 d43_notrn_eNone_gNone e0 | safe landing | 6 site lit | -1.006 | 0.003225 | -1.002 |  | sun_elevation 1.71; horizon 2.716; sun_azimuth 17.57; margin_lower_limb -1.273; lit_disc_fraction 0; ray_spacing_deg 4.339e-05; march_step_deg 0.00318; samples_outside_grid 0 |
| 20261062 d45_trn_e0_g168.0 e0 | safe landing | 6 site lit | -2.233 | 0.3204 | -1.913 |  | sun_elevation 1.707; horizon 3.94; sun_azimuth 15.21; margin_lower_limb -2.5; lit_disc_fraction 0; ray_spacing_deg 0.0001417; march_step_deg 0.3203; samples_outside_grid 0 |
| 20261062 d45_trn_e0_g24.0 e0 | unsafe landing | 3 glide slope | -0.6696 | 0.2875 | -0.3821 | 31 | lowest_elevation_margin_deg -0.6696; d81_over_deg 1.858 |
| 20261062 d45_trn_e2_g-72.0 e2 | unsafe landing | 3 glide slope | -0.3007 | 0.2803 | -0.02032 | 5 | lowest_elevation_margin_deg -0.3007; d81_over_deg 0.7239 |
| 20261062 d46_trn_e2_g-24.0 e2 | safe landing | 3 glide slope | -0.4968 | 0.3472 | -0.1496 | 16 | lowest_elevation_margin_deg -0.4968; d81_over_deg -0.1993 |
| 20261062 d46_trn_e2_g-72.0 e2 | unsafe landing | 3 glide slope | -0.369 | 0.3261 | -0.04291 | 9 | lowest_elevation_margin_deg -0.369; d81_over_deg 0.1222 |
| 20261062 d47_trn_e0_g-24.0 e0 | unsafe landing | 3 glide slope | -1.259 | 0.3862 | -0.873 | 38 | lowest_elevation_margin_deg -1.259; d81_over_deg 1.596 |
| 20261062 d47_trn_e0_g168.0 e0 | safe landing | 3 glide slope | -0.05901 | 0.02747 | -0.03154 | 34 | lowest_elevation_margin_deg -0.05901; d81_over_deg 5.851 |
| 20261062 d47_trn_e0_g24.0 e0 | unsafe landing | 3 glide slope | -0.4082 | 0.306 | -0.1022 | 13 | lowest_elevation_margin_deg -0.4082; d81_over_deg 1.389 |
| 20261062 d47_trn_e1_g-72.0 e1 | unsafe landing | 3 glide slope | -1.47 | 0.7108 | -0.7594 | 24 | lowest_elevation_margin_deg -1.47; d81_over_deg 3.198 |
| 20261062 d47_trn_e2_g-72.0 e2 | safe landing | 3 glide slope | -1.311 | 0.4003 | -0.9105 | 40 | lowest_elevation_margin_deg -1.314; d81_over_deg 1.746 |
| 20261062 d47_trn_e3_g-168.0 e3 | safe landing | 3 glide slope | -2.577 | 1.273 | -1.303 | 35 | lowest_elevation_margin_deg -2.608; d81_over_deg 5.475 |
| 20261062 d48_trn_e0_g168.0 e0 | safe landing | 6 site lit | -0.9 | 0.5188 | -0.3811 |  | sun_elevation 1.706; horizon 2.606; sun_azimuth 15.32; margin_lower_limb -1.167; lit_disc_fraction 0; ray_spacing_deg 0.0001603; march_step_deg 0.5187; samples_outside_grid 0 |
| 20261062 d49_trn_e0_g168.0 e0 | safe landing | 6 site lit | -1.542 | 0.05275 | -1.489 |  | sun_elevation 1.705; horizon 3.248; sun_azimuth 15.22; margin_lower_limb -1.809; lit_disc_fraction 0; ray_spacing_deg 9.431e-05; march_step_deg 0.05265; samples_outside_grid 0 |
| 20261062 d49_trn_e0_g24.0 e0 | safe landing | 6 site lit | -0.2561 | 0.02163 | -0.2344 |  | sun_elevation 1.704; horizon 1.96; sun_azimuth 15.36; margin_lower_limb -0.523; lit_disc_fraction 0.005864; ray_spacing_deg 0.0001553; march_step_deg 0.02148; samples_outside_grid 0 |
| 20261062 d50_trn_e0_g0.0 e0 | safe landing | 6 site lit | -0.3197 | 0.01268 | -0.3071 |  | sun_elevation 1.704; horizon 2.024; sun_azimuth 15.38; margin_lower_limb -0.5867; lit_disc_fraction 0; ray_spacing_deg 0.008275; march_step_deg 0.004404; samples_outside_grid 0 |
| 20261062 d50_trn_e0_g72.0 e0 | safe landing | 6 site lit | -0.7003 | 0.01008 | -0.6903 |  | sun_elevation 1.704; horizon 2.405; sun_azimuth 15.32; margin_lower_limb -0.9673; lit_disc_fraction 0; ray_spacing_deg 0.007427; march_step_deg 0.002647; samples_outside_grid 0 |
| 20261062 d51_trn_e0_g0.0 e0 | safe landing | 6 site lit | -0.2676 | 0.04017 | -0.2275 |  | sun_elevation 1.704; horizon 1.972; sun_azimuth 15.37; margin_lower_limb -0.5346; lit_disc_fraction 9.856e-05; ray_spacing_deg 0.0001603; march_step_deg 0.04; samples_outside_grid 0 |
| 20261062 d51_trn_e0_g24.0 e0 | safe landing | 6 site lit | -0.198 | 0.01132 | -0.1867 |  | sun_elevation 1.704; horizon 1.902; sun_azimuth 15.37; margin_lower_limb -0.465; lit_disc_fraction 0.0773; ray_spacing_deg 0.000154; march_step_deg 0.01117; samples_outside_grid 0 |
| 20261062 d52_notrn_eNone_gNone e0 | safe landing | 6 site lit | -0.7578 | 0.5235 | -0.2342 |  | sun_elevation 1.718; horizon 2.476; sun_azimuth 14.58; margin_lower_limb -1.025; lit_disc_fraction 0; ray_spacing_deg 0.0001036; march_step_deg 0.5234; samples_outside_grid 0 |
| 20261062 d52_trn_e0_g-24.0 e0 | safe landing | 6 site lit | -0.08804 | 0.03522 | -0.05282 |  | sun_elevation 1.705; horizon 1.793; sun_azimuth 15.43; margin_lower_limb -0.355; lit_disc_fraction 0.2953; ray_spacing_deg 0.0002269; march_step_deg 0.03499; samples_outside_grid 0 |
| 20261062 d52_trn_e0_g24.0 e0 | safe landing | 6 site lit | -0.02185 | 0.0002404 | -0.02161 |  | sun_elevation 1.705; horizon 1.727; sun_azimuth 15.45; margin_lower_limb -0.2888; lit_disc_fraction 0.4483; ray_spacing_deg 0.0002385; march_step_deg 0; samples_outside_grid 0 |
| 20261062 d53_trn_e2_g-168.0 e2 | abort | 3 glide slope | -0.7171 | 0.4587 | -0.2584 | 2 | lowest_elevation_margin_deg -0.7171; d81_over_deg 4.318 |
| 20261062 d54_trn_e1_g-72.0 e1 | unsafe landing | 3 glide slope | -1.528 | 0.7573 | -0.7707 | 24 | lowest_elevation_margin_deg -1.528; d81_over_deg 3.035 |
| 20261062 d54_trn_e3_g-72.0 e3 | unsafe landing | 3 glide slope | -2.969 | 1.415 | -1.554 | 27 | lowest_elevation_margin_deg -2.976; d81_over_deg 5.914 |
| 20261062 d59_trn_e0_g168.0 e0 | safe landing | 6 site lit | -1.67 | 0.1931 | -1.477 |  | sun_elevation 1.705; horizon 3.376; sun_azimuth 15.16; margin_lower_limb -1.937; lit_disc_fraction 0; ray_spacing_deg 4.466e-05; march_step_deg 0.1931; samples_outside_grid 0 |
| 20261062 d59_trn_e0_g24.0 e0 | unsafe landing | 3 glide slope | -5.016 | 1.906 | -3.11 | 41 | lowest_elevation_margin_deg -5.016; d81_over_deg 5.823 |
| 20261062 d59_trn_e3_g72.0 e3 | safe landing | 6 site lit | -2.942 | 0.2942 | -2.648 |  | sun_elevation 1.641; horizon 4.583; sun_azimuth 297.5; margin_lower_limb -3.208; lit_disc_fraction 0; ray_spacing_deg 0.0007158; march_step_deg 0.2934; samples_outside_grid 0 |
| 20261062 d60_trn_e1_g-72.0 e1 | unsafe landing | 3 glide slope | -0.7899 | 0.4811 | -0.3089 | 18 | lowest_elevation_margin_deg -0.7899; d81_over_deg 1.512 |
| 20261062 d60_trn_e3_g-72.0 e3 | safe landing | 3 glide slope | -0.4229 | 0.3589 | -0.06402 | 11 | lowest_elevation_margin_deg -0.4229; d81_over_deg 0.614 |
| 20261062 d61_trn_e0_g168.0 e0 | safe landing | 6 site lit | -3.119 | 4.622e-05 | -3.119 |  | sun_elevation 1.707; horizon 4.825; sun_azimuth 15.12; margin_lower_limb -3.386; lit_disc_fraction 0; ray_spacing_deg 4.432e-05; march_step_deg 0; samples_outside_grid 0 |
| 20261062 d63_trn_e0_g-24.0 e0 | unsafe landing | 3 glide slope | -3.065 | 1.292 | -1.773 | 47 | lowest_elevation_margin_deg -3.088; d81_over_deg 5.969 |
| 20261062 d63_trn_e0_g72.0 e0 | unsafe landing | 3 glide slope | -1.164 | 0.6475 | -0.5164 | 22 | lowest_elevation_margin_deg -1.164; d81_over_deg 1.52 |
| 20261062 d63_trn_e3_g72.0 e3 | abort | 3 glide slope | -0.08788 | 0.02164 | -0.06623 | 5 | lowest_elevation_margin_deg -0.08788; d81_over_deg 5.969 |
| 20261062 d65_trn_e0_g0.0 e0 | safe landing | 6 site lit | -0.01869 | 0.004295 | -0.0144 |  | sun_elevation 1.705; horizon 1.724; sun_azimuth 15.45; margin_lower_limb -0.2857; lit_disc_fraction 0.4557; ray_spacing_deg 0.004293; march_step_deg 0; samples_outside_grid 0 |
| 20261062 d66_trn_e0_g168.0 e0 | safe landing | 6 site lit | -2.494 | 0.1728 | -2.321 |  | sun_elevation 1.706; horizon 4.2; sun_azimuth 15.11; margin_lower_limb -2.761; lit_disc_fraction 0; ray_spacing_deg 4.871e-06; march_step_deg 0.1728; samples_outside_grid 0 |
| 20261062 d66_trn_e0_g72.0 e0 | safe landing | 6 site lit | -0.01255 | 0.002159 | -0.01039 |  | sun_elevation 1.704; horizon 1.716; sun_azimuth 15.39; margin_lower_limb -0.2795; lit_disc_fraction 0.4702; ray_spacing_deg 0.0001835; march_step_deg 0.001974; samples_outside_grid 0 |
| 20261062 d70_trn_e0_g-24.0 e0 | unsafe landing | 3 glide slope | -0.4796 | 0.2364 | -0.2432 | 25 | lowest_elevation_margin_deg -0.4796; d81_over_deg 0.7998 |
| 20261062 d70_trn_e1_g-72.0 e1 | unsafe landing | 3 glide slope | -0.3934 | 0.2668 | -0.1266 | 18 | lowest_elevation_margin_deg -0.3962; d81_over_deg 0.924 |
| 20261062 d70_trn_e2_g-72.0 e2 | unsafe landing | 3 glide slope | -0.4684 | 0.2062 | -0.2623 | 30 | lowest_elevation_margin_deg -0.4684; d81_over_deg 0.8459 |
| 20261062 d71_trn_e0_g168.0 e0 | safe landing | 6 site lit | -3.137 | 0.1832 | -2.954 |  | sun_elevation 1.707; horizon 4.844; sun_azimuth 15.1; margin_lower_limb -3.404; lit_disc_fraction 0; ray_spacing_deg 3.885e-05; march_step_deg 0.1832; samples_outside_grid 0 |
| 20261062 d72_trn_e0_g-24.0 e0 | safe landing | 6 site lit | -0.3206 | 0.01364 | -0.307 |  | sun_elevation 1.704; horizon 2.024; sun_azimuth 15.33; margin_lower_limb -0.5876; lit_disc_fraction 0; ray_spacing_deg 0.0007784; march_step_deg 0.01286; samples_outside_grid 0 |
| 20261062 d75_trn_e0_g-24.0 e0 | safe landing | 6 site lit | -0.5469 | 0.01728 | -0.5296 |  | sun_elevation 1.706; horizon 2.253; sun_azimuth 15.4; margin_lower_limb -0.8139; lit_disc_fraction 0; ray_spacing_deg 0.0001096; march_step_deg 0.01717; samples_outside_grid 0 |
| 20261062 d75_trn_e0_g168.0 e0 | safe landing | 6 site lit | -2.246 | 0.08113 | -2.165 |  | sun_elevation 1.706; horizon 3.952; sun_azimuth 15.12; margin_lower_limb -2.513; lit_disc_fraction 0; ray_spacing_deg 3.857e-05; march_step_deg 0.08109; samples_outside_grid 0 |
| 20261062 d77_trn_e0_g72.0 e0 | safe landing | 6 site lit | -0.5639 | 0.02904 | -0.5348 |  | sun_elevation 1.704; horizon 2.268; sun_azimuth 15.31; margin_lower_limb -0.8308; lit_disc_fraction 0; ray_spacing_deg 0.0001487; march_step_deg 0.02889; samples_outside_grid 0 |
| 20261062 d80_trn_e0_g0.0 e0 | safe landing | 6 site lit | -0.1622 | 0.02108 | -0.1412 |  | sun_elevation 1.704; horizon 1.866; sun_azimuth 15.38; margin_lower_limb -0.4292; lit_disc_fraction 0.1401; ray_spacing_deg 0.000158; march_step_deg 0.02092; samples_outside_grid 0 |
| 20261062 d80_trn_e0_g24.0 e0 | safe landing | 6 site lit | -0.09049 | 0.01882 | -0.07167 |  | sun_elevation 1.704; horizon 1.795; sun_azimuth 15.39; margin_lower_limb -0.3574; lit_disc_fraction 0.2896; ray_spacing_deg 0.0001602; march_step_deg 0.01866; samples_outside_grid 0 |
| 20261062 d81_trn_e0_g168.0 e0 | safe landing | 6 site lit | -3.026 | 0.1396 | -2.887 |  | sun_elevation 1.707; horizon 4.733; sun_azimuth 14.93; margin_lower_limb -3.293; lit_disc_fraction 0; ray_spacing_deg 2.715e-05; march_step_deg 0.1396; samples_outside_grid 0 |
| 20261062 d84_trn_e0_g0.0 e0 | safe landing | 6 site lit | -0.297 | 0.0184 | -0.2786 |  | sun_elevation 1.704; horizon 2.001; sun_azimuth 15.35; margin_lower_limb -0.5639; lit_disc_fraction 0; ray_spacing_deg 0.0001535; march_step_deg 0.01824; samples_outside_grid 0 |
| 20261062 d85_trn_e0_g-24.0 e0 | unsafe landing | 3 glide slope | -0.3966 | 0.1876 | -0.2091 | 30 | lowest_elevation_margin_deg -0.3966; d81_over_deg 0.3885 |
| 20261062 d85_trn_e1_g-72.0 e1 | unsafe landing | 3 glide slope | -0.7285 | 0.2377 | -0.4909 | 37 | lowest_elevation_margin_deg -0.7285; d81_over_deg 0.1793 |
| 20261062 d85_trn_e1_g72.0 e1 | unsafe landing | 3 glide slope | -0.386 | 0.2041 | -0.1819 | 24 | lowest_elevation_margin_deg -0.386; d81_over_deg 0.0794 |
| 20261062 d85_trn_e2_g-72.0 e2 | safe landing | 3 glide slope | -0.4877 | 0.3526 | -0.1351 | 15 | lowest_elevation_margin_deg -0.4881; d81_over_deg 0.6668 |
| 20261062 d85_trn_e3_g-72.0 e3 | unsafe landing | 3 glide slope | -0.6135 | 0.2991 | -0.3144 | 25 | lowest_elevation_margin_deg -0.6137; d81_over_deg 0.487 |
| 20261062 d86_trn_e0_g168.0 e0 | safe landing | 6 site lit | -2.114 | 0.2442 | -1.869 |  | sun_elevation 1.707; horizon 3.82; sun_azimuth 15.23; margin_lower_limb -2.381; lit_disc_fraction 0; ray_spacing_deg 0.0001497; march_step_deg 0.2441; samples_outside_grid 0 |
| 20261062 d86_trn_e0_g72.0 e0 | safe landing | 6 site lit | -0.348 | 0.04746 | -0.3005 |  | sun_elevation 1.705; horizon 2.053; sun_azimuth 15.39; margin_lower_limb -0.6149; lit_disc_fraction 0; ray_spacing_deg 0.0003522; march_step_deg 0.04711; samples_outside_grid 0 |
| 20261062 d87_trn_e0_g168.0 e0 | safe landing | 6 site lit | -2.787 | 0.3269 | -2.46 |  | sun_elevation 1.706; horizon 4.493; sun_azimuth 15.03; margin_lower_limb -3.054; lit_disc_fraction 0; ray_spacing_deg 2.927e-06; march_step_deg 0.3269; samples_outside_grid 0 |
| 20261062 d88_trn_e0_g168.0 e0 | safe landing | 6 site lit | -2.186 | 0.02334 | -2.163 |  | sun_elevation 1.687; horizon 3.873; sun_azimuth 23.54; margin_lower_limb -2.453; lit_disc_fraction 0; ray_spacing_deg 0.00257; march_step_deg 0.02077; samples_outside_grid 0 |
| 20261062 d88_trn_e2_g-168.0 e2 | safe landing | 3 glide slope | -0.05088 | 0.01686 | -0.03403 | 36 | lowest_elevation_margin_deg -0.05088; d81_over_deg 2.409 |
| 20261062 d91_trn_e0_g24.0 e0 | safe landing | 6 site lit | -0.02075 | 0.0176 | -0.003148 |  | sun_elevation 1.704; horizon 1.724; sun_azimuth 15.37; margin_lower_limb -0.2877; lit_disc_fraction 0.4508; ray_spacing_deg 0.0001767; march_step_deg 0.01742; samples_outside_grid 0 |
| 20261062 d92_trn_e0_g168.0 e0 | safe landing | 6 site lit | -3.303 | 0.0006199 | -3.303 |  | sun_elevation 1.707; horizon 5.01; sun_azimuth 14.89; margin_lower_limb -3.57; lit_disc_fraction 0; ray_spacing_deg 3.478e-05; march_step_deg 0.0005833; samples_outside_grid 0 |
| 20261062 d92_trn_e2_g-168.0 e2 | unsafe landing | 3 glide slope | -1.438 | 0.4147 | -1.024 | 37 | lowest_elevation_margin_deg -1.438; d81_over_deg 5.885 |
| 20261062 d93_trn_e0_g168.0 e0 | safe landing | 6 site lit | -1.698 | 0.2311 | -1.466 |  | sun_elevation 1.706; horizon 3.404; sun_azimuth 15.24; margin_lower_limb -1.964; lit_disc_fraction 0; ray_spacing_deg 0.0001933; march_step_deg 0.2309; samples_outside_grid 0 |
| 20261062 d93_trn_e3_g-72.0 e3 | unsafe landing | 3 glide slope | -1.203 | 0.5242 | -0.6787 | 28 | lowest_elevation_margin_deg -1.204; d81_over_deg 1.281 |
| 20261062 d97_trn_e0_g0.0 e0 | safe landing | 6 site lit | -0.1637 | 0.03438 | -0.1293 |  | sun_elevation 1.704; horizon 1.868; sun_azimuth 15.39; margin_lower_limb -0.4306; lit_disc_fraction 0.1375; ray_spacing_deg 0.0001611; march_step_deg 0.03422; samples_outside_grid 0 |
| 20261062 d98_trn_e0_g168.0 e0 | safe landing | 6 site lit | -2.854 | 0.1037 | -2.751 |  | sun_elevation 1.706; horizon 4.561; sun_azimuth 15.02; margin_lower_limb -3.121; lit_disc_fraction 0; ray_spacing_deg 1.685e-06; march_step_deg 0.1037; samples_outside_grid 0 |
| 20261062 d98_trn_e1_g24.0 e1 | unsafe landing | 3 glide slope | -1.588 | 1.257 | -0.331 | 17 | lowest_elevation_margin_deg -1.588; d81_over_deg 1.284 |
| 20261062 d98_trn_e2_g-72.0 e2 | unsafe landing | 3 glide slope | -1.51 | 0.6135 | -0.896 | 30 | lowest_elevation_margin_deg -1.51; d81_over_deg 2.131 |

## Where each violation falls (examples/phase7_replay_locate.py)

- 20261062 d01_trn_e0_g-24.0 e0 (safe landing), 6 site lit: margin -0.4601 against tolerance 0.03921 deg; touchdown at map (1702.9, 6516.8) m, epoch 0, t = 137.92 s; sun elevation 1.7043 deg against horizon 2.1643 deg
- 20261062 d03_trn_e0_g168.0 e0 (safe landing), 6 site lit: margin -2.422 against tolerance 0.1513 deg; touchdown at map (1741.0, 6561.5) m, epoch 0, t = 142.96 s; sun elevation 1.7062 deg against horizon 4.1282 deg
- 20261062 d03_trn_e1_g-24.0 e1 (unsafe landing), 3 glide slope: margin -1.639 against tolerance 0.5639 deg; 33 of 2000 guided ticks beyond tolerance, t = 82.35 to 83.95 s (guidance steps [41]; guided flight ends at 100.00 s); worst at t = 83.10 s, the estimate 4.96 m above and 33.73 m from the site in force
- 20261062 d03_trn_e3_g-72.0 e3 (safe landing), 3 glide slope: margin -0.8014 against tolerance 0.4006 deg; 25 of 2000 guided ticks beyond tolerance, t = 82.60 to 83.80 s (guidance steps [41]; guided flight ends at 100.00 s); worst at t = 83.20 s, the estimate 7.58 m above and 46.79 m from the site in force
- 20261062 d05_trn_e0_g0.0 e0 (safe landing), 6 site lit: margin -0.2311 against tolerance 0.04446 deg; touchdown at map (1698.5, 6521.5) m, epoch 0, t = 139.21 s; sun elevation 1.7043 deg against horizon 1.9354 deg
- 20261062 d06_trn_e0_g-24.0 e0 (safe landing), 6 site lit: margin -0.2594 against tolerance 0.08601 deg; touchdown at map (1699.1, 6532.4) m, epoch 0, t = 141.69 s; sun elevation 1.7046 deg against horizon 1.9640 deg
- 20261062 d06_trn_e0_g168.0 e0 (safe landing), 6 site lit: margin -1.344 against tolerance 0.08827 deg; touchdown at map (1721.1, 6538.1) m, epoch 0, t = 142.55 s; sun elevation 1.7052 deg against horizon 3.0488 deg
- 20261062 d07_trn_e0_g168.0 e0 (safe landing), 6 site lit: margin -2.493 against tolerance 0.09695 deg; touchdown at map (1736.8, 6576.8) m, epoch 0, t = 137.53 s; sun elevation 1.7065 deg against horizon 4.1993 deg
- 20261062 d08_trn_e0_g0.0 e0 (safe landing), 6 site lit: margin -0.1455 against tolerance 0.06733 deg; touchdown at map (1699.1, 6543.7) m, epoch 0, t = 140.44 s; sun elevation 1.7050 deg against horizon 1.8505 deg
- 20261062 d10_trn_e0_g168.0 e0 (safe landing), 6 site lit: margin -1.081 against tolerance 0.1771 deg; touchdown at map (1720.3, 6550.9) m, epoch 0, t = 138.87 s; sun elevation 1.7055 deg against horizon 2.7866 deg
- 20261062 d12_trn_e0_g-24.0 e0 (safe landing), 6 site lit: margin -0.9821 against tolerance 0.01183 deg; touchdown at map (1711.6, 6523.4) m, epoch 0, t = 134.03 s; sun elevation 1.7046 deg against horizon 2.6867 deg
- 20261062 d12_trn_e0_g0.0 e0 (safe landing), 6 site lit: margin -0.2238 against tolerance 0.01338 deg; touchdown at map (1680.3, 6551.4) m, epoch 0, t = 145.16 s; sun elevation 1.7049 deg against horizon 1.9287 deg
- 20261062 d13_trn_e0_g-24.0 e0 (unsafe landing), 3 glide slope: margin -0.4248 against tolerance 0.2677 deg; 19 of 2240 guided ticks beyond tolerance, t = 83.75 to 84.65 s (guidance steps [41, 42]; guided flight ends at 112.00 s); worst at t = 84.15 s, the estimate 11.80 m above and 69.96 m from the site in force
- 20261062 d13_trn_e0_g0.0 e0 (safe landing), 6 site lit: margin -0.2803 against tolerance 0.01898 deg; touchdown at map (1699.6, 6514.0) m, epoch 0, t = 144.83 s; sun elevation 1.7041 deg against horizon 1.9845 deg
- 20261062 d13_trn_e0_g72.0 e0 (unsafe landing), 3 glide slope: margin -0.3132 against tolerance 0.2635 deg; 10 of 2080 guided ticks beyond tolerance, t = 85.25 to 85.70 s (guidance steps [42]; guided flight ends at 104.00 s); worst at t = 85.60 s, the estimate 12.24 m above and 71.68 m from the site in force
- 20261062 d13_trn_e1_g-72.0 e1 (unsafe landing), 3 glide slope: margin -0.4977 against tolerance 0.238 deg; 27 of 2240 guided ticks beyond tolerance, t = 83.70 to 85.00 s (guidance steps [41, 42]; guided flight ends at 112.00 s); worst at t = 84.15 s, the estimate 13.07 m above and 78.10 m from the site in force
- 20261062 d13_trn_e1_g72.0 e1 (unsafe landing), 3 glide slope: margin -0.4324 against tolerance 0.2574 deg; 20 of 2120 guided ticks beyond tolerance, t = 84.90 to 85.85 s (guidance steps [42]; guided flight ends at 106.00 s); worst at t = 85.30 s, the estimate 12.14 m above and 72.02 m from the site in force
- 20261062 d13_trn_e3_g-72.0 e3 (safe landing), 3 glide slope: margin -0.4133 against tolerance 0.3142 deg; 14 of 2080 guided ticks beyond tolerance, t = 85.15 to 85.80 s (guidance steps [42]; guided flight ends at 104.00 s); worst at t = 85.50 s, the estimate 10.07 m above and 59.65 m from the site in force
- 20261062 d14_trn_e1_g-72.0 e1 (unsafe landing), 3 glide slope: margin -1.185 against tolerance 0.3479 deg; 39 of 2120 guided ticks beyond tolerance, t = 84.10 to 86.00 s (guidance steps [42]; guided flight ends at 106.00 s); worst at t = 84.85 s, the estimate 8.30 m above and 53.52 m from the site in force
- 20261062 d14_trn_e3_g-72.0 e3 (safe landing), 3 glide slope: margin -0.4796 against tolerance 0.3108 deg; 17 of 2080 guided ticks beyond tolerance, t = 85.10 to 85.90 s (guidance steps [42]; guided flight ends at 104.00 s); worst at t = 85.45 s, the estimate 10.04 m above and 59.89 m from the site in force
- 20261062 d15_trn_e1_g72.0 e1 (unsafe landing), 3 glide slope: margin -0.6588 against tolerance 0.6081 deg; 5 of 2000 guided ticks beyond tolerance, t = 85.45 to 85.65 s (guidance steps [42]; guided flight ends at 100.00 s); worst at t = 85.60 s, the estimate 5.11 m above and 31.08 m from the site in force
- 20261062 d18_trn_e0_g-24.0 e0 (safe landing), 6 site lit: margin -0.2171 against tolerance 0.01219 deg; touchdown at map (1680.3, 6550.2) m, epoch 0, t = 143.64 s; sun elevation 1.7048 deg against horizon 1.9220 deg
- 20261062 d19_trn_e0_g-24.0 e0 (safe landing), 6 site lit: margin -0.3051 against tolerance 0.06687 deg; touchdown at map (1706.0, 6556.2) m, epoch 0, t = 133.72 s; sun elevation 1.7054 deg against horizon 2.0106 deg
- 20261062 d19_trn_e0_g168.0 e0 (safe landing), 6 site lit: margin -0.7692 against tolerance 0.08323 deg; touchdown at map (1713.4, 6572.7) m, epoch 0, t = 138.91 s; sun elevation 1.7060 deg against horizon 2.4752 deg
- 20261062 d19_trn_e0_g24.0 e0 (safe landing), 6 site lit: margin -0.08605 against tolerance 0.009064 deg; touchdown at map (1701.0, 6557.2) m, epoch 0, t = 142.49 s; sun elevation 1.7054 deg against horizon 1.7914 deg
- 20261062 d20_notrn_eNone_gNone e0 (safe landing), 6 site lit: margin -0.4147 against tolerance 0.2283 deg; touchdown at map (1709.8, 6570.1) m, epoch 0, t = 143.52 s; sun elevation 1.7059 deg against horizon 2.1206 deg
- 20261062 d20_trn_e0_g-24.0 e0 (safe landing), 6 site lit: margin -0.1615 against tolerance 0.004463 deg; touchdown at map (1702.3, 6558.2) m, epoch 0, t = 143.03 s; sun elevation 1.7054 deg against horizon 1.8669 deg
- 20261062 d25_trn_e0_g0.0 e0 (safe landing), 6 site lit: margin -0.07149 against tolerance 0.04737 deg; touchdown at map (1681.2, 6549.3) m, epoch 0, t = 145.44 s; sun elevation 1.7048 deg against horizon 1.7763 deg
- 20261062 d25_trn_e0_g168.0 e0 (safe landing), 6 site lit: margin -2.102 against tolerance 0.3836 deg; touchdown at map (1735.6, 6580.9) m, epoch 0, t = 164.02 s; sun elevation 1.7066 deg against horizon 3.8083 deg
- 20261062 d25_trn_e0_g24.0 e0 (safe landing), 6 site lit: margin -0.1096 against tolerance 0.01825 deg; touchdown at map (1675.6, 6549.6) m, epoch 0, t = 143.28 s; sun elevation 1.7047 deg against horizon 1.8143 deg
- 20261062 d26_trn_e0_g168.0 e0 (safe landing), 6 site lit: margin -3.168 against tolerance 0.1269 deg; touchdown at map (1789.5, 6551.8) m, epoch 0, t = 139.10 s; sun elevation 1.7067 deg against horizon 4.8751 deg
- 20261062 d27_trn_e0_g-24.0 e0 (unsafe landing), 3 glide slope: margin -3.904 against tolerance 2.324 deg; 21 of 1960 guided ticks beyond tolerance, t = 82.95 to 83.95 s (guidance steps [41]; guided flight ends at 98.00 s); worst at t = 83.35 s, the estimate 0.88 m above and 8.25 m from the site in force
- 20261062 d27_trn_e1_g-72.0 e1 (unsafe landing), 3 glide slope: margin -4.25 against tolerance 1.522 deg; 54 of 1920 guided ticks beyond tolerance, t = 81.40 to 84.05 s (guidance steps [40, 41, 42]; guided flight ends at 96.00 s); worst at t = 82.95 s, the estimate 1.27 m above and 12.65 m from the site in force
- 20261062 d28_notrn_eNone_gNone e0 (safe landing), 3 glide slope: margin -0.3782 against tolerance 0.234 deg; 18 of 2080 guided ticks beyond tolerance, t = 84.80 to 85.65 s (guidance steps [42]; guided flight ends at 104.00 s); worst at t = 85.30 s, the estimate 13.53 m above and 79.80 m from the site in force
- 20261062 d28_notrn_eNone_gNone e1 (safe landing), 3 glide slope: margin -0.3782 against tolerance 0.234 deg; 18 of 2080 guided ticks beyond tolerance, t = 84.80 to 85.65 s (guidance steps [42]; guided flight ends at 104.00 s); worst at t = 85.30 s, the estimate 13.53 m above and 79.80 m from the site in force
- 20261062 d28_notrn_eNone_gNone e2 (safe landing), 3 glide slope: margin -0.3782 against tolerance 0.234 deg; 18 of 2080 guided ticks beyond tolerance, t = 84.80 to 85.65 s (guidance steps [42]; guided flight ends at 104.00 s); worst at t = 85.30 s, the estimate 13.53 m above and 79.80 m from the site in force
- 20261062 d28_notrn_eNone_gNone e3 (safe landing), 3 glide slope: margin -0.3782 against tolerance 0.234 deg; 18 of 2080 guided ticks beyond tolerance, t = 84.80 to 85.65 s (guidance steps [42]; guided flight ends at 104.00 s); worst at t = 85.30 s, the estimate 13.53 m above and 79.80 m from the site in force
- 20261062 d28_trn_e0_g-24.0 e0 (unsafe landing), 3 glide slope: margin -1.107 against tolerance 0.4495 deg; 28 of 2040 guided ticks beyond tolerance, t = 82.50 to 83.85 s (guidance steps [41]; guided flight ends at 102.00 s); worst at t = 83.10 s, the estimate 6.52 m above and 41.69 m from the site in force
- 20261062 d28_trn_e0_g168.0 e0 (safe landing), 3 glide slope: margin -2.155 against tolerance 0.7445 deg; 36 of 2000 guided ticks beyond tolerance, t = 82.10 to 83.85 s (guidance steps [41]; guided flight ends at 100.00 s); worst at t = 82.90 s, the estimate 3.51 m above and 25.46 m from the site in force
- 20261062 d28_trn_e1_g-72.0 e1 (unsafe landing), 3 glide slope: margin -0.9362 against tolerance 0.3065 deg; 37 of 2080 guided ticks beyond tolerance, t = 82.05 to 83.85 s (guidance steps [41]; guided flight ends at 104.00 s); worst at t = 82.85 s, the estimate 9.67 m above and 60.64 m from the site in force
- 20261062 d28_trn_e1_g24.0 e1 (safe landing), 3 glide slope: margin -1.468 against tolerance 1.149 deg; 19 of 1920 guided ticks beyond tolerance, t = 82.35 to 83.25 s (guidance steps [41]; guided flight ends at 96.00 s); worst at t = 82.85 s, the estimate 2.45 m above and 16.33 m from the site in force
- 20261062 d28_trn_e2_g-72.0 e2 (unsafe landing), 3 glide slope: margin -1.415 against tolerance 0.5203 deg; 38 of 2000 guided ticks beyond tolerance, t = 82.00 to 83.85 s (guidance steps [40, 41]; guided flight ends at 100.00 s); worst at t = 82.85 s, the estimate 5.44 m above and 36.00 m from the site in force
- 20261062 d28_trn_e3_g-72.0 e3 (unsafe landing), 3 glide slope: margin -0.3204 against tolerance 0.2359 deg; 16 of 2200 guided ticks beyond tolerance, t = 82.05 to 82.80 s (guidance steps [41]; guided flight ends at 110.00 s); worst at t = 82.35 s, the estimate 13.45 m above and 78.88 m from the site in force
- 20261062 d29_trn_e0_g0.0 e0 (safe landing), 6 site lit: margin -0.2547 against tolerance 0.008438 deg; touchdown at map (1699.1, 6540.8) m, epoch 0, t = 141.22 s; sun elevation 1.7049 deg against horizon 1.9595 deg
- 20261062 d33_trn_e0_g-24.0 e0 (safe landing), 6 site lit: margin -0.778 against tolerance 0.03299 deg; touchdown at map (1708.3, 6527.8) m, epoch 0, t = 132.80 s; sun elevation 1.7047 deg against horizon 2.4827 deg
- 20261062 d40_trn_e0_g24.0 e0 (safe landing), 6 site lit: margin -0.1161 against tolerance 0.01344 deg; touchdown at map (1701.7, 6557.1) m, epoch 0, t = 148.24 s; sun elevation 1.7054 deg against horizon 1.8215 deg
- 20261062 d41_trn_e0_g168.0 e0 (safe landing), 6 site lit: margin -3.312 against tolerance 0.2135 deg; touchdown at map (1783.6, 6563.0) m, epoch 0, t = 139.78 s; sun elevation 1.7069 deg against horizon 5.0185 deg
- 20261062 d41_trn_e0_g24.0 e0 (safe landing), 6 site lit: margin -0.1368 against tolerance 0.1146 deg; touchdown at map (1697.5, 6534.3) m, epoch 0, t = 141.22 s; sun elevation 1.7047 deg against horizon 1.8415 deg
- 20261062 d43_notrn_eNone_gNone e0 (safe landing), 6 site lit: margin -1.006 against tolerance 0.003225 deg; touchdown at map (1505.5, 6841.0) m, epoch 0, t = 136.70 s; sun elevation 1.7103 deg against horizon 2.7159 deg
- 20261062 d45_trn_e0_g168.0 e0 (safe landing), 6 site lit: margin -2.233 against tolerance 0.3204 deg; touchdown at map (1736.3, 6584.0) m, epoch 0, t = 151.06 s; sun elevation 1.7067 deg against horizon 3.9401 deg
- 20261062 d45_trn_e0_g24.0 e0 (unsafe landing), 3 glide slope: margin -0.6696 against tolerance 0.2875 deg; 31 of 2160 guided ticks beyond tolerance, t = 83.90 to 85.40 s (guidance steps [41, 42]; guided flight ends at 108.00 s); worst at t = 84.55 s, the estimate 10.70 m above and 65.10 m from the site in force
- 20261062 d45_trn_e2_g-72.0 e2 (unsafe landing), 3 glide slope: margin -0.3007 against tolerance 0.2803 deg; 5 of 2080 guided ticks beyond tolerance, t = 85.45 to 85.65 s (guidance steps [42]; guided flight ends at 104.00 s); worst at t = 85.50 s, the estimate 11.28 m above and 66.02 m from the site in force
- 20261062 d46_trn_e2_g-24.0 e2 (safe landing), 3 glide slope: margin -0.4968 against tolerance 0.3472 deg; 16 of 2000 guided ticks beyond tolerance, t = 84.30 to 85.05 s (guidance steps [42]; guided flight ends at 100.00 s); worst at t = 84.70 s, the estimate 8.97 m above and 53.59 m from the site in force
- 20261062 d46_trn_e2_g-72.0 e2 (unsafe landing), 3 glide slope: margin -0.369 against tolerance 0.3261 deg; 9 of 2000 guided ticks beyond tolerance, t = 84.30 to 84.70 s (guidance steps [42]; guided flight ends at 100.00 s); worst at t = 84.45 s, the estimate 9.66 m above and 56.90 m from the site in force
- 20261062 d47_trn_e0_g-24.0 e0 (unsafe landing), 3 glide slope: margin -1.259 against tolerance 0.3862 deg; 38 of 2080 guided ticks beyond tolerance, t = 84.15 to 86.00 s (guidance steps [42]; guided flight ends at 104.00 s); worst at t = 85.00 s, the estimate 7.54 m above and 49.05 m from the site in force
- 20261062 d47_trn_e0_g168.0 e0 (safe landing), 3 glide slope: margin -0.05901 against tolerance 0.02747 deg; 34 of 2520 guided ticks beyond tolerance, t = 80.25 to 81.90 s (guidance steps [40]; guided flight ends at 126.00 s); worst at t = 81.20 s, the estimate 120.36 m above and 686.75 m from the site in force
- 20261062 d47_trn_e0_g24.0 e0 (unsafe landing), 3 glide slope: margin -0.4082 against tolerance 0.306 deg; 13 of 2080 guided ticks beyond tolerance, t = 85.10 to 85.70 s (guidance steps [42]; guided flight ends at 104.00 s); worst at t = 85.35 s, the estimate 10.23 m above and 60.52 m from the site in force
- 20261062 d47_trn_e1_g-72.0 e1 (unsafe landing), 3 glide slope: margin -1.47 against tolerance 0.7108 deg; 24 of 2040 guided ticks beyond tolerance, t = 84.80 to 85.95 s (guidance steps [42]; guided flight ends at 102.00 s); worst at t = 85.30 s, the estimate 3.98 m above and 26.51 m from the site in force
- 20261062 d47_trn_e2_g-72.0 e2 (safe landing), 3 glide slope: margin -1.311 against tolerance 0.4003 deg; 40 of 2080 guided ticks beyond tolerance, t = 84.05 to 86.00 s (guidance steps [42]; guided flight ends at 104.00 s); worst at t = 85.00 s, the estimate 7.27 m above and 47.57 m from the site in force
- 20261062 d47_trn_e3_g-168.0 e3 (safe landing), 3 glide slope: margin -2.577 against tolerance 1.273 deg; 35 of 1960 guided ticks beyond tolerance, t = 83.80 to 85.50 s (guidance steps [41, 42]; guided flight ends at 98.00 s); worst at t = 84.65 s, the estimate 1.93 m above and 14.81 m from the site in force
- 20261062 d48_trn_e0_g168.0 e0 (safe landing), 6 site lit: margin -0.9 against tolerance 0.5188 deg; touchdown at map (1721.4, 6582.4) m, epoch 0, t = 142.98 s; sun elevation 1.7064 deg against horizon 2.6064 deg
- 20261062 d49_trn_e0_g168.0 e0 (safe landing), 6 site lit: margin -1.542 against tolerance 0.05275 deg; touchdown at map (1724.4, 6543.7) m, epoch 0, t = 135.95 s; sun elevation 1.7054 deg against horizon 3.2476 deg
- 20261062 d49_trn_e0_g24.0 e0 (safe landing), 6 site lit: margin -0.2561 against tolerance 0.02163 deg; touchdown at map (1699.1, 6514.7) m, epoch 0, t = 136.45 s; sun elevation 1.7041 deg against horizon 1.9602 deg
- 20261062 d50_trn_e0_g0.0 e0 (safe landing), 6 site lit: margin -0.3197 against tolerance 0.01268 deg; touchdown at map (1699.3, 6523.4) m, epoch 0, t = 138.06 s; sun elevation 1.7044 deg against horizon 2.0241 deg
- 20261062 d50_trn_e0_g72.0 e0 (safe landing), 6 site lit: margin -0.7003 against tolerance 0.01008 deg; touchdown at map (1706.3, 6523.5) m, epoch 0, t = 135.83 s; sun elevation 1.7045 deg against horizon 2.4048 deg
- 20261062 d51_trn_e0_g0.0 e0 (safe landing), 6 site lit: margin -0.2676 against tolerance 0.04017 deg; touchdown at map (1699.2, 6520.7) m, epoch 0, t = 142.15 s; sun elevation 1.7043 deg against horizon 1.9719 deg
- 20261062 d51_trn_e0_g24.0 e0 (safe landing), 6 site lit: margin -0.198 against tolerance 0.01132 deg; touchdown at map (1698.1, 6512.3) m, epoch 0, t = 140.31 s; sun elevation 1.7040 deg against horizon 1.9021 deg
- 20261062 d52_notrn_eNone_gNone e0 (safe landing), 6 site lit: margin -0.7578 against tolerance 0.5235 deg; touchdown at map (1898.5, 6895.3) m, epoch 0, t = 146.12 s; sun elevation 1.7183 deg against horizon 2.4761 deg
- 20261062 d52_trn_e0_g-24.0 e0 (safe landing), 6 site lit: margin -0.08804 against tolerance 0.03522 deg; touchdown at map (1702.0, 6559.2) m, epoch 0, t = 144.33 s; sun elevation 1.7054 deg against horizon 1.7935 deg
- 20261062 d52_trn_e0_g24.0 e0 (safe landing), 6 site lit: margin -0.02185 against tolerance 0.0002404 deg; touchdown at map (1699.8, 6558.3) m, epoch 0, t = 146.39 s; sun elevation 1.7054 deg against horizon 1.7272 deg
- 20261062 d53_trn_e2_g-168.0 e2 (abort), 3 glide slope: margin -0.7171 against tolerance 0.4587 deg; 2 of 1720 guided ticks beyond tolerance, t = 85.95 to 86.00 s (guidance steps [42]; guided flight ends at 86.00 s); worst at t = 86.00 s, the estimate 7.67 m above and 46.95 m from the site in force
- 20261062 d54_trn_e1_g-72.0 e1 (unsafe landing), 3 glide slope: margin -1.528 against tolerance 0.7573 deg; 24 of 1920 guided ticks beyond tolerance, t = 80.75 to 81.90 s (guidance steps [40]; guided flight ends at 96.00 s); worst at t = 81.30 s, the estimate 3.71 m above and 24.91 m from the site in force
- 20261062 d54_trn_e3_g-72.0 e3 (unsafe landing), 3 glide slope: margin -2.969 against tolerance 1.415 deg; 27 of 1920 guided ticks beyond tolerance, t = 80.60 to 81.90 s (guidance steps [40]; guided flight ends at 96.00 s); worst at t = 81.15 s, the estimate 1.66 m above and 13.43 m from the site in force
- 20261062 d59_trn_e0_g168.0 e0 (safe landing), 6 site lit: margin -1.67 against tolerance 0.1931 deg; touchdown at map (1727.9, 6533.1) m, epoch 0, t = 143.71 s; sun elevation 1.7051 deg against horizon 3.3755 deg
- 20261062 d59_trn_e0_g24.0 e0 (unsafe landing), 3 glide slope: margin -5.016 against tolerance 1.906 deg; 41 of 2000 guided ticks beyond tolerance, t = 85.95 to 87.95 s (guidance steps [42, 43]; guided flight ends at 100.00 s); worst at t = 87.30 s, the estimate 0.88 m above and 10.07 m from the site in force
- 20261062 d59_trn_e3_g72.0 e3 (safe landing), 6 site lit: margin -2.942 against tolerance 0.2942 deg; touchdown at map (244.4, 5564.4) m, epoch 3, t = 197.10 s; sun elevation 1.6406 deg against horizon 4.5830 deg
- 20261062 d60_trn_e1_g-72.0 e1 (unsafe landing), 3 glide slope: margin -0.7899 against tolerance 0.4811 deg; 18 of 2040 guided ticks beyond tolerance, t = 85.05 to 85.90 s (guidance steps [42]; guided flight ends at 102.00 s); worst at t = 85.40 s, the estimate 6.26 m above and 38.60 m from the site in force
- 20261062 d60_trn_e3_g-72.0 e3 (safe landing), 3 glide slope: margin -0.4229 against tolerance 0.3589 deg; 11 of 2040 guided ticks beyond tolerance, t = 85.25 to 85.75 s (guidance steps [42]; guided flight ends at 102.00 s); worst at t = 85.40 s, the estimate 8.74 m above and 51.80 m from the site in force
- 20261062 d61_trn_e0_g168.0 e0 (safe landing), 6 site lit: margin -3.119 against tolerance 4.622e-05 deg; touchdown at map (1744.8, 6575.7) m, epoch 0, t = 137.14 s; sun elevation 1.7066 deg against horizon 4.8254 deg
- 20261062 d63_trn_e0_g-24.0 e0 (unsafe landing), 3 glide slope: margin -3.065 against tolerance 1.292 deg; 47 of 2000 guided ticks beyond tolerance, t = 85.40 to 87.70 s (guidance steps [42, 43]; guided flight ends at 100.00 s); worst at t = 86.70 s, the estimate 1.79 m above and 14.72 m from the site in force
- 20261062 d63_trn_e0_g72.0 e0 (unsafe landing), 3 glide slope: margin -1.164 against tolerance 0.6475 deg; 22 of 2000 guided ticks beyond tolerance, t = 84.65 to 85.70 s (guidance steps [42]; guided flight ends at 100.00 s); worst at t = 85.20 s, the estimate 4.48 m above and 28.85 m from the site in force
- 20261062 d63_trn_e3_g72.0 e3 (abort), 3 glide slope: margin -0.08788 against tolerance 0.02164 deg; 5 of 1400 guided ticks beyond tolerance, t = 69.80 to 70.00 s (guidance steps [34]; guided flight ends at 70.00 s); worst at t = 70.00 s, the estimate 173.32 m above and 991.84 m from the site in force
- 20261062 d65_trn_e0_g0.0 e0 (safe landing), 6 site lit: margin -0.01869 against tolerance 0.004295 deg; touchdown at map (1699.7, 6558.2) m, epoch 0, t = 140.41 s; sun elevation 1.7054 deg against horizon 1.7241 deg
- 20261062 d66_trn_e0_g168.0 e0 (safe landing), 6 site lit: margin -2.494 against tolerance 0.1728 deg; touchdown at map (1742.9, 6562.3) m, epoch 0, t = 134.69 s; sun elevation 1.7062 deg against horizon 4.1999 deg
- 20261062 d66_trn_e0_g72.0 e0 (safe landing), 6 site lit: margin -0.01255 against tolerance 0.002159 deg; touchdown at map (1694.4, 6507.1) m, epoch 0, t = 139.08 s; sun elevation 1.7038 deg against horizon 1.7164 deg
- 20261062 d70_trn_e0_g-24.0 e0 (unsafe landing), 3 glide slope: margin -0.4796 against tolerance 0.2364 deg; 25 of 2080 guided ticks beyond tolerance, t = 82.75 to 83.95 s (guidance steps [41]; guided flight ends at 104.00 s); worst at t = 83.25 s, the estimate 13.15 m above and 78.40 m from the site in force
- 20261062 d70_trn_e1_g-72.0 e1 (unsafe landing), 3 glide slope: margin -0.3934 against tolerance 0.2668 deg; 18 of 2040 guided ticks beyond tolerance, t = 83.00 to 83.85 s (guidance steps [41]; guided flight ends at 102.00 s); worst at t = 83.50 s, the estimate 11.93 m above and 70.51 m from the site in force
- 20261062 d70_trn_e2_g-72.0 e2 (unsafe landing), 3 glide slope: margin -0.4684 against tolerance 0.2062 deg; 30 of 2200 guided ticks beyond tolerance, t = 81.90 to 83.35 s (guidance steps [40, 41]; guided flight ends at 110.00 s); worst at t = 82.65 s, the estimate 15.29 m above and 91.08 m from the site in force
- 20261062 d71_trn_e0_g168.0 e0 (safe landing), 6 site lit: margin -3.137 against tolerance 0.1832 deg; touchdown at map (1747.7, 6577.7) m, epoch 0, t = 163.94 s; sun elevation 1.7067 deg against horizon 4.8442 deg
- 20261062 d72_trn_e0_g-24.0 e0 (safe landing), 6 site lit: margin -0.3206 against tolerance 0.01364 deg; touchdown at map (1700.2, 6505.3) m, epoch 0, t = 135.66 s; sun elevation 1.7039 deg against horizon 2.0245 deg
- 20261062 d75_trn_e0_g-24.0 e0 (safe landing), 6 site lit: margin -0.5469 against tolerance 0.01728 deg; touchdown at map (1709.9, 6574.1) m, epoch 0, t = 139.39 s; sun elevation 1.7060 deg against horizon 2.2529 deg
- 20261062 d75_trn_e0_g168.0 e0 (safe landing), 6 site lit: margin -2.246 against tolerance 0.08113 deg; touchdown at map (1739.4, 6555.5) m, epoch 0, t = 141.38 s; sun elevation 1.7060 deg against horizon 3.9522 deg
- 20261062 d77_trn_e0_g72.0 e0 (safe landing), 6 site lit: margin -0.5639 against tolerance 0.02904 deg; touchdown at map (1704.6, 6513.4) m, epoch 0, t = 146.26 s; sun elevation 1.7042 deg against horizon 2.2680 deg
- 20261062 d80_trn_e0_g0.0 e0 (safe landing), 6 site lit: margin -0.1622 against tolerance 0.02108 deg; touchdown at map (1697.3, 6516.3) m, epoch 0, t = 142.45 s; sun elevation 1.7041 deg against horizon 1.8664 deg
- 20261062 d80_trn_e0_g24.0 e0 (safe landing), 6 site lit: margin -0.09049 against tolerance 0.01882 deg; touchdown at map (1696.0, 6517.3) m, epoch 0, t = 136.16 s; sun elevation 1.7042 deg against horizon 1.7946 deg
- 20261062 d81_trn_e0_g168.0 e0 (safe landing), 6 site lit: margin -3.026 against tolerance 0.1396 deg; touchdown at map (1764.8, 6563.8) m, epoch 0, t = 140.22 s; sun elevation 1.7066 deg against horizon 4.7328 deg
- 20261062 d84_trn_e0_g0.0 e0 (safe landing), 6 site lit: margin -0.297 against tolerance 0.0184 deg; touchdown at map (1699.9, 6513.5) m, epoch 0, t = 136.57 s; sun elevation 1.7041 deg against horizon 2.0011 deg
- 20261062 d85_trn_e0_g-24.0 e0 (unsafe landing), 3 glide slope: margin -0.3966 against tolerance 0.1876 deg; 30 of 2160 guided ticks beyond tolerance, t = 82.00 to 83.45 s (guidance steps [40, 41]; guided flight ends at 108.00 s); worst at t = 82.70 s, the estimate 16.82 m above and 99.40 m from the site in force
- 20261062 d85_trn_e1_g-72.0 e1 (unsafe landing), 3 glide slope: margin -0.7285 against tolerance 0.2377 deg; 37 of 2120 guided ticks beyond tolerance, t = 82.05 to 83.85 s (guidance steps [41]; guided flight ends at 106.00 s); worst at t = 82.90 s, the estimate 12.80 m above and 78.42 m from the site in force
- 20261062 d85_trn_e1_g72.0 e1 (unsafe landing), 3 glide slope: margin -0.386 against tolerance 0.2041 deg; 24 of 2080 guided ticks beyond tolerance, t = 82.80 to 83.95 s (guidance steps [41]; guided flight ends at 104.00 s); worst at t = 83.30 s, the estimate 15.46 m above and 91.26 m from the site in force
- 20261062 d85_trn_e2_g-72.0 e2 (safe landing), 3 glide slope: margin -0.4877 against tolerance 0.3526 deg; 15 of 2040 guided ticks beyond tolerance, t = 83.15 to 83.85 s (guidance steps [41]; guided flight ends at 102.00 s); worst at t = 83.50 s, the estimate 8.89 m above and 53.08 m from the site in force
- 20261062 d85_trn_e3_g-72.0 e3 (unsafe landing), 3 glide slope: margin -0.6135 against tolerance 0.2991 deg; 25 of 2040 guided ticks beyond tolerance, t = 82.75 to 83.95 s (guidance steps [41]; guided flight ends at 102.00 s); worst at t = 83.30 s, the estimate 10.33 m above and 62.48 m from the site in force
- 20261062 d86_trn_e0_g168.0 e0 (safe landing), 6 site lit: margin -2.114 against tolerance 0.2442 deg; touchdown at map (1733.4, 6584.9) m, epoch 0, t = 138.35 s; sun elevation 1.7067 deg against horizon 3.8204 deg
- 20261062 d86_trn_e0_g72.0 e0 (safe landing), 6 site lit: margin -0.348 against tolerance 0.04746 deg; touchdown at map (1703.4, 6543.4) m, epoch 0, t = 137.24 s; sun elevation 1.7050 deg against horizon 2.0530 deg
- 20261062 d87_trn_e0_g168.0 e0 (safe landing), 6 site lit: margin -2.787 against tolerance 0.3269 deg; touchdown at map (1752.7, 6563.5) m, epoch 0, t = 134.65 s; sun elevation 1.7064 deg against horizon 4.4930 deg
- 20261062 d88_trn_e0_g168.0 e0 (safe landing), 6 site lit: margin -2.186 against tolerance 0.02334 deg; touchdown at map (729.6, 6468.7) m, epoch 0, t = 197.67 s; sun elevation 1.6868 deg against horizon 3.8733 deg
- 20261062 d88_trn_e2_g-168.0 e2 (safe landing), 3 glide slope: margin -0.05088 against tolerance 0.01686 deg; 36 of 2920 guided ticks beyond tolerance, t = 75.15 to 76.90 s (guidance steps [37, 38]; guided flight ends at 146.00 s); worst at t = 75.85 s, the estimate 196.57 m above and 1120.60 m from the site in force
- 20261062 d91_trn_e0_g24.0 e0 (safe landing), 6 site lit: margin -0.02075 against tolerance 0.0176 deg; touchdown at map (1694.1, 6499.8) m, epoch 0, t = 145.97 s; sun elevation 1.7036 deg against horizon 1.7244 deg
- 20261062 d92_trn_e0_g168.0 e0 (safe landing), 6 site lit: margin -3.303 against tolerance 0.0006199 deg; touchdown at map (1769.9, 6565.7) m, epoch 0, t = 144.67 s; sun elevation 1.7068 deg against horizon 5.0099 deg
- 20261062 d92_trn_e2_g-168.0 e2 (unsafe landing), 3 glide slope: margin -1.438 against tolerance 0.4147 deg; 37 of 2240 guided ticks beyond tolerance, t = 94.05 to 95.85 s (guidance steps [47]; guided flight ends at 112.00 s); worst at t = 95.00 s, the estimate 6.88 m above and 45.73 m from the site in force
- 20261062 d93_trn_e0_g168.0 e0 (safe landing), 6 site lit: margin -1.698 against tolerance 0.2311 deg; touchdown at map (1727.8, 6568.9) m, epoch 0, t = 141.99 s; sun elevation 1.7062 deg against horizon 3.4037 deg
- 20261062 d93_trn_e3_g-72.0 e3 (unsafe landing), 3 glide slope: margin -1.203 against tolerance 0.5242 deg; 28 of 2000 guided ticks beyond tolerance, t = 82.55 to 83.90 s (guidance steps [41]; guided flight ends at 100.00 s); worst at t = 83.20 s, the estimate 5.57 m above and 36.01 m from the site in force
- 20261062 d97_trn_e0_g0.0 e0 (safe landing), 6 site lit: margin -0.1637 against tolerance 0.03438 deg; touchdown at map (1697.2, 6519.6) m, epoch 0, t = 139.39 s; sun elevation 1.7042 deg against horizon 1.8679 deg
- 20261062 d98_trn_e0_g168.0 e0 (safe landing), 6 site lit: margin -2.854 against tolerance 0.1037 deg; touchdown at map (1752.9, 6559.9) m, epoch 0, t = 143.96 s; sun elevation 1.7063 deg against horizon 4.5606 deg
- 20261062 d98_trn_e1_g24.0 e1 (unsafe landing), 3 glide slope: margin -1.588 against tolerance 1.257 deg; 17 of 1920 guided ticks beyond tolerance, t = 82.60 to 83.40 s (guidance steps [41]; guided flight ends at 96.00 s); worst at t = 83.15 s, the estimate 2.19 m above and 14.82 m from the site in force
- 20261062 d98_trn_e2_g-72.0 e2 (unsafe landing), 3 glide slope: margin -1.51 against tolerance 0.6135 deg; 30 of 2000 guided ticks beyond tolerance, t = 82.50 to 83.95 s (guidance steps [41]; guided flight ends at 100.00 s); worst at t = 83.15 s, the estimate 4.59 m above and 30.76 m from the site in force
