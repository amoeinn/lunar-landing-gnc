"""Mutations for tests/test_result_figures.py (step 8.2), run with
work/step11a/mutharness.py. Every mutation targets what a data function reads
or how it orders, filters or groups it."""
V = "examples/make_result_figures.py"
M = [
    ("safe fraction read at epoch 0", V, 'safe=[sm["safe_fraction"][k]["pooled"] for k in ks]',
     'safe=[sm["safe_fraction"][k]["epoch 0"] for k in ks]'),
    ("median Q read as the p90", V, 'q_median=[sm["q"][k]["pooled"]["median"] for k in ks]',
     'q_median=[sm["q"][k]["pooled"]["p90"] for k in ks]'),
    ("no-TRN level read at epoch 0", V, 'notrn_safe=sm["safe_fraction"]["no-TRN"]["pooled"]',
     'notrn_safe=sm["safe_fraction"]["no-TRN"]["epoch 0"]'),
    ("gaps in file order", V, 'return sorted((k for k in d if k != "no-TRN"), key=gap_hours)',
     'return [k for k in d if k != "no-TRN"]'),
    ("reference seeds read as the scored seed", V, 'sm = load(str(P6_SCORE).format(s))["summary_6_2"]',
     'sm = load(str(P6_SCORE).format(P6_SEED))["summary_6_2"]'),
    ("split read at epoch 0", V, 'counts={c: [tw[k]["pooled"][c] for k in ks] for c in SPLIT}',
     'counts={c: [tw[k]["epoch 0"][c] for k in ks] for c in SPLIT}'),
    ("no-TRN bar dropped", V, 'ks = gap_keys(tw) + ["no-TRN"]', 'ks = gap_keys(tw)'),
    ("premise H read as the fix median", V, 'h=[tab[k]["worse"] for k in ks]', 'h=[tab[k]["fix_median"] for k in ks]'),
    ("premise gaps with no fix kept", V, 'ks = [k for k in sorted(tab, key=gap_hours) if tab[k]["fixes"] > 0]',
     'ks = [k for k in sorted(tab, key=gap_hours)]'),
    ("premise read at one fix time", V, 'tab = d["tables"][str(s)]["all"]', 'tab = d["tables"][str(s)]["40.0"]'),
    ("phase 5 system and blind swapped", V,
     'system=[r[k]["safe"]["system"] for k in eps], blind=[r[k]["safe"]["blind"] for k in eps],',
     'system=[r[k]["safe"]["blind"] for k in eps], blind=[r[k]["safe"]["system"] for k in eps],'),
    ("rock fields out of order", V, 'ROCK_FIELDS = ("minimum", "nominal", "maximum")',
     'ROCK_FIELDS = ("minimum", "maximum", "nominal")'),
    ("seed-alone counts read from the sensitivity run", V,
     'one = {c: pc[c]["registered"].get("FAIL", 0) for c in checks}',
     'one = {c: pc[c]["sensitivity"].get("FAIL", 0) for c in checks}'),
    ("glide causes read at the worst violation", V,
     'return {"3 glide slope": dd["glide"]["first"]["primary"], "4 terrain clearance":',
     'return {"3 glide slope": dd["glide"]["worst"]["primary"], "4 terrain clearance":'),
    ("clearance causes read by sample", V, 'dd["clearance"]["flight_causes"], "6 site lit"',
     'dd["clearance"]["sample_causes"], "6 site lit"'),
    ("seed-alone lit causes read from the sensitivity run", V,
     '"6 site lit": r1["lit_primary"]}', '"6 site lit": ds["lit"]["primary"]}'),
]
TESTS = ["tests/test_result_figures.py", "-k", "not end_to_end"]
