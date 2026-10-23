# combined: tonight's combination measured end to end on the 2,000-pair timeline sample

Variant `combined` = `night/combined` (worktree, base perf/fused-screen c6674ae) = merge of `night/hostside` 4c06c78
(shared-source run-table cache + fused mode + GaussDCA pre-gate; the pre-gate is NOT used here) and `night/policy-gate`
38b5908 (`couple.cascade.min_neffoverL` + degenerate-alignment guard). Merge clean (both touch
`config_template/project.yml` in different blocks, no conflict). Merge commit `39e08c6`; not pushed.
One GPU hold on data GPU 0: 00:32:37–00:37:30 (lock held 292.7 s). Same 2,000 pairs as classic/fused
(`data:$B/runs/night_timeline/pairs.txt`, copied), same `sampler.py` (v2), same `verify.py`. verify_ok true.

## 1. Phase table (classic vs fused vs combined; `timeline/phase_table.md`, same loader as the figure)

| phase | classic (norank; concat CPU → gauss → plm top 16 %, B ≤ 30k) | fused (norank; gauss inline, 64 host proc → plm top 16 %, cap 10k, 64 proc) | **combined** (gauss inline, 16 host proc, run-table cache → plm top 20 % + Neff/L ≥ 0.1, uncapped fp16 + maxeval 50, 16 proc) |
|---|---|---|---|
| concatenate (host) | 149.0 s · 1,999 pairs · 13.41 pairs/s · GPU 0 % · 53.5 cores | — | — |
| GaussDCA (GPU) | 110.6 s · 1,999 · **18.07 pairs/s** · 91.8 %† · 6.4 cores | 124.7 s · 1,999 · **16.03 pairs/s** · SM 87.9 % · 32.5 cores | 116.2 s · 1,999 · **17.20 pairs/s** · SM 93.4 % (device 94.4) · **7.2 cores** (peak 98) |
| cascade select | 1.3 s | 1.6 s | 1.7 s (kept 127 of 1,999: skipped by rank 1,599, by depth 273, unranked 0) |
| plmDCA (GPU) | 260.1 s · 317 pairs · 1.22 pairs/s (0.82 s/pair) · SM 92.9 % · 1.8 cores | 239.0 s · 317 · 1.33 pairs/s (0.75 s/pair) · SM 90.5 % · 3.1 cores | 166.4 s · **127 pairs · 0.763 pairs/s = 1.31 s/pair** · SM 92.4 % (device 94.0) · 1.3 cores |
| whole pipeline | 519.8 s · 3.85 pairs/s | 363.8 s · 5.50 pairs/s | **282.7 s · 7.07 pairs/s** |
| plm FAILED | 4 (3 × B ≤ 3 crash "Cannot normalize", 1 shared-source) | 4 (same) | **1** (shared-source pair O25802–O25871, carried from the GaussDCA FAILED row, unranked, `skip_unranked` false); **0 inside plmDCA** |
| GaussDCA vs screen DB | 20/20 identical | 20/20 identical | 20/20 identical (max diff 0) |
| host load / co-tenant SM | 120 / 0 % | 98 / 0 % | 45 / 0 % |

† classic's GaussDCA phase has device-level GPU % only (sampler v1). The plmDCA bands are three different pair sets
(top 16 % B ≤ 30k; top 16 % cap 10k; top 20 % + gate, uncapped), so compare the whole-pipeline row and the GaussDCA row,
not the plm pairs/s. The 3 degenerate pairs (B = 2–3, Beff 1, GaussDCA ranks 1/5/6 of the kept set) that crashed plmDCA
in classic and fused have Neff/L < 0.1 and were gated out, so the analyzer guard was not exercised in this run.

Hostside StageClock (gauss.log, first minute, P = 16): 1,077 pairs in 60 s (17.9/s); per pair load 0.181 s, waiting
for a GPU 1.468 s, on the GPU 0.216 s, analysis 0.060 s; sessions busy 97 %. The stage is GPU-bound at 7 host cores
(fused P = 64 without the cache: 32.5 cores for 16.0 pairs/s). plmDCA StageClock: 43 pairs/62 s then 72 pairs/61 s,
sessions busy 97 % / 100 %, 1 session.

## 2. Full-screen hours restated from the measured rates (PLAN §2 arithmetic; `timeline/projection.json`)

GaussDCA on all 1,206,681 pairs at 17.20 pairs/s = **19.49 h**. plmDCA on the gated set of **76,212 pairs** (20 % cut +
Neff/L ≥ 0.1 on `helpy_gauss.sqlite3`, policy-gate's count, re-derived here: ranked 1,193,603, top 20 % 238,721, gated
76,212) at the measured 1.3105 s/pair = **27.74 h** (naive). Ratio estimator on the cost model
(7.2e-7·N² + 3.8e-10·B·N², B ≤ 30k): r = 166.44 s / 177.7 model-s = 0.937 on the sample; population model 33.26 h ⇒
**31.2 h**. The 127-pair sample is lighter than the gated population (model 1.399 vs 1.571 s/pair; B > 10k 18.1 % vs
21.7 %; B q10/50/90 218/598/20,369 vs 206/744/20,703), so the naive 27.7 h is optimistic by ~12 %; policy-gate's
1,000-kept-pair measurement of the same profile gives 30.48 h [29.14, 32.36]. **Quote the plmDCA stage as 28–31 h.**

| GPUs | GaussDCA (all pairs, inline concat) | plmDCA (76,212 gated, uncapped fp16 + maxeval 50) | total | note |
|---|---|---|---|---|
| 1 H100 (measured rates, shared host) | 19.5 h | 27.7 h (ratio-adjusted 31.2) | **47.2 h (50.7)** | fused measured 61.3 h; brief "today" 58–60 h; ×1.30 vs fused |
| 2 H100 (linear) | 9.7 | 13.9 (15.6) | **23.6 (25.3)** | needs 16 host proc/GPU (7 cores/GPU measured) |
| 4 H100-eq (linear) | 4.9 | 6.9 (7.8) | **11.8 (12.7)** | + coordinator per 2 GPUs, claim index; 4th GPU unmeasured |
| plmDCA only on the finished GaussDCA screen | 0 | 27.7 / 13.9 / 6.9 (31.2 / 15.6 / 7.8) on 1 / 2 / 4 | | the "one night on 3–4 GPUs" case |

Not reached: "well under 24 h on one H100" (47 h); 2 GPUs land at the 24 h line. The rates are from a shared host
(load 45, GPU 0 co-tenant at 0 % SM) at 93 % SM by our own processes: lower bounds on throughput, upper bounds on hours;
the idle-H100 gap is at most the 6–7 % of SM we did not get. The K2 kernel (not accepted tonight) and the GaussDCA
pre-gate (hostside stretch: −24 % of the GaussDCA stage ⇒ ≈ 14.8 h here) are NOT in these numbers. If the synthesis
keeps the 10k cap instead of the uncapped profile, the plm stage is policy-gate's measured 23.8 h (total ≈ 43 h), not
measured in this run.

## 3. Hits retained (labelled protocol, from the eval tasks; nothing is measured on labels in this run)

- Policy (20 % cut + Neff/L ≥ 0.1; `policy-gate/REPORT.md` §1.3, protein bootstrap, he2021, in-sample): plm top-5 %
  positives **91 % [83, 97]** vs today's 16 % cut 91 % [83, 96]; detector hits **86 % [75, 94]** vs 84 % [72, 92]. Gate's
  own cost at a fixed 16 % cut: 97 % [92, 100] / 99 % [97, 100]. "No detectable loss"; CIs admit ~8–11 points; the
  benchmark has 13 of 4,545 pairs below the gate.
- Profile (uncapped fp16 + maxeval 50, `eval-screenprofile/REPORT.md` fp16m50, 1,247 pairs, 160 ref hits): hits
  76/24/61 = **+1 [0, +3]**, AUPRC −0.0037 [−0.008, +0.001], retw@455 −0.039 OK, deep-hit deflation +0.27 (none) →
  **ACCEPT**. (The 10k cap: −3 hits, −3.1 llr deflation of deep hits, REJECT on seed 0 — that is why the combined run is
  uncapped.)
- Combined = policy × profile: no detectable labelled loss from either factor; the combination was not scored on the
  labelled set as one run (0.75 GPU-h; not done tonight).
- In situ on the 2,000 pairs (artefact-dominated by construction, FINDING_shallow_artefacts.md): fused (top 16 %, cap
  10k) 23 `high_precision_hit` of 317; combined 1 of 127 (O25394–O26012, B 8,501, Beff 3,287, Neff/L 1.89, llr 1.89).
  All 22 fused hits absent from the combined set have Neff/L 0.002–0.093 (B 15–134, Beff 1–40): removed by the gate as
  designed. The one fused hit inside the gated set is retained. On the 85 pairs both runs scored, |Δlog_lr_raw|
  uncapped vs cap-10k: median 0.021, q90 0.31, max 1.16.

## 4. Exact commands

```
# worktree + merge (local)
git -C $ROLL worktree add $ROLL/../night/combined -b night/combined perf/fused-screen
cd $ROLL/../night/combined && git merge --no-edit night/hostside && git merge --no-edit night/policy-gate   # 39e08c6
# build on data
rsync -a --delete --exclude .git --exclude build --exclude '*.egg-info' --exclude __pycache__ $W/ data:$B/src-combined/
ssh data 'cd $B && bash tools/build_variant.sh combined'        # BUILD_OK, log $B/build-combined.log, .so 00:30:12
# stage + run (data), scripts in perf/night/combined/ and data:$B/runs/night_combined/
python -I setup_combined.py          # pairs.txt from night_timeline; gauss.yml / plm.yml; cache /dev/shm/kohlbach_ss_k10
nohup ./run_pipeline_combined.sh > ../combined.out 2>&1 &       # flock $B/.gpu0.lock, sampler.py v2, verify.py combined $R
# figures (local)
cd perf/night/timeline
rsync -a --exclude '*.sqlite3*' --exclude ecs --exclude msa data:$B/runs/night_combined/combined/ data/combined/
uv run --with matplotlib --with pandas --with numpy python fill_projection.py --run combined --plm-pairs 76212
uv run --with matplotlib --with pandas --with numpy python plot_timeline.py
uv run --with matplotlib --with pandas --with numpy python analyze.py
```

Settings (`data/combined/{gauss,plm}.yml`): gauss `method gauss, gpu_sessions 4, gpu_host_processes 16, gpu_workers 36,
keep_concatenated false, concatenate.shared-source_cache /dev/shm/kohlbach_ss_k10, downsample 30,000`, no cascade block
(no pre-gate); plm `method plm, maxit 50, fp16_history true, gpu_host_processes 16, gpu_workers 36, keep_concatenated
false, downsample 30,000 (uncapped), cascade {source_db gauss.sqlite3, keep_fraction 0.20, rank_by log_lr_raw,
min_neffoverL 0.1}`.

## 5. Raw data

- data: `$B/runs/night_combined/{combined.out,setup_combined.py,run_pipeline_combined.sh,pairs.txt}`,
  `$B/runs/night_combined/combined/{sampler.csv,phases.csv,counts.json,verify.txt,gauss.yml,plm.yml,gauss.sqlite3,plm.sqlite3,logs/}`
  (`logs/gauss.log`, `logs/plm.log` with the StageClock lines and the cascade line, `gpu_apps_{start,end}.txt`).
- local: `perf/night/timeline/data/combined/` (fetched copy without the DBs), `timeline/projection.json` (now from
  the combined run; the previous fused-run version is reproducible with `fill_projection.py --run fused`),
  `timeline/timeline_measured.{png,pdf}` (three rows), `timeline/timeline_projected.{png,pdf}` (bar (b) = combined),
  `timeline/phase_table.{md,json}`, `timeline/phases.png`.
- code changes: `timeline/plot_timeline.py` (combined row: title, default settings, `min_neffoverL` in the settings
  line, run list, footnote, subtitle, legend), `timeline/analyze.py` (RUNS), `timeline/fill_projection.py` (`--run`,
  `--plm-pairs`).
- worktree `night/combined`: merge 39e08c6 + `perf/night/combined/{REPORT.md,setup_combined.py,run_pipeline_combined.sh}`.

## 6. Caveats

1. One run, one host, 127 plmDCA pairs: the plm s/pair carries sampling error (ratio estimator 31.2 h vs naive 27.7 h;
   policy-gate's 1,000-pair measurement 30.5 h). The plm stage wall includes apply_cascade (1.7 s), session start and
   the queue ramp (first-minute wait 21 s/pair), a larger share at 127 pairs than over 76,212.
2. Shared host and GPU: our SM share 92–93 %, device 94 %; host load 45 (fused ran at 98, classic at 120), so part of
   combined's GaussDCA gain over fused (17.2 vs 16.0 pairs/s) may be the quieter host; hostside's own like-for-like
   A/B gives ×1.08 (17.4 vs 16.0). The cache's measurable effect here is the host cores (7.2 vs 32.5), not the rate.
3. 2- and 4-GPU rows are linear projections; nothing multi-GPU was run.
4. Hit retention is the eval tasks' labelled result (in-sample, CI ~10 points), applied multiplicatively; the
   combination itself was not scored on labels. In-situ hit counts prove nothing (artefacts).
5. The gated population count 76,212 is over today's 1,193,603 DONE rows; task 5's finished 13,078 rows add
   +1.0–5.0 h to the plm stage (policy-gate open issue). The shared-source pair is FAILED in both stages, as in fused.
6. Only the uncapped profile was run; the 10k-cap combined pipeline (23.8 h plm stage by policy-gate) is not measured
   end to end. K2 and the GaussDCA pre-gate are excluded.
7. `tools/build_variant.sh combined` finished in ~10 s because `$B/build-combined` already existed (a pre-existing
   cmake build dir of that name); the four `.so` files carry the 00:30:12 timestamp and the merged Python
   (`min_neffoverL` in db.py, `shared-source_cache` in concatenate.py/couple.py) is in the variant.
