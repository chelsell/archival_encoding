Here’s a tight “context pack” you can drop into a README header or a top-of-file block comment so Codex knows what we’re doing and why.

# AV1 zebrafish compression study — approach + fixed decisions

**Goal.** Find AV1 settings that make files **smaller than HEVC** while **preserving behaviorally relevant motion** measured by a simple feature: `cd10` = per-frame count of pixels whose absolute brightness delta exceeds τ=10.

**Why cd10.** It’s cheap, robust on grayscale tank videos, and correlates with the “does it still look like the same bout?” reviewer demand better than PSNR/SSIM.

---

## Metrics we trust (and the ones we don’t)

* **Primary:** `p95pix%` = 100 × p95(|Δcd10|) / frame_pixels. Resolution-invariant, not fooled by low-activity frames. This is the cliff detector.
* **Shape:** `CCC` (concordance correlation) between cd10 time series. Guardrail for obvious distortions.
* **Ignore for decisions:** the “activity-aware” `%~` columns are diagnostic only; they can still blow up on near-zero cd10.

**Acceptable band (current):**

* `p95pix% ≤ 0.08` and `CCC ≥ 0.995` → “preserved”.

---

## Encoder space and what we fixed

* **Monochrome AV1, 10-bit:** SVT-AV1 (fallback: libaom).
* **GOP length:** **fixed at `g=240`**. This gave a 10–30× size drop vs `g=1` with negligible cd10 change. It’s the structural knee; we don’t vary it anymore.
* **Lookahead:** **`la=0`**. In CQP, `la>0` mostly doesn’t help and sometimes increases size; we observed no cd10 benefit.
* **Preset:** **`preset=4`** (stable middle ground). We’re not using preset as a quality knob.
* **Quantizer (QP):** **the only variable that matters now**. We sweep to find the **QP cliff** (first sharp jump in `p95pix%` / drop in `CCC`) and choose **one step before** it.

---

## QP search policy (the “cliff”)

* Start at **QP {23,27,31}** to confirm the flat region.
* Then probe **{33,35,37,39}** at fixed `g=240, la=0, preset=4`.
* **Cliff rule:** first QP where either:

  * `p95pix%` jumps by **≥1.6×** vs the previous QP *and* is **≥0.06**, or
  * `CCC < 0.995`.
* **Per-video recommend:** the **QP just before** that cliff.
* **Global pick:** the **max** of per-video recommendations (most conservative).
  (If we bucket by difficulty: “easy” can go to 39, “medium” ~37, “hard” ~35.)

---

## Speed/scalability choices

* **Sampling not full decode:** analyze **3× 60-s windows** per video at **10%, 50%, 90%** of duration (fast seek with `-ss` *before* `-i`).
* **Stride:** default **1** (analysis), **2** when speed matters; invariant for rankings.
* **Parallelism:** `--jobs N` (per-video multiprocessing) + `--threads` for ffmpeg decode.
* **Caching:**

  * Baseline cd10: `baseline.tau{τ}.s{stride}[.ss{start}t{dur}…].npy`
  * Test cd10 per setting: `qp{qp}.g{g}.la{la}.tau{τ}.s{stride}[.ss…].npy`
* **File naming for encodes:** `{stem}/qp{qp}.g{g}.la{la}.mkv`
* **Per-video summary:** `{stem}/summary.csv`
  Columns include: `video, qp, g, la, size_mb, p95_pixpct, ccc, ...`
* **Merged analysis:** `analysis/all_summary_merged.csv` + plots.

---

## Existence proof (HEVC vs AV1)

For each stem, among rows meeting `p95pix% ≤ 0.08 && CCC ≥ 0.995`, choose the **smallest AV1 file** and compare to the source HEVC size. Emit `existence_size_ratios.csv` with:

`video, src_mb, best_qp, best_mb, ratio(=AV1/HEVC), p95pix, ccc, note`

If stems don’t match 1:1, we still write a row with `note` (“no matching HEVC stem” or “no AV1 row meets thresholds”).

---

## Typical commands

**Encode + measure a QP sweep (new stems or gaps):**

```bash
python av1_la_grid.py \
  --indir /media/ssd1/sauronx_videos/hevc_sample \
  --out   /media/ssd1/sauronx_videos/av1_grid \
  --tau 10 --g 240 --la 0 --preset 4 \
  --qp 33 35 37 39 \
  --sample-secs 60 --sample-positions 0.1 0.5 0.9 \
  --stride 1 --jobs 8 --threads 4
```

**Re-index features without re-encoding (fast):**

```bash
python av1_la_grid.py \
  --indir /media/ssd1/sauronx_videos/hevc_sample \
  --out   /media/ssd1/sauronx_videos/av1_grid \
  --features-only --rewrite-summary \
  --tau 10 --g 240 --la 0 --preset 4 \
  --sample-secs 60 --sample-positions 0.1 0.5 0.9 \
  --stride 1 --jobs 8 --threads 4
```

**Aggregate, call the cliff, make the plots, and compare to HEVC:**

```bash
python -u merge_and_analyze_qp.py \
  --root /media/ssd1/sauronx_videos/av1_grid \
  --hevc-root /media/ssd1/sauronx_videos/hevc_sample \
  --abs-floor 0.06 --cliff-ratio 1.6 --ccc-min 0.995 \
  --max-p95pix 0.08 --min-ccc 0.995
```

---

## Edge notes Codex should remember

* The data are **grayscale zebrafish tanks** with uniform dynamics; telecentric lens means negligible Z parallax—fine for cd10.
* `la>0` under SVT-AV1 **CQP** is a time sink; we deliberately fix **`la=0`**.
* `g=1` runs are **guardrails** only; we don’t consider them candidates.
* If sampling windows or `stride` change, **baseline and test must match** (we always recompute cd10 under identical sampling).
* When in doubt, prefer `p95pix%` over relative `%~` metrics.


