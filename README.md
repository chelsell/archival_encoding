# AV1 Zebrafish Archive Tuning

This repo is now organized around one narrow workflow:

1. Make short representative clips from the original HEVC videos.
2. Transcode those clips over a small AV1 parameter grid.
3. Compare each transcode to its source clip using the `cd10` motion feature.

The older analysis files are still here for reference, but the default workflow is just:

- [`make_window_proxies.py`](/home/cole/code/archival_encoding/make_window_proxies.py)
- [`compare_av1_grid.py`](/home/cole/code/archival_encoding/compare_av1_grid.py)

## What We Measure

`cd10` is the per-frame count of pixels whose absolute brightness change exceeds `tau=10`.

For each AV1 transcode, we compare its `cd10` time series to the source clip and record:

- `p95_pixpct`: 95th percentile of `|delta cd10|`, normalized by frame pixels
- `ccc`: concordance correlation coefficient
- `size_mb`: output size in MB

These are enough to see the size/quality tradeoff and identify the QP cliff. If you want more diagnostics, the comparison script can also compute:

- `mi_*`: the same comparison statistics on motion intensity, where MI is `sum(abs(frame_t - frame_t-1))`
- `psnr`
- `ssim`

## Step 1: Create Representative Clips

Use [`make_window_proxies.py`](/home/cole/code/archival_encoding/make_window_proxies.py) to extract a few short windows from each long source video and concatenate them into a smaller proxy clip.

Example:

```bash
python make_window_proxies.py \
  --src /path/to/hevc_videos \
  --dst /path/to/clips \
  --sample-secs 10 \
  --sample-positions 0.1 0.5 0.9 \
  --threads 4
```

What this does:

- picks three 10-second windows per video
- samples near 10%, 50%, and 90% of duration
- writes one clip per source video into `--dst`
- writes `proxy_manifest.csv` for traceability

## Step 2: Compare an AV1 Grid

Use [`compare_av1_grid.py`](/home/cole/code/archival_encoding/compare_av1_grid.py) on those clips.

Example:

```bash
python compare_av1_grid.py \
  --clips /path/to/clips \
  --out /path/to/av1_grid \
  --qp 33 35 37 39 \
  --g 240 \
  --la 0 \
  --preset 4 \
  --tau 10 \
  --metrics mi psnr \
  --threads 4
```

What this does:

- transcodes each clip for every `qp x g x la` combination
- computes `cd10` on the source clip and each transcode
- optionally computes MI, PSNR, and SSIM
- writes one flat summary CSV at `--out/summary.csv`

Output columns:

- `clip`
- `qp`, `g`, `la`, `preset`
- `size_mb`
- `pearson`
- `ccc`
- `med_abs`
- `p95_abs`
- `p95_pixpct`

Optional columns:

- `mi_pearson`
- `mi_ccc`
- `mi_med_abs`
- `mi_p95_abs`
- `psnr`
- `ssim`

Each clip also gets its own output directory containing the transcodes.

## Suggested Default Grid

Unless you are deliberately testing structure, keep these fixed:

- `g=240`
- `la=0`
- `preset=4`

Then sweep:

- `qp=33 35 37 39`

If you want a wider bracket, start with:

- `qp=23 27 31 33 35 37 39`

## Reading Results

The usual pattern is:

- lower `size_mb` is better
- lower `p95_pixpct` is better
- higher `ccc` is better

Practical rule of thumb from the earlier work:

- `p95_pixpct <= 0.08`
- `ccc >= 0.995`

Within that acceptable region, pick the smallest file.

## Notes

- `compare_av1_grid.py` prefers `libsvtav1` and falls back to `libaom-av1`.
- Both scripts support `--ffmpeg-prefix` if you need to run ffmpeg through a container such as `apptainer exec ...`.
- The exploratory files like [`av1_la_grid.py`](/home/cole/code/archival_encoding/av1_la_grid.py), plotting scripts, and older summaries are not required for the simplified workflow.
