#!/usr/bin/env python3
# Scalable cd10 vs AV1 grid with sampling + parallelism
# - Temporal windows via fast seek (-ss before -i): --sample-secs, --sample-positions
# - Multiprocessing across videos: --jobs
# - Stride, ROI, decoder threads (from prior version)
# - Features-only reindex mode works with sampling/caches

import argparse, os, sys, subprocess, json, re, csv, math, time, shlex, tempfile
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

from multiprocessing import Pool, cpu_count

try:
    import matplotlib.pyplot as plt
    _HAS_PLOT = True
except Exception:
    _HAS_PLOT = False

# ---------- utilities ----------
# ffmpeg/ffprobe command wrapper (supports apptainer/singularity prefixes)
FFMPEG_PREFIX: list[str] = []
FFMPEG_BIN: str = "ffmpeg"
FFPROBE_BIN: str = "ffprobe"

def set_ffmpeg_prefix(prefix: str | None, ffmpeg_bin: str, ffprobe_bin: str):
    global FFMPEG_PREFIX, FFMPEG_BIN, FFPROBE_BIN
    FFMPEG_PREFIX = shlex.split(prefix) if prefix else []
    FFMPEG_BIN = ffmpeg_bin
    FFPROBE_BIN = ffprobe_bin

def ffmpeg_cmd(args: list[str]):
    return FFMPEG_PREFIX + [FFMPEG_BIN] + args

def ffprobe_cmd(args: list[str]):
    return FFMPEG_PREFIX + [FFPROBE_BIN] + args

def run(cmd: list[str]):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or "ffmpeg failed")

def ffprobe_json(args):
    out = subprocess.check_output(ffprobe_cmd(["-v", "error", "-of", "json"] + args), text=True)
    return json.loads(out)

def probe_gray_shape(path: Path):
    j = ffprobe_json(["-select_streams", "v:0", "-show_entries", "stream=width,height", str(path)])
    s = j["streams"][0]
    return int(s["width"]), int(s["height"])

def probe_duration(path: Path) -> float:
    j = ffprobe_json(["-show_entries", "format=duration", "-i", str(path)])
    return float(j["format"]["duration"])

def has_encoder(name: str) -> bool:
    try:
        out = subprocess.check_output(ffmpeg_cmd(["-hide_banner", "-encoders"]), text=True)
        return name in out
    except Exception:
        return False

def ffmpeg_supports_svt_params() -> bool:
    try:
        out = subprocess.check_output(ffmpeg_cmd(["-hide_banner", "-h", "encoder=libsvtav1"]),
                                      text=True, stderr=subprocess.STDOUT)
        return "-svtav1-params" in out
    except Exception:
        return False

def encoder_supports_option(encoder: str, token: str) -> bool:
    """Return True if ffmpeg -h encoder=<encoder> output contains token."""
    try:
        out = subprocess.check_output(ffmpeg_cmd(["-hide_banner", "-h", f"encoder={encoder}"]),
                                      text=True, stderr=subprocess.STDOUT)
        return token in out
    except Exception:
        return False

def has_filter(token: str) -> bool:
    try:
        out = subprocess.check_output(ffmpeg_cmd(["-hide_banner","-filters"]), text=True)
        return token in out
    except Exception:
        return False
HAS_SVT = HAS_AOM = HAS_PARAM = HAS_MONO_SVT = HAS_MONO_AOM = False
HAS_VMAF = False

# ---------- cd10 core with sampling ----------
def _cd10_rawpipe(cmd, w, h, tau, calc_mi=False):
    bpf = w*h
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    def read_frame():
        buf = p.stdout.read(bpf)
        if not buf or len(buf) < bpf:
            return None
        return np.frombuffer(buf, np.uint8).reshape(h, w)
    prev = read_frame()
    if prev is None:
        p.wait(); 
        if calc_mi:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
        return np.empty((0,), dtype=np.int64)
    vals, mi_vals, diff = [], [], np.empty((h, w), dtype=np.int16)
    while True:
        cur = read_frame()
        if cur is None:
            break
        np.subtract(cur, prev, dtype=np.int16, casting="unsafe", out=diff)
        vals.append(int(np.sum(np.abs(diff) > tau, dtype=np.int64)))
        if calc_mi:
            mi_vals.append(int(np.sum(np.abs(diff), dtype=np.int64)))
        prev = cur
    p.wait()
    if calc_mi:
        return np.asarray(vals, dtype=np.int64), np.asarray(mi_vals, dtype=np.int64)
    return np.asarray(vals, dtype=np.int64)

def _ffmpeg_vf(roi, stride):
    vf = []
    if roi:
        x0,y0,x1,y1 = roi
        vf.append(f"crop={x1-x0}:{y1-y0}:{x0}:{y0}")
    if stride and stride>1:
        vf.append(f"select='not(mod(n\\,{stride}))'")
    return ",".join(vf) if vf else None

def cd10_windows(video: Path, tau: int, roi=None, stride: int = 1, threads: int = 0,
                 windows: list[tuple[float,float]] = None, calc_mi: bool = False):
    """Concatenate cd10 (and optional MI=sum|Δ|) from multiple windows using fast seek."""
    if not windows:
        # fall back to full decode (not recommended for long vids)
        w,h = probe_gray_shape(video)
        cmd = ffmpeg_cmd(["-hide_banner","-loglevel","error","-vsync","0"])
        if threads>0: cmd += ["-threads", str(threads)]
        cmd += ["-i", str(video)]
        vf = _ffmpeg_vf(roi, stride)
        if vf: cmd += ["-vf", vf]
        cmd += ["-f","rawvideo","-pix_fmt","gray","pipe:"]
        res = _cd10_rawpipe(cmd, w, h, tau, calc_mi=calc_mi)
        return res if calc_mi else (res, None)

    series = []
    mi_series = []
    # Determine dimensions once
    if roi:
        w,h = (roi[2]-roi[0], roi[3]-roi[1])
    else:
        w,h = probe_gray_shape(video)

    for start, dur in windows:
        # Fast seek: -ss BEFORE -i; approximate but very fast for long GOP inputs
        cmd = ffmpeg_cmd(["-hide_banner","-loglevel","error","-vsync","0","-ss", f"{start}","-t", f"{dur}"])
        if threads>0: cmd += ["-threads", str(threads)]
        cmd += ["-i", str(video)]
        vf = _ffmpeg_vf(roi, stride)
        if vf: cmd += ["-vf", vf]
        cmd += ["-f","rawvideo","-pix_fmt","gray","pipe:"]
        res = _cd10_rawpipe(cmd, w, h, tau, calc_mi=calc_mi)
        if calc_mi:
            cd10_arr, mi_arr = res
            series.append(cd10_arr)
            mi_series.append(mi_arr)
        else:
            series.append(res)
    if not series:
        return (np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64) if calc_mi else None)
    cd10_concat = np.concatenate(series)
    mi_concat = np.concatenate(mi_series) if calc_mi else None
    return cd10_concat, mi_concat

# ---------- Quality metrics over sampled windows ----------
def _avg_stat_from_file(tmp_path: Path, keys):
    total = {k:0.0 for k in keys}
    frames = 0
    with tmp_path.open() as f:
        for line in f:
            parts = dict(p.split(":") for p in line.strip().split() if ":" in p)
            if not parts:
                continue
            frames += 1
            for k in keys:
                try:
                    total[k] += float(parts.get(k, 0.0))
                except Exception:
                    pass
    return total, frames

def psnr_windows(ref: Path, tst: Path, roi=None, stride: int = 1, threads: int = 0,
                 windows: list[tuple[float,float]] = None, peak: int = 1023):
    if not tst.exists() or not ref.exists():
        return None
    windows = windows or [(0, None)]
    total_mse = 0.0; total_frames = 0
    for start, dur in windows:
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        vf_parts = []
        if roi:
            x0,y0,x1,y1 = roi
            vf_parts.append(f"crop={x1-x0}:{y1-y0}:{x0}:{y0}")
        if stride and stride>1:
            vf_parts.append(f"select='not(mod(n\\,{stride}))'")
            vf_parts.append("setpts=N/(FRAME_RATE*TB)")
        vf_chain = ",".join(vf_parts) if vf_parts else None
        filters = []
        if vf_chain:
            filters.append(f"[0:v]{vf_chain}[r]")
            filters.append(f"[1:v]{vf_chain}[t]")
            psnr_in = "[r][t]"
        else:
            psnr_in = "[0:v][1:v]"
        filters.append(f"{psnr_in}psnr=stats_file={tmp_path}")
        fc = ";".join(filters)
        cmd = ffmpeg_cmd(["-hide_banner","-loglevel","error",
                          "-ss", f"{start}"] + (["-t", f"{dur}"] if dur else []) + ["-i", str(ref),
                          "-ss", f"{start}"] + (["-t", f"{dur}"] if dur else []) + ["-i", str(tst),
                          "-filter_complex", fc,
                          "-an","-f","null","-"])
        subprocess.run(cmd, check=True)
        stat, frames = _avg_stat_from_file(tmp_path, ["mse_avg"])
        try: tmp_path.unlink()
        except Exception: pass
        total_mse += stat["mse_avg"]
        total_frames += frames
    if total_frames == 0:
        return None
    mse_avg = total_mse / total_frames
    if mse_avg <= 0:
        return None
    return 10 * math.log10((peak**2) / mse_avg)

def ssim_windows(ref: Path, tst: Path, roi=None, stride: int = 1, threads: int = 0,
                 windows: list[tuple[float,float]] = None):
    if not tst.exists() or not ref.exists():
        return None
    windows = windows or [(0, None)]
    total_ssim = 0.0; total_frames = 0
    for start, dur in windows:
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        vf_parts = []
        if roi:
            x0,y0,x1,y1 = roi
            vf_parts.append(f"crop={x1-x0}:{y1-y0}:{x0}:{y0}")
        if stride and stride>1:
            vf_parts.append(f"select='not(mod(n\\,{stride}))'")
            vf_parts.append("setpts=N/(FRAME_RATE*TB)")
        vf_chain = ",".join(vf_parts) if vf_parts else None
        filters = []
        if vf_chain:
            filters.append(f"[0:v]{vf_chain}[r]")
            filters.append(f"[1:v]{vf_chain}[t]")
            ssim_in = "[r][t]"
        else:
            ssim_in = "[0:v][1:v]"
        filters.append(f"{ssim_in}ssim=stats_file={tmp_path}")
        fc = ";".join(filters)
        cmd = ffmpeg_cmd(["-hide_banner","-loglevel","error",
                          "-ss", f"{start}"] + (["-t", f"{dur}"] if dur else []) + ["-i", str(ref),
                          "-ss", f"{start}"] + (["-t", f"{dur}"] if dur else []) + ["-i", str(tst),
                          "-filter_complex", fc,
                          "-an","-f","null","-"])
        subprocess.run(cmd, check=True)
        stat, frames = _avg_stat_from_file(tmp_path, ["All"])
        try: tmp_path.unlink()
        except Exception: pass
        total_ssim += stat["All"]
        total_frames += frames
    if total_frames == 0:
        return None
    return total_ssim / total_frames

def vmaf_windows(ref: Path, tst: Path, roi=None, stride: int = 1, threads: int = 0,
                 windows: list[tuple[float,float]] = None, model_path: str = None):
    if not HAS_VMAF or not tst.exists() or not ref.exists():
        return None
    windows = windows or [(0, None)]
    total_vmaf = 0.0; total_frames = 0
    for start, dur in windows:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        vf_parts = []
        if roi:
            x0,y0,x1,y1 = roi
            vf_parts.append(f"crop={x1-x0}:{y1-y0}:{x0}:{y0}")
        if stride and stride>1:
            vf_parts.append(f"select='not(mod(n\\,{stride}))'")
            vf_parts.append("setpts=N/(FRAME_RATE*TB)")
        vf_chain = ",".join(vf_parts) if vf_parts else None
        filters = []
        if vf_chain:
            filters.append(f"[0:v]{vf_chain}[r]")
            filters.append(f"[1:v]{vf_chain}[t]")
            in_a, in_b = "[r]", "[t]"
        else:
            in_a, in_b = "[0:v]", "[1:v]"
        model = f":model_path={model_path}" if model_path else ""
        filters.append(f"{in_a}{in_b}libvmaf=log_fmt=json:log_path={tmp_path}{model}")
        fc = ";".join(filters)
        cmd = ffmpeg_cmd(["-hide_banner","-loglevel","error",
                          "-ss", f"{start}"] + (["-t", f"{dur}"] if dur else []) + ["-i", str(ref),
                          "-ss", f"{start}"] + (["-t", f"{dur}"] if dur else []) + ["-i", str(tst),
                          "-filter_complex", fc,
                          "-an","-f","null","-"])
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            # If VMAF fails (e.g., missing model), skip gracefully
            try: Path(tmp_path).unlink()
            except Exception: pass
            return None
        try:
            j = json.load(open(tmp_path))
            for fr in j.get("frames", []):
                m = fr.get("metrics", {})
                if "vmaf" in m:
                    total_vmaf += float(m["vmaf"])
                    total_frames += 1
        except Exception:
            pass
        try: Path(tmp_path).unlink()
        except Exception: pass
    if total_frames == 0:
        return None
    return total_vmaf / total_frames

# ---------- metrics ----------
def pearson(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    am=a-a.mean(); bm=b-b.mean()
    d=np.linalg.norm(am)*np.linalg.norm(bm)
    return float(am.dot(bm)/d) if d else np.nan

def ccc(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    ma,mb=a.mean(),b.mean(); va,vb=a.var(),b.var()
    r=pearson(a,b); denom=va+vb+(ma-mb)**2
    return (2*r*np.sqrt(va*vb))/denom if denom>0 else np.nan

def _activity_rel(ref, abs_err):
    floor = max(50.0, float(np.percentile(ref, 10)))
    mask = ref >= floor
    if mask.sum() < 50:
        return float("nan"), float("nan")
    rel = abs_err[mask] / np.maximum(ref[mask], 1.0)
    return float(np.median(rel)), float(np.percentile(rel, 95))

# ---------- encoding ----------
def transcode_av1(in_path: Path, out_path: Path, qp: int, preset: int, g: int, la: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if HAS_SVT:
        args = ["-y","-vsync","0","-i",str(in_path),
                "-an","-c:v","libsvtav1","-qp",str(qp),"-preset",str(preset),
                "-g",str(g),"-pix_fmt","yuv420p10le"]
        if HAS_PARAM:
            svt_params=[]
            if HAS_MONO_SVT:
                svt_params.append("monochrome=1")
            svt_params.append("tune=0")
            svt_params.append("scd=0")
            if la>0:
                svt_params.append(f"lad={la}")
            args += ["-svtav1-params", ":".join(svt_params)]
        args = ffmpeg_cmd(args + [str(out_path)])
    elif HAS_AOM:
        args = ["-y","-vsync","0","-i",str(in_path),
                "-an","-c:v","libaom-av1","-crf",str(qp),"-b:v","0",
                "-cpu-used",str(max(4,min(8,preset+2))),"-g",str(g),
                "-row-mt","1","-aq-mode","0"]
        if HAS_MONO_AOM:
            args += ["-monochrome","1"]
        args = ffmpeg_cmd(args + ["-pix_fmt","yuv420p10le", str(out_path)])
    else:
        raise RuntimeError("No AV1 encoder available (svt-a1/libaom-av1).")
    run(args)

# ---------- caching helpers ----------
def cache_path(outdir: Path, tag: str) -> Path:
    return outdir / f"{tag}.npy"

def save_cache(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)

def load_cache(path: Path):
    try: return np.load(path)
    except Exception: return None

# ---------- per-video worker ----------
def _metric_row(name, qp, g, la, preset, base, tst, outp: Path, tau: int,
                encode_sec=None, cd10_sec=None, psnr=None, ssim=None, vmaf=None,
                base_mi=None, tst_mi=None):
    L=min(len(base),len(tst)); ref,y=base[:L],tst[:L]
    r=pearson(ref,y); c=ccc(ref,y)
    rng=(ref.max()-ref.min()); nrmse=float(np.sqrt(np.mean((y-ref)**2))/rng) if rng else 0.0
    abs_err=np.abs(y-ref); med_abs=float(np.median(abs_err)); p95_abs=float(np.percentile(abs_err,95))
    med_rel,p95_rel=_activity_rel(ref,abs_err)
    try:
        w,h=probe_gray_shape(outp) if outp.exists() else (None,None)
        pix=(w*h) if (w and h) else None
    except Exception: pix=None
    p95_pixpct=float(100.0*p95_abs/pix) if pix else float("nan")
    size_mb=outp.stat().st_size/(1024*1024) if outp.exists() else float("nan")
    timing = []
    if encode_sec is not None: timing.append(f"enc={encode_sec:.1f}s")
    if cd10_sec is not None: timing.append(f"cd10={cd10_sec:.1f}s")
    timing_str = ("  " + " ".join(timing)) if timing else ""
    metrics_str = []
    if psnr is not None: metrics_str.append(f"psnr={psnr:.2f}")
    if ssim is not None: metrics_str.append(f"ssim={ssim:.4f}")
    if vmaf is not None: metrics_str.append(f"vmaf={vmaf:.2f}")
    mstr = ("  " + " ".join(metrics_str)) if metrics_str else ""
    print(f"[{name}] qp={qp:2d} g={g:3d} la={la:2d}  r={r:.4f}  ccc={c:.4f}  "
          f"med|Δ|={med_abs:.2f}  p95|Δ|={p95_abs:.2f}  med%~={100*med_rel:.2f}%  "
          f"p95%~={100*p95_rel:.2f}%  p95pix%={p95_pixpct:.4f}%  size={size_mb:.1f}MB"
          f"{mstr}{timing_str}")
    mi_L = None
    if base_mi is not None and tst_mi is not None:
        mi_L = min(len(base_mi), len(tst_mi))
    mi_ref = base_mi[:mi_L] if (mi_L and mi_L>0) else None
    mi_y   = tst_mi[:mi_L] if (mi_L and mi_L>0) else None

    return {"video":name,"qp":qp,"g":g,"la":la,"preset":preset,
            "pearson":r,"ccc":c,"nrmse":nrmse,
            "med_abs":med_abs,"p95_abs":p95_abs,
            "med_rel":med_rel,"p95_rel":p95_rel,
            "p95_pixpct":p95_pixpct,"tau":tau,"size_mb":size_mb,
            "psnr":psnr,"ssim":ssim,"vmaf":vmaf,
            "mi_pearson":pearson(mi_ref,mi_y) if (mi_ref is not None and mi_y is not None) else None,
            "mi_ccc":ccc(mi_ref,mi_y) if (mi_ref is not None and mi_y is not None) else None,
            "mi_med_abs":float(np.median(np.abs(mi_y-mi_ref))) if (mi_ref is not None and mi_y is not None) else None,
            "mi_p95_abs":float(np.percentile(np.abs(mi_y-mi_ref),95)) if (mi_ref is not None and mi_y is not None) else None,
            "encode_sec":encode_sec,"cd10_sec":cd10_sec}

def process_video(args_pack):
    (vid, outroot, tau, qps, gs, las, preset, roi, force, skip_decode,
     stride, threads, windows, metrics, vmaf_model) = args_pack

    name=vid.stem; vdir=outroot/name; vdir.mkdir(parents=True, exist_ok=True)
    roi_tag = f".roi{roi[0]}_{roi[1]}_{roi[2]}_{roi[3]}" if roi else ""
    win_tag = "." + "_".join([f"ss{int(s)}t{int(d)}" for s,d in windows]) if windows else ".full"
    base_tag=f"baseline.tau{tau}{roi_tag}.s{stride}{win_tag}"
    base_cache=cache_path(vdir, base_tag)

    # Baseline
    base=None; base_mi=None; base_time=None
    if not skip_decode:
        base=load_cache(base_cache)
        if base is None:
            t0=time.time()
            base, base_mi = cd10_windows(vid, tau, roi=roi, stride=stride, threads=threads,
                                         windows=windows, calc_mi=("mi" in metrics))
            base_time=time.time()-t0
            save_cache(base_cache, base)
            if base_mi is not None:
                save_cache(vdir/f"{base_tag}.mi.npy", base_mi)
        else:
            if "mi" in metrics:
                base_mi = load_cache(vdir/f"{base_tag}.mi.npy")
    else:
        base=load_cache(base_cache)
        if "mi" in metrics:
            base_mi = load_cache(vdir/f"{base_tag}.mi.npy")
    if base is None or len(base)==0:
        print(f"[{name}] no baseline available; skip."); return []
    if base_time is not None:
        print(f"[{name}] baseline cd10 frames={len(base)}  time={base_time:.1f}s")

    rows=[]
    for qp in qps:
        for g in gs:
            for la in las:
                outp=vdir/f"qp{qp}.g{g}.la{la}.mkv"
                tag=f"qp{qp}.g{g}.la{la}.tau{tau}{roi_tag}.s{stride}{win_tag}"
                tst_cache=cache_path(vdir, tag)
                mi_cache = cache_path(vdir, tag + ".mi")
                encode_sec=None

                if not skip_decode:
                    if outp.exists() and outp.stat().st_size>0 and not force:
                        pass
                    else:
                        if force and outp.exists():
                            try: outp.unlink()
                            except Exception: pass
                        t0=time.time()
                        transcode_av1(vid,outp,qp=qp,preset=preset,g=g,la=la)
                        encode_sec=time.time()-t0

                cd10_sec=None; psnr_val=None; ssim_val=None; vmaf_val=None
                tst=load_cache(tst_cache); tst_mi=None
                if "mi" in metrics:
                    tst_mi = load_cache(mi_cache)
                if tst is None or ("mi" in metrics and tst_mi is None):
                    if not outp.exists():
                        continue
                    t1=time.time()
                    tst, tst_mi = cd10_windows(outp,tau,roi=roi,stride=stride,threads=threads,
                                               windows=windows, calc_mi=("mi" in metrics))
                    cd10_sec=time.time()-t1
                    save_cache(tst_cache,tst)
                    if tst_mi is not None:
                        save_cache(mi_cache, tst_mi)
                if outp.exists():
                    if "psnr" in metrics:
                        psnr_val = psnr_windows(vid, outp, roi=roi, stride=stride, threads=threads, windows=windows)
                    if "ssim" in metrics:
                        ssim_val = ssim_windows(vid, outp, roi=roi, stride=stride, threads=threads, windows=windows)
                    if "vmaf" in metrics and HAS_VMAF:
                        vmaf_val = vmaf_windows(vid, outp, roi=roi, stride=stride, threads=threads, windows=windows, model_path=vmaf_model)

                rows.append(_metric_row(name, qp, g, la, preset, base, tst, outp, tau,
                                        encode_sec=encode_sec, cd10_sec=cd10_sec,
                                        psnr=psnr_val, ssim=ssim_val, vmaf=vmaf_val,
                                        base_mi=base_mi, tst_mi=tst_mi))

    if rows:
        pd.DataFrame(rows).to_csv(vdir/"summary.csv", index=False)
    return rows

# ---------- features-only (flat outdir) ----------
FN_RE = re.compile(r"^(?P<stem>.+)\.qp(?P<qp>\d+)\.g(?P<g>\d+)\.la(?P<la>\d+)\.mkv$")

def features_only(outdir: Path, indir: Path, tau: int, roi, rewrite_summary: bool,
                  stride: int, threads: int, windows, windows_map=None,
                  metrics=(), vmaf_model=None):
    mkvs=sorted([p for p in outdir.glob("*.mkv") if FN_RE.match(p.name)])
    if not mkvs:
        print(f"[features-only] no .mkv in {outdir}"); return []

    groups=defaultdict(list)
    for p in mkvs: groups[FN_RE.match(p.name).group("stem")].append(p)

    original_map={}
    for ext in ("*.mkv","*.mp4","*.mov","*.m4v","*.webm"):
        for p in indir.glob(ext): original_map[p.stem]=p

    summary_csv=outdir/"summary.csv"
    existing=set()
    if summary_csv.exists() and not rewrite_summary:
        with summary_csv.open() as f:
            for row in csv.DictReader(f):
                existing.add((row.get("video"), int(row.get("qp")), int(row.get("g")), int(row.get("la"))))
    else:
        if summary_csv.exists(): summary_csv.unlink()

    all_rows=[]
    for stem, files in groups.items():
        if stem not in original_map:
            print(f"[features-only] missing original for {stem}"); continue
        vid=original_map[stem]; vdir=outdir/stem; vdir.mkdir(parents=True, exist_ok=True)
        roi_tag = f".roi{roi[0]}_{roi[1]}_{roi[2]}_{roi[3]}" if roi else ""
        wins = (windows_map.get(stem) if windows_map else windows)
        win_tag = "." + "_".join([f"ss{int(s)}t{int(d)}" for s,d in wins]) if wins else ".full"
        base_tag=f"baseline.tau{tau}{roi_tag}.s{stride}{win_tag}"
        base_cache=cache_path(vdir, base_tag)
        base_mi_cache = cache_path(vdir, base_tag + ".mi")
        base=load_cache(base_cache); base_mi=None; base_time=None
        if "mi" in metrics:
            base_mi=load_cache(base_mi_cache)
        if base is None:
            t0=time.time()
            base, base_mi=cd10_windows(vid,tau,roi=roi,stride=stride,threads=threads,
                                       windows=wins, calc_mi=("mi" in metrics))
            base_time=time.time()-t0
            save_cache(base_cache, base)
            if base_mi is not None:
                save_cache(base_mi_cache, base_mi)
            print(f"[{stem}] baseline cd10 frames={len(base)}  time={base_time:.1f}s")

        for f in sorted(files):
            m=FN_RE.match(f.name); qp,g,la=int(m.group("qp")),int(m.group("g")),int(m.group("la"))
            if (stem,qp,g,la) in existing: continue
            tag=f"qp{qp}.g{g}.la{la}.tau{tau}{roi_tag}.s{stride}{win_tag}"
            tst_cache=cache_path(vdir, tag)
            mi_cache=cache_path(vdir, tag + ".mi")
            tst=load_cache(tst_cache); tst_mi=None; cd10_sec=None; psnr_val=None; ssim_val=None; vmaf_val=None
            if "mi" in metrics:
                tst_mi=load_cache(mi_cache)
            if tst is None or ("mi" in metrics and tst_mi is None):
                t1=time.time()
                tst, tst_mi=cd10_windows(f,tau,roi=roi,stride=stride,threads=threads,windows=wins, calc_mi=("mi" in metrics))
                cd10_sec=time.time()-t1
                save_cache(tst_cache,tst)
                if tst_mi is not None:
                    save_cache(mi_cache, tst_mi)
            if f.exists():
                if "psnr" in metrics:
                    psnr_val = psnr_windows(vid, f, roi=roi, stride=stride, threads=threads, windows=wins)
                if "ssim" in metrics:
                    ssim_val = ssim_windows(vid, f, roi=roi, stride=stride, threads=threads, windows=wins)
                if "vmaf" in metrics and HAS_VMAF:
                    vmaf_val = vmaf_windows(vid, f, roi=roi, stride=stride, threads=threads, windows=wins, model_path=vmaf_model)
            row=_metric_row(stem, qp, g, la, np.nan, base, tst, f, tau,
                            encode_sec=None, cd10_sec=cd10_sec,
                            psnr=psnr_val, ssim=ssim_val, vmaf=vmaf_val,
                            base_mi=base_mi, tst_mi=tst_mi)
            all_rows.append(row)

    if all_rows:
        mode="a" if summary_csv.exists() and not rewrite_summary else "w"
        pd.DataFrame(all_rows).to_csv(summary_csv, index=False, mode=mode, header=(mode=="w"))
        print(f"[features-only] wrote {len(all_rows)} rows → {summary_csv}")
    else:
        print("[features-only] no new rows.")
    return all_rows

# ---------- main ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--indir", type=Path, required=True)
    ap.add_argument("--pattern", type=str, default="*.mkv")
    ap.add_argument("--out", dest="outdir", type=Path, required=True)
    ap.add_argument("--tau", type=int, default=10)
    ap.add_argument("--qp", type=int, nargs="+", default=[23,27,31,33,35,37,39])
    ap.add_argument("--g", type=int, nargs="+", default=[240])
    ap.add_argument("--la", type=int, nargs="+", default=[0])
    ap.add_argument("--preset", type=int, default=4)
    ap.add_argument("--roi", type=str, default=None, help="x0,y0,x1,y1")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-decode", action="store_true")

    # Scalability knobs
    ap.add_argument("--jobs", type=int, default=max(1, cpu_count()//2),
                    help="Parallel videos (processes).")
    ap.add_argument("--stride", type=int, default=1,
                    help="Every Nth frame decoded.")
    ap.add_argument("--threads", type=int, default=0,
                    help="Decoder threads for ffmpeg (0=auto).")
    ap.add_argument("--sample-secs", type=int, default=60,
                    help="Duration per sampled window (seconds).")
    ap.add_argument("--sample-positions", type=float, nargs="+", default=[0.1, 0.5, 0.9],
                    help="Relative positions in [0,1] to place windows.")
    ap.add_argument("--features-only","--reindex", action="store_true")
    ap.add_argument("--rewrite-summary", action="store_true")
    ap.add_argument("--metrics", type=str, nargs="+", default=[],
                    help="Additional metrics: choose from mi, psnr, ssim, vmaf")
    ap.add_argument("--vmaf-model", type=str, default=None, help="Optional libvmaf model path")
    ap.add_argument("--ffmpeg-prefix", type=str, default=None,
                    help="Command prefix before ffmpeg/ffprobe (e.g., 'apptainer exec ... ffmpeg.sif').")
    ap.add_argument("--ffmpeg-bin", type=str, default="ffmpeg", help="ffmpeg binary name (default: ffmpeg)")
    ap.add_argument("--ffprobe-bin", type=str, default="ffprobe", help="ffprobe binary name (default: ffprobe)")

    args=ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    set_ffmpeg_prefix(args.ffmpeg_prefix, args.ffmpeg_bin, args.ffprobe_bin)
    global HAS_SVT, HAS_AOM, HAS_PARAM, HAS_MONO_SVT, HAS_MONO_AOM, HAS_VMAF
    HAS_SVT   = has_encoder("libsvtav1")
    HAS_AOM   = has_encoder("libaom-av1")
    HAS_PARAM = ffmpeg_supports_svt_params()
    HAS_MONO_SVT = HAS_SVT and encoder_supports_option("libsvtav1", "monochrome")
    HAS_MONO_AOM = HAS_AOM and encoder_supports_option("libaom-av1", "monochrome")
    HAS_VMAF  = has_filter("libvmaf") or has_filter("vmaf")
    metrics = set(m.lower() for m in args.metrics)
    metrics = {m for m in metrics if m in {"mi","psnr","ssim","vmaf"}}
    if "vmaf" in metrics and not HAS_VMAF:
        print("[warn] vmaf requested but libvmaf filter not available; skipping.")
        metrics.discard("vmaf")

    roi=None
    if args.roi:
        x0,y0,x1,y1=map(int,args.roi.split(","))
        roi=(x0,y0,x1,y1)

    # Build per-video windows list (start, dur)
    def windows_for(p: Path):
        try:
            dur=probe_duration(p)
        except Exception:
            return None
        wins=[]
        for frac in args.sample_positions:
            start=max(0.0, min(dur-args.sample_secs, frac*dur))
            if args.sample_secs>0 and start<dur:
                wins.append((start, min(args.sample_secs, dur-start)))
        return wins or None

    if args.features_only:
        # Use any original in indir to compute windows (per-stem)
        # We derive windows per-file to be robust to different durations.
        # features_only() will call cd10_windows with those windows.
        windows_map={}
        for ext in ("*.mkv","*.mp4","*.mov","*.m4v","*.webm"):
            for p in args.indir.glob(ext):
                windows_map[p.stem]=windows_for(p)
        return features_only(args.outdir, args.indir, args.tau, roi, args.rewrite_summary,
                             stride=args.stride, threads=args.threads,
                             windows=None, windows_map=windows_map,
                             metrics=metrics, vmaf_model=args.vmaf_model)

    # Normal path (encode+measure), but parallelize per video
    vids=sorted(args.indir.glob(args.pattern))
    if not vids:
        print(f"No videos in {args.indir} matching {args.pattern}"); return

    print(f"[enc] using: SVT={HAS_SVT}  AOM={HAS_AOM}  svt-params={HAS_PARAM}")
    print(f"[grid] qp={args.qp}  g={args.g}  la={args.la}  preset={args.preset}  stride={args.stride}  jobs={args.jobs}")
    print(f"[sample] {args.sample_secs}s @ positions {args.sample_positions}")

    packs=[]
    for vid in vids:
        wins = windows_for(vid)
        packs.append((vid, args.outdir, args.tau, args.qp, args.g, args.la, args.preset,
                      roi, args.force, args.skip_decode, args.stride, args.threads, wins, metrics, args.vmaf_model))

    all_rows=[]
    if args.jobs==1:
        for p in packs:
            all_rows.extend(process_video(p))
    else:
        with Pool(processes=args.jobs) as pool:
            for rows in pool.imap_unordered(process_video, packs):
                all_rows.extend(rows or [])

    if not all_rows:
        print("No results recorded."); return
    df=pd.DataFrame(all_rows)
    df.to_csv(args.outdir/"summary.csv", index=False)
    print(f"[write] combined summary: {args.outdir/'summary.csv'}")

if __name__=="__main__":
    main()
