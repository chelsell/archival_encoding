#!/usr/bin/env python3
# Build per-video proxies containing only the sampled windows (e.g., 3×10s) for fast AV1 sweeps.
# Usage:
#   python make_window_proxies.py --src /path/to/hevc --dst /tmp/proxies \
#       --sample-secs 10 --sample-positions 0.2 0.5 0.8 --threads 4 \
#       [--ffmpeg-prefix "apptainer exec ... ffmpeg.sif"] [--ffmpeg-bin ffmpeg] [--ffprobe-bin ffprobe]
#
# The output files live in --dst with the same stem, e.g., dst/stem.mkv. They
# are concatenations of the sampled windows. We keep them copy-only unless we
# hit timestamp issues, in which case we fall back to a lossless re-encode.

import argparse, json, subprocess, shlex, tempfile, os
from pathlib import Path

FFMPEG_PREFIX = []
FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"

def set_ffmpeg(prefix, ffmpeg_bin, ffprobe_bin):
    global FFMPEG_PREFIX, FFMPEG_BIN, FFPROBE_BIN
    FFMPEG_PREFIX = shlex.split(prefix) if prefix else []
    FFMPEG_BIN = ffmpeg_bin
    FFPROBE_BIN = ffprobe_bin

def ffmpeg_cmd(args):
    return FFMPEG_PREFIX + [FFMPEG_BIN] + args

def ffprobe_cmd(args):
    return FFMPEG_PREFIX + [FFPROBE_BIN] + args

def probe_duration(path: Path) -> float:
    out = subprocess.check_output(ffprobe_cmd(["-v","error","-of","json","-show_entries","format=duration","-i",str(path)]), text=True)
    j = json.loads(out)
    return float(j["format"]["duration"])

def probe_keyframes(path: Path):
    out = subprocess.check_output(ffprobe_cmd([
        "-v","error","-select_streams","v",
        "-skip_frame","nokey",
        "-show_entries","frame=pkt_pts_time",
        "-of","csv=p=0",
        str(path)
    ]), text=True)
    times=[]
    for line in out.strip().splitlines():
        try: times.append(float(line.strip()))
        except Exception: pass
    return sorted(times)

def validate_proxy(path: Path, threads: int):
    cmd = ffmpeg_cmd(["-hide_banner","-loglevel","error","-xerror"])
    if threads>0: cmd += ["-threads", str(threads)]
    cmd += ["-i", str(path), "-f","null","-"]
    subprocess.run(cmd, check=True)

def recode_proxy(src: Path, dst: Path, windows, threads: int):
    trims=[]; labels=[]
    for idx,(ss,t) in enumerate(windows):
        vlabel=f"v{idx}"
        trims.append(f"[0:v]trim=start={ss}:end={ss+t},setpts=PTS-STARTPTS[{vlabel}]")
        labels.append(f"[{vlabel}]")
    fc = ";".join(trims) + f";{''.join(labels)}concat=n={len(windows)}:v=1:a=0[vout]"
    cmd = ffmpeg_cmd(["-hide_banner","-loglevel","error"])
    if threads>0: cmd += ["-threads", str(threads)]
    cmd += ["-i", str(src),
            "-filter_complex", fc,
            "-map","[vout]",
            "-c:v","libx265","-preset","ultrafast","-x265-params","lossless=1",
            "-y", str(dst)]
    subprocess.run(cmd, check=True)

def build_proxy(src: Path, dst: Path, sample_secs: int, positions, threads: int, snap_keyframe: bool = True):
    dur = probe_duration(src)
    kfs = probe_keyframes(src) if snap_keyframe else []
    windows = []
    for frac in positions:
        start = max(0.0, min(dur - sample_secs, frac * dur))
        if snap_keyframe and kfs:
            # choose nearest keyframe <= start
            ks = [t for t in kfs if t <= start+1e-3]
            if ks: start = ks[-1]
        if sample_secs > 0 and start < dur:
            windows.append((start, min(sample_secs, dur - start)))
    if not windows:
        print(f"[skip] {src} (no windows)"); return
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Try copy-only segments + concat demuxer; if that fails, fall back to lossless x265.
    tmp_dir = Path(tempfile.mkdtemp(prefix="proxy_", dir=str(dst.parent)))
    parts = []
    for idx, (ss, t) in enumerate(windows):
        part = tmp_dir / f"part{idx}.mkv"
        cmd = ffmpeg_cmd(["-hide_banner","-loglevel","error","-threads", str(threads),
                          "-ss", f"{ss}", "-t", f"{t}", "-i", str(src),
                          "-map","0","-copyts","-c","copy","-y", str(part)])
        subprocess.run(cmd, check=True)
        parts.append(part)

    list_path = tmp_dir / "list.txt"
    with list_path.open("w") as f:
        for p in parts:
            f.write(f"file '{p}'\n")

    copy_cmd = ffmpeg_cmd(["-hide_banner","-loglevel","error","-fflags","+genpts",
                           "-f","concat","-safe","0","-i", str(list_path),
                           "-c","copy","-y", str(dst)])
    try:
        subprocess.run(copy_cmd, check=True)
        # validate copy output; if it fails, recode
        try:
            validate_proxy(dst, threads)
            print(f"[ok] {dst} ({len(windows)} windows, copy)")
            status = "copy"
        except subprocess.CalledProcessError:
            recode_proxy(src, dst, windows, threads)
            print(f"[ok] {dst} ({len(windows)} windows, lossless x265 fallback after validation)")
            status = "recode"
    except subprocess.CalledProcessError:
        recode_proxy(src, dst, windows, threads)
        print(f"[ok] {dst} ({len(windows)} windows, lossless x265 fallback)")
        status = "recode"

    for p in parts + [list_path]:
        try: os.remove(p)
        except OSError: pass
    try: os.rmdir(tmp_dir)
    except OSError: pass
    return status, windows, kfs if snap_keyframe else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--pattern", type=str, default="*.mkv")
    ap.add_argument("--sample-secs", type=int, default=10)
    ap.add_argument("--sample-positions", type=float, nargs="+", default=[0.2,0.5,0.8])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--force-recode", action="store_true",
                    help="Force lossless re-encode (skip copy attempts).")
    ap.add_argument("--ffmpeg-prefix", type=str, default=None)
    ap.add_argument("--ffmpeg-bin", type=str, default="ffmpeg")
    ap.add_argument("--ffprobe-bin", type=str, default="ffprobe")
    args = ap.parse_args()

    set_ffmpeg(args.ffmpeg_prefix, args.ffmpeg_bin, args.ffprobe_bin)
    args.dst.mkdir(parents=True, exist_ok=True)

    vids = sorted(args.src.glob(args.pattern))
    if not vids:
        print(f"No inputs matching {args.pattern} in {args.src}")
        return
    manifest = []
    for v in vids:
        out = args.dst / v.name
        if out.exists() and out.stat().st_size > 0:
            manifest.append((v, out, "exists", None, None))
            continue
        try:
            if args.force_recode:
                status, wins, kfs = "recode", None, None
                recode_proxy(v, out, [(max(0.0, min(probe_duration(v)-args.sample_secs, frac*probe_duration(v))), args.sample_secs) for frac in args.sample_positions], args.threads)
                print(f"[ok] {out} (forced lossless x265)")
            else:
                status, wins, kfs = build_proxy(v, out, args.sample_secs, args.sample_positions, args.threads)
            manifest.append((v, out, status, wins, kfs))
        except subprocess.CalledProcessError as e:
            print(f"[fail] {v}: {e}")
            manifest.append((v, out, "fail", None, None))

    # Write a manifest for traceability
    man_path = args.dst / "proxy_manifest.csv"
    with man_path.open("w") as f:
        f.write("src,dst,status,sample_secs,sample_positions,keyframes_used\n")
        for src,dst,status,wins,kfs in manifest:
            kf_used = "snap" if kfs else ""
            f.write(f"{src},{dst},{status},{args.sample_secs},\"{';'.join(map(str,args.sample_positions))}\",{kf_used}\n")
    print(f"[write] manifest -> {man_path}")

if __name__ == "__main__":
    main()
