#!/usr/bin/env python3
"""
from_video_to_frame.py

Usage examples:
# single video
python from_video_to_frame.py --mode video --input /path/wildfire.mp4 --output /path/out_dir

# folder of videos
python from_video_to_frame.py --mode folder --input /path/videos_folder --output /path/out_dir

Options:
  --mode {video,folder}   mode: process a single video file or all videos in a folder
  --input PATH            input file (if mode=video) or input folder (if mode=folder)
  --output PATH           output folder where PNG frames will be saved
  --ext EXT              image extension to save (default: .png)
  --overwrite            overwrite existing frames (default: False -> skip existing)
  --skip_count_pass      if frame count is unknown, do NOT perform a double pass to determine exact zero-padding (default: False -> use double pass)
  --verbose              show extra debug prints
"""
import argparse
from pathlib import Path
import cv2
import math
import sys

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg", ".ts", ".flv"}

def ensure_out_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def gather_videos_in_folder(folder: Path):
    vids = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            vids.append(p)
    return vids

def get_frame_count(cap):
    # Try to read property
    try:
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fc > 0 and fc < 1_000_000_000:
            return fc
    except Exception:
        pass
    return None

def count_frames_one_pass(video_path: Path, verbose=False):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        if verbose:
            print(f"[WARN] cannot open {video_path}")
        return 0
    cnt = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        cnt += 1
    cap.release()
    if verbose:
        print(f"[INFO] counted {cnt} frames in {video_path.name}")
    return cnt

def extract_frames(video_path: Path, out_dir: Path, ext: str = ".png", overwrite=False,
                   skip_count_pass=False, verbose=False):
    """
    Extract frames from single video and save them as out_dir/<stem>_NNN.ext
    Returns number of saved frames.
    """
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    stem = video_path.stem
    ensure_out_dir(out_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = get_frame_count(cap)

    # If frame_count unknown, optionally do a counting pass to determine digits
    if frame_count is None and not skip_count_pass:
        if verbose:
            print(f"[INFO] frame count unknown for {video_path.name}, doing a quick count pass to set zero-padding.")
        cap.release()
        frame_count = count_frames_one_pass(video_path, verbose=verbose)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot re-open video: {video_path}")

    if frame_count is None or frame_count <= 0:
        # fallback to a reasonable default digits (6) if still unknown
        digits = 6
    else:
        digits = max(3, len(str(int(frame_count))))

    # iterate frames
    idx = 0
    saved = 0
    progress_iter = None
    if _HAS_TQDM:
        # if frame_count known set total, else leave unknown
        total = int(frame_count) if frame_count and frame_count > 0 else None
        progress_iter = tqdm(total=total, desc=f"{stem}", unit="frame")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        filename = f"{stem}_{str(idx).zfill(digits)}{ext}"
        out_path = out_dir / filename
        if out_path.exists() and not overwrite:
            # skip
            if verbose:
                print(f"[SKIP] {out_path.name} exists")
        else:
            # write (frame is BGR)
            ok = cv2.imwrite(str(out_path), frame)
            if not ok:
                raise RuntimeError(f"cv2.imwrite failed for {out_path}")
            saved += 1
        if progress_iter is not None:
            progress_iter.update(1)
        else:
            # simple progress print every 100 frames
            if idx % 100 == 0:
                print(f"[{stem}] frame {idx} ...")
    if progress_iter is not None:
        progress_iter.close()
    cap.release()
    if verbose:
        print(f"[INFO] finished {video_path.name}: total frames read={idx}, saved={saved}")
    return saved

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="video", choices=("video","folder"), help="video or folder mode")
    ap.add_argument("--input", default=r"/home/nicola/Immagini/cut out video reg cal/cutout_regcal_26.mp4", help="input video file or input folder")
    ap.add_argument("--output", default=r"/home/nicola/Immagini/Video Frames - Regione Calabria/reg cal 26", help="output folder to store frames")
    ap.add_argument("--ext", default=".png", help="image extension (default .png)")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing frames")
    ap.add_argument("--skip_count_pass", action="store_true", help="skip double-pass counting for zero-padding when frame count is unknown")
    ap.add_argument("--verbose", action="store_true", help="verbose prints")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    ensure_out_dir(out)

    if args.mode == "video":
        if not inp.exists():
            print("[ERROR] input video not found:", inp)
            sys.exit(1)
        # create subdir in output for this video? user asked to save in output folder directly
        # We'll save directly into out_folder with filenames like stem_001.png
        extract_frames(inp, out, ext=args.ext if args.ext.startswith(".") else "."+args.ext,
                       overwrite=args.overwrite, skip_count_pass=args.skip_count_pass, verbose=args.verbose)
    else:
        # folder mode: iterate videos in folder (non recursive)
        if not inp.exists() or not inp.is_dir():
            print("[ERROR] input folder not found or not a directory:", inp)
            sys.exit(1)
        vids = gather_videos_in_folder(inp)
        if len(vids) == 0:
            print("[WARN] no video files found in folder:", inp)
            sys.exit(0)
        for v in vids:
            if args.verbose:
                print(f"[INFO] processing {v.name} ...")
            extract_frames(v, out, ext=args.ext if args.ext.startswith(".") else "."+args.ext,
                           overwrite=args.overwrite, skip_count_pass=args.skip_count_pass, verbose=args.verbose)

if __name__ == "__main__":
    main()
