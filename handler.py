"""RunPod Serverless handler: GPU-accelerated FFmpeg render & Frame Extraction.

Modes:
  auto_subs      — segment extraction (re-encode) → concat + ASS burn
  subs_only      — single-pass ASS subtitle burn on original video
  extract_frames — extracts 1 frame per second as JPG, zips them, uploads via FTP

Input:
  source_url:      URL for source video (download)
  output_filename: Name of the file to be saved on FTP (e.g., "video.mp4" or "frames.zip")
  mode:            "auto_subs" | "subs_only" | "extract_frames"
  fps:             (extract_frames only) frames per second to extract (default 1)
  segments:        list of {start, end} keep-segments (auto_subs only)
  ass_content:     ASS subtitle content string (for sub modes)
"""
import ftplib
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import requests
import runpod

# ---------------------------------------------------------------------------
# NVENC detection
# ---------------------------------------------------------------------------
def _check_nvenc() -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        return "h264_nvenc" in result.stdout
    except Exception:
        return False

HAS_NVENC = _check_nvenc()

# ---------------------------------------------------------------------------
# FFmpeg Command Builders
# ---------------------------------------------------------------------------
def build_zoom_flags(keep_segments: list[dict]) -> list[bool]:
    if not keep_segments: return []
    flags: list[bool] = []
    zoomed = False
    for idx, seg in enumerate(keep_segments):
        if idx > 0 and (seg["start"] - keep_segments[idx - 1]["end"]) > 0.05:
            zoomed = not zoomed
        flags.append(zoomed)
    return flags

def build_extraction_cmd(input_path: str, out_file: str, start: float, duration: float, zoom: bool, use_nvenc: bool) -> list[str]:
    vf = ("crop=trunc(iw*0.97/2)*2:trunc(ih*0.97/2)*2,scale=ceil(iw/0.97/2)*2:ceil(ih/0.97/2)*2,format=yuv420p" if zoom else "format=yuv420p")
    fade_out_start = max(0, duration - 0.015)
    af = f"afade=t=in:d=0.015,afade=t=out:st={fade_out_start:.3f}:d=0.015"

    if use_nvenc:
        return ["ffmpeg", "-y", "-hwaccel", "cuda", "-ss", f"{start:.3f}", "-i", input_path, "-t", f"{duration:.3f}", "-c:v", "h264_nvenc", "-preset", "p1", "-cq", "18", "-vf", vf, "-af", af, "-c:a", "aac", "-b:a", "128k", "-avoid_negative_ts", "make_zero", out_file]
    return ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", input_path, "-t", f"{duration:.3f}", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-vf", vf, "-af", af, "-c:a", "aac", "-b:a", "128k", "-avoid_negative_ts", "make_zero", out_file]

def build_concat_burn_cmd(list_path: str, ass_path: str, output_path: str, use_nvenc: bool) -> list[str]:
    escaped_ass = ass_path.replace(":", "\\:")
    if use_nvenc:
        return ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-vf", f"ass='{escaped_ass}':fontsdir=/usr/local/share/fonts", "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23", "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", output_path]
    return ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-vf", f"ass='{escaped_ass}':fontsdir=/usr/local/share/fonts", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", output_path]

def build_subs_only_cmd(input_path: str, ass_path: str, output_path: str, use_nvenc: bool) -> list[str]:
    escaped_ass = ass_path.replace(":", "\\:")
    if use_nvenc:
        return ["ffmpeg", "-y", "-hwaccel", "cuda", "-i", input_path, "-vf", f"ass='{escaped_ass}':fontsdir=/usr/local/share/fonts", "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23", "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", output_path]
    return ["ffmpeg", "-y", "-i", input_path, "-vf", f"ass='{escaped_ass}':fontsdir=/usr/local/share/fonts", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", output_path]

def build_extract_frames_cmd(input_path: str, output_pattern: str, fps: int) -> list[str]:
    return [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", f"fps={fps}",
        "-qscale:v", "2", 
        output_pattern
    ]

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def download_file(url: str, dest: str) -> int:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    r = requests.get(url, stream=True, timeout=300, headers=headers)
    r.raise_for_status()
    size = 0
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
            size += len(chunk)
    return size

def upload_to_ftp(file_path: str, ftp_host: str, ftp_user: str, ftp_pass: str, file_name: str) -> None:
    print(f"FTP sunucusuna bağlanılıyor: {ftp_host}...")
    try:
        with ftplib.FTP(ftp_host) as ftp:
            ftp.login(user=ftp_user, passwd=ftp_pass)
            print(f"FTP'ye yükleniyor: {file_name}")
            with open(file_path, "rb") as f:
                ftp.storbinary(f"STOR {file_name}", f)
        print("FTP yüklemesi başarıyla tamamlandı.")
    except Exception as e:
        raise RuntimeError(f"FTP Upload hatası: {e}")

def run_ffmpeg(cmd: list[str], timeout: int = 600) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        lines = result.stderr.splitlines()
        error_lines = [ln for ln in lines if ln.strip() and not ln.strip().startswith("frame=") and "size=" not in ln[:20]]
        err = "\n".join(error_lines[-10:]) if error_lines else result.stderr[-500:]
        raise RuntimeError(f"FFmpeg failed (rc={result.returncode}): {err}")

# ---------------------------------------------------------------------------
# Render pipelines
# ---------------------------------------------------------------------------
def render_extract_frames(input_path: str, tmp_dir: str, fps: int) -> tuple[float, str]:
    t0 = time.monotonic()
    frames_dir = os.path.join(tmp_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    output_pattern = os.path.join(frames_dir, "kare_%04d.jpg")
    cmd = build_extract_frames_cmd(input_path, output_pattern, fps)
    
    print(f"  Extracting frames at {fps} fps...")
    run_ffmpeg(cmd)
    
    print("  Zipping extracted frames...")
    zip_path = os.path.join(tmp_dir, "frames_output.zip")
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', frames_dir)
    
    elapsed = time.monotonic() - t0
    print(f"  Frames extracted and zipped in {elapsed:.1f}s")
    return elapsed, zip_path

def render_auto_subs(input_path: str, segments: list[dict], ass_content: str, output_path: str, tmp_dir: str) -> float:
    t0 = time.monotonic()
    use_nvenc = HAS_NVENC
    zoom_flags = build_zoom_flags(segments)
    seg_files: list[str] = []
    
    for i, seg in enumerate(segments):
        pad_start = max(0, seg["start"] - 0.03)
        pad_end = seg["end"] + 0.03
        duration = pad_end - pad_start
        out_file = os.path.join(tmp_dir, f"seg_{i:03d}.mp4")

        cmd = build_extraction_cmd(input_path, out_file, start=pad_start, duration=duration, zoom=zoom_flags[i], use_nvenc=use_nvenc)
        try:
            run_ffmpeg(cmd)
        except RuntimeError:
            if use_nvenc:
                cmd = build_extraction_cmd(input_path, out_file, start=pad_start, duration=duration, zoom=zoom_flags[i], use_nvenc=False)
                run_ffmpeg(cmd)
            else: raise
        seg_files.append(out_file)

    list_path = os.path.join(tmp_dir, "concat_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for sf in seg_files: f.write(f"file '{sf}'\n")

    ass_path = os.path.join(tmp_dir, "subtitles.ass")
    with open(ass_path, "w", encoding="utf-8") as f: f.write(ass_content)

    t1 = time.monotonic()
    cmd = build_concat_burn_cmd(list_path, ass_path, output_path, use_nvenc)
    try:
        run_ffmpeg(cmd)
    except RuntimeError:
        if use_nvenc:
            cmd = build_concat_burn_cmd(list_path, ass_path, output_path, use_nvenc=False)
            run_ffmpeg(cmd)
        else: raise

    return time.monotonic() - t0

def render_subs_only(input_path: str, ass_content: str, output_path: str, tmp_dir: str) -> float:
    t0 = time.monotonic()
    use_nvenc = HAS_NVENC
    ass_path = os.path.join(tmp_dir, "subtitles.ass")
    with open(ass_path, "w", encoding="utf-8") as f: f.write(ass_content)

    cmd = build_subs_only_cmd(input_path, ass_path, output_path, use_nvenc)
    try:
        run_ffmpeg(cmd)
    except RuntimeError:
        if use_nvenc:
            cmd = build_subs_only_cmd(input_path, ass_path, output_path, use_nvenc=False)
            run_ffmpeg(cmd)
        else: raise
    return time.monotonic() - t0

# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------
def handler(event: dict) -> dict:
    inp = event.get("input", {})
    if inp.get("health_check"): return {"status": "warm"}

    source_url = inp.get("source_url")
    if not source_url: return {"error": "source_url required"}

    mode = inp.get("mode", "subs_only")
    valid_modes = ("auto_subs", "subs_only", "extract_frames")
    if mode not in valid_modes: return {"error": f"invalid mode: {mode}"}

    # FTP Config
    ftp_host = "45.147.46.253"
    ftp_user = "pienart-ftp"
    ftp_pass = os.environ.get("FTP_PASSWORD")
    if not ftp_pass: return {"error": "FTP_PASSWORD ortam değişkeni RunPod'da ayarlanmamış."}

    output_filename = inp.get("output_filename", "processed_video.mp4")

    tmp_dir = tempfile.mkdtemp(prefix="klipza_gpu_")
    input_path = os.path.join(tmp_dir, "input.mp4")
    output_path = os.path.join(tmp_dir, "output.mp4")

    try:
        t0 = time.monotonic()
        video_size = download_file(source_url, input_path)
        dl_time = time.monotonic() - t0
        print(f"Downloaded {video_size / 1024 / 1024:.1f} MB in {dl_time:.1f}s")

        if mode == "auto_subs":
            segments = inp.get("segments")
            if not segments: return {"error": "segments required for auto_subs"}
            render_time = render_auto_subs(input_path, segments, inp.get("ass_content", ""), output_path, tmp_dir)
        elif mode == "subs_only":
            render_time = render_subs_only(input_path, inp.get("ass_content", ""), output_path, tmp_dir)
        elif mode == "extract_frames":
            fps = int(inp.get("fps", 1))
            render_time, zip_output_path = render_extract_frames(input_path, tmp_dir, fps)
            output_path = zip_output_path # Zip dosyasını çıktı olarak ayarla

        output_size = os.path.getsize(output_path)
        print(f"Processed in {render_time:.1f}s, output {output_size / 1024 / 1024:.1f} MB")

        t_up = time.monotonic()
        upload_to_ftp(output_path, ftp_host, ftp_user, ftp_pass, output_filename)
        upload_time = time.monotonic() - t_up
        print(f"Uploaded to FTP in {upload_time:.1f}s")

        return {
            "status": "done",
            "ftp_file": output_filename,
            "mode": mode,
            "gpu": HAS_NVENC,
            "download_time": round(dl_time, 1),
            "render_time": round(render_time, 1),
            "upload_time": round(upload_time, 1),
            "total_time": round(time.monotonic() - t0, 1),
            "input_size_mb": round(video_size / 1024 / 1024, 1),
            "output_size_mb": round(output_size / 1024 / 1024, 1),
        }

    except Exception as e:
        print(f"Handler error: {e}")
        return {"error": str(e)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

runpod.serverless.start({"handler": handler})
