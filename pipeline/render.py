"""Decode with OpenCV, render blurred frames, encode with ffmpeg (audio kept)."""

import os
import subprocess
import tempfile

import cv2


def _has_audio_stream(src_path):
    """True if ffprobe finds an audio stream. Decided up-front so ffmpeg
    never blocks on optional audio mapping for audio-less inputs."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv", src_path],
        capture_output=True, text=True, check=False)
    return proc.returncode == 0 and "stream" in proc.stdout


def render_blurred_video(src_path, active_boxes, dst_tmp_path, fps, width,
                         height, render_cfg, blur_fn, max_frames=0):
    """Write blurred frames to dst_tmp_path, then ffmpeg-encode with audio."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    intermediate = dst_tmp_path + ".silent.mp4"
    writer = cv2.VideoWriter(intermediate, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("could not open intermediate writer")
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        writer.release()
        raise RuntimeError("could not reopen input for render: %s" % src_path)
    rendered = 0
    print("render: starting (%dx%d @ %.2f fps)" % (width, height, fps),
          flush=True)
    try:
        while True:
            if max_frames and rendered >= max_frames:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            writer.write(blur_fn(frame, rendered, active_boxes))
            rendered += 1
            if rendered % 10 == 0:
                print("render: %d frames" % rendered, flush=True)
    finally:
        cap.release()
        writer.release()
    print("render: wrote %d silent frames, invoking ffmpeg" % rendered,
          flush=True)
    has_audio = _has_audio_stream(src_path)
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-nostdin",
        "-i", intermediate,
        "-i", src_path,
        "-map", "0:v:0",
        "-c:v", render_cfg.get("video_codec", "libx264"),
        "-crf", str(render_cfg.get("crf", 20)),
        "-preset", render_cfg.get("preset", "veryfast"),
        "-pix_fmt", render_cfg.get("pix_fmt", "yuv420p"),
        "-frames:v", str(rendered),
    ]
    if has_audio:
        cmd += ["-map", "1:a", "-c:a", "aac"]
    cmd.append(dst_tmp_path)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    print("render: ffmpeg rc=%d tail=%s" % (
        proc.returncode, proc.stderr.strip()[-300:]), flush=True)
    try:
        os.remove(intermediate)
    except OSError:
        pass
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg render failed: %s" % proc.stderr.strip()[-2000:])
    return rendered


def temp_artifact_path(suffix=".mp4"):
    """Create a temp path for an artifact (caller renames atomically)."""
    handle, path = tempfile.mkstemp(suffix=suffix, prefix="anonymise_")
    os.close(handle)
    os.remove(path)
    return path
