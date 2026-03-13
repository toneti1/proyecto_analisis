import json
import os
import re
import subprocess
import sys
import time
import uuid
import zipfile
from io import BytesIO
from pathlib import Path

import streamlit as st

PREVIEW_MAX_MB = 120
MAX_SAFE_DOWNLOAD_MB = int(os.getenv("MAX_SAFE_DOWNLOAD_MB", "80"))
JOBS_DIR = Path("user_data") / "jobs"
LOG_TAIL_LINES = 200
AUTO_REFRESH_S = float(os.getenv("AUTO_REFRESH_S", "2.0"))


def human_size(num_bytes: int) -> str:
    size_mb = num_bytes / (1024 * 1024)
    return f"{size_mb:.1f} MB"


def build_local_downloader_zip() -> BytesIO:
    downloader_py = '''#!/usr/bin/env python3
import os
import subprocess
import sys


def main() -> int:
    url = input("Paste the YouTube URL and press Enter: ").strip()
    if not url:
        print("No URL provided.")
        return 1

    print("Installing/updating yt-dlp...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", "yt-dlp"], check=True)

    output_dir = os.path.join(os.path.expanduser("~"), "Downloads")
    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, "podcast_%(title).80s.%(ext)s")

    print("Downloading video...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "yt_dlp",
            "-f",
            "best",
            "--no-playlist",
            url,
            "-o",
            output_template,
        ],
        check=True,
    )

    print("Done. Check your Downloads folder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

    launcher_bat = '''@echo off
setlocal
cd /d %~dp0
py -3 download_podcast.py
if errorlevel 1 (
  echo.
  echo Download failed.
  pause
  exit /b 1
)
echo.
echo Download completed.
pause
'''

    readme_txt = '''Local Downloader (Windows)

1) Extract this ZIP.
2) Double-click run_downloader.bat.
3) Paste the YouTube URL.
4) Wait for completion.
5) Upload the downloaded video file in the web app.
'''

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("download_podcast.py", downloader_py)
        zf.writestr("run_downloader.bat", launcher_bat)
        zf.writestr("README.txt", readme_txt)
    buf.seek(0)
    return buf


def total_size_bytes(file_paths) -> int:
    return sum(os.path.getsize(path) for path in file_paths if os.path.exists(path))


def render_download_section(file_paths, key_prefix: str, title: str):
    existing = [p for p in file_paths if os.path.exists(p)]
    if not existing:
        st.info(f"No files available in {title}.")
        return

    st.subheader(title)
    st.caption(f"{len(existing)} files, total size: {human_size(total_size_bytes(existing))}")

    for path in existing:
        st.write(f"- {os.path.basename(path)} ({human_size(os.path.getsize(path))})")

    selected = st.selectbox(
        "Choose a clip to download",
        options=existing,
        format_func=lambda p: os.path.basename(p),
        key=f"{key_prefix}_selected",
    )

    selected_size_mb = os.path.getsize(selected) / (1024 * 1024)
    if selected_size_mb <= MAX_SAFE_DOWNLOAD_MB:
        with open(selected, "rb") as f:
            st.download_button(
                "Download Selected Clip",
                data=f,
                file_name=os.path.basename(selected),
                mime="video/mp4",
                key=f"{key_prefix}_download_selected",
                use_container_width=True,
            )
    else:
        st.warning(
            f"This clip is {selected_size_mb:.1f} MB. "
            "Download is disabled in cloud mode for large files to prevent app crashes."
        )
        st.caption("Reduce clip size/bitrate or run locally for large-file downloads.")

    if selected_size_mb <= PREVIEW_MAX_MB:
        st.caption("Preview available for this file after download.")
    else:
        st.caption(
            f"Preview disabled for large files ({selected_size_mb:.1f} MB). "
            "Download the clip to view it locally."
        )


def save_uploaded_video(uploaded_file) -> str:
    uploads_dir = Path("user_data") / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    original_name = uploaded_file.name or "uploaded_video.mp4"
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", original_name)
    destination = uploads_dir / f"{int(time.time())}_{safe_name}"

    uploaded_file.seek(0)
    with open(destination, "wb") as f:
        while True:
            chunk = uploaded_file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    uploaded_file.seek(0)

    return str(destination)


def read_job(job_path: Path) -> dict:
    try:
        with open(job_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def write_job(job_path: Path, payload: dict) -> None:
    job_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = job_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    tmp_path.replace(job_path)


def tail_log(log_path: Path, max_lines: int = LOG_TAIL_LINES) -> str:
    if not log_path.exists():
        return ""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return "\n".join(lines)


def render_log_panel(job: dict) -> None:
    log_path = Path(job.get("log_path", "")) if job else Path()
    if not log_path or not log_path.exists():
        st.info("No log output available yet.")
        return
    log_text = tail_log(log_path)
    st.subheader("Console output")
    st.code(log_text or "(no logs yet)", language="text")


def trigger_rerun() -> None:
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


def find_latest_job_path() -> str:
    if not JOBS_DIR.exists():
        return ""
    candidates = sorted(
        JOBS_DIR.glob("*/job.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else ""


def start_pipeline_job(upload_path: str, clip_count: int) -> dict:
    job_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    job_path = job_dir / "job.json"
    log_path = job_dir / "run.log"

    payload = {
        "id": job_id,
        "state": "running",
        "input_path": upload_path,
        "clip_count": int(clip_count),
        "log_path": str(log_path),
        "job_path": str(job_path),
        "started_at": int(time.time()),
    }
    write_job(job_path, payload)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        sys.executable,
        "run_pipeline.py",
        "--input",
        upload_path,
        "--clip-count",
        str(int(clip_count)),
        "--job",
        str(job_path),
    ]

    log_file = open(log_path, "a", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).parent),
            stdout=log_file,
            stderr=log_file,
            env=env,
        )
    finally:
        log_file.flush()
        log_file.close()

    payload["pid"] = proc.pid
    write_job(job_path, payload)
    return payload


st.set_page_config(page_title="Clip Generator", layout="wide")
st.title("Clip Generator")

st.info(
    "Streamlit Cloud cannot reliably download YouTube videos directly. "
    "Shared cloud IPs are frequently blocked by YouTube (HTTP 403 / token checks). "
    "Download the video locally with the tool below, then upload the file here."
)

st.download_button(
    "Download Local Podcast Downloader (Windows ZIP)",
    data=build_local_downloader_zip(),
    file_name="local_podcast_downloader.zip",
    mime="application/zip",
    use_container_width=True,
)

uploaded_video = st.file_uploader(
    "Upload video file",
    type=["mp4", "mov", "mkv", "webm", "avi"],
    accept_multiple_files=False,
)

clip_count = st.number_input("How many clips?", min_value=1, max_value=40, value=12, step=1)
run_btn = st.button("Generate Clips", type="primary")

active_job_path = st.session_state.get("active_job_path")
if not active_job_path:
    latest_job_path = find_latest_job_path()
    if latest_job_path:
        active_job_path = latest_job_path
        st.session_state["active_job_path"] = latest_job_path

job = read_job(Path(active_job_path)) if active_job_path else {}
job_state = job.get("state") if job else None

if run_btn:
    if job_state == "running":
        st.warning("A pipeline job is already running. Please wait for it to finish.")
    elif uploaded_video is None:
        st.error("Please upload a video file.")
    else:
        local_upload_path = save_uploaded_video(uploaded_video)
        job = start_pipeline_job(local_upload_path, clip_count=int(clip_count))
        st.session_state["active_job_path"] = job.get("job_path")
        job_state = "running"

if job_state == "running":
    st.info("Pipeline is running. This can take a while for long videos.")
    if job.get("id"):
        st.caption(f"Job ID: {job.get('id')}")
    st.button("Refresh status")
    render_log_panel(job)
    auto_refresh = st.checkbox("Auto-refresh logs", value=True)
    if auto_refresh:
        st.caption(f"Auto-refresh every {AUTO_REFRESH_S:.0f}s")
        time.sleep(AUTO_REFRESH_S)
        trigger_rerun()
    st.stop()

if job_state == "error":
    st.error(job.get("error", "Pipeline failed. Check logs for details."))
    render_log_panel(job)
    if st.button("Clear job"):
        if "active_job_path" in st.session_state:
            del st.session_state["active_job_path"]
        st.stop()

if job_state == "done":
    result = job.get("result") or {}
    clips_raw = [p for p in result.get("clips_raw", []) if os.path.exists(p)]
    clips_edited = [p for p in result.get("clips_edited", []) if os.path.exists(p)]

    if not clips_raw and not clips_edited:
        st.error("The pipeline finished without generating clips. Check logs.")
        render_log_panel(job)

    if clips_edited:
        st.success(f"Generated {len(clips_edited)} edited clips.")
        st.caption("Bulk ZIP download is disabled in cloud mode for stability. Download clips one by one below.")
        render_download_section(clips_edited, key_prefix="edited", title="Edited Clips")
    else:
        st.warning("No edited clips were generated.")

    if clips_raw:
        with st.expander("Raw clips (downloads only)"):
            st.caption("Raw clips are optional and can be very large.")
            show_raw = st.checkbox(
                "Show raw clip downloads",
                value=False,
                key="show_raw_downloads",
            )
            if show_raw:
                render_download_section(clips_raw, key_prefix="raw", title="Raw Clips")

    with st.expander("Console output (logs)"):
        render_log_panel(job)

    if st.button("Start new job"):
        if "active_job_path" in st.session_state:
            del st.session_state["active_job_path"]
