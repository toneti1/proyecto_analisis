import os
import re
import time
import zipfile
from io import BytesIO
from pathlib import Path

import streamlit as st

ZIP_MAX_TOTAL_MB = 700


def human_size(num_bytes: int) -> str:
    size_mb = num_bytes / (1024 * 1024)
    return f"{size_mb:.1f} MB"


def build_zip_buffer(file_paths):
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in file_paths:
            if os.path.exists(path):
                zf.write(path, os.path.basename(path))
    buf.seek(0)
    return buf


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
        "Choose a clip to download and preview",
        options=existing,
        format_func=lambda p: os.path.basename(p),
        key=f"{key_prefix}_selected",
    )

    with open(selected, "rb") as f:
        st.download_button(
            "Download Selected Clip",
            data=f.read(),
            file_name=os.path.basename(selected),
            mime="video/mp4",
            key=f"{key_prefix}_download_selected",
            use_container_width=True,
        )

    with st.expander("Preview selected clip"):
        st.video(selected)


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

if run_btn:
    if uploaded_video is None:
        st.error("Please upload a video file.")
    else:
        local_upload_path = None
        try:
            with st.spinner("Running pipeline. This can take a few minutes..."):
                from automatizador import process_video_file

                local_upload_path = save_uploaded_video(uploaded_video)
                result = process_video_file(local_upload_path, clip_count=int(clip_count))

            clips_raw = [p for p in result.get("clips_raw", []) if os.path.exists(p)]
            clips_edited = [p for p in result.get("clips_edited", []) if os.path.exists(p)]

            if not clips_raw and not clips_edited:
                st.error("The pipeline finished without generating clips. Check terminal logs.")
            else:
                if clips_edited:
                    st.success(f"Generated {len(clips_edited)} edited clips.")

                    edited_total_mb = total_size_bytes(clips_edited) / (1024 * 1024)
                    if edited_total_mb <= ZIP_MAX_TOTAL_MB:
                        zip_buf = build_zip_buffer(clips_edited)
                        st.download_button(
                            "Download All Edited Clips (ZIP)",
                            data=zip_buf,
                            file_name="edited_clips.zip",
                            mime="application/zip",
                            use_container_width=True,
                        )
                    else:
                        st.warning(
                            "ZIP download is disabled for very large outputs to keep the app stable. "
                            "Download clips one by one below."
                        )

                    render_download_section(clips_edited, key_prefix="edited", title="Edited Clips")
                else:
                    st.warning("No edited clips were generated.")

                if clips_raw:
                    with st.expander("Raw clips (downloads + preview)"):
                        render_download_section(clips_raw, key_prefix="raw", title="Raw Clips")

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            if local_upload_path and os.path.exists(local_upload_path):
                try:
                    os.remove(local_upload_path)
                except Exception:
                    pass

