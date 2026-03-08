import os
import re
import time
import zipfile
from io import BytesIO
from pathlib import Path

import streamlit as st


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


def render_download_list(file_paths, key_prefix: str):
    for idx, path in enumerate(file_paths):
        if not os.path.exists(path):
            continue

        file_name = os.path.basename(path)
        file_size = human_size(os.path.getsize(path))

        col_name, col_size, col_dl = st.columns([6, 2, 2])
        col_name.write(file_name)
        col_size.caption(file_size)

        with open(path, "rb") as f:
            data = f.read()

        col_dl.download_button(
            "Download",
            data=data,
            file_name=file_name,
            mime="video/mp4",
            key=f"{key_prefix}_dl_{idx}",
            use_container_width=True,
        )


def save_uploaded_video(uploaded_file) -> str:
    uploads_dir = Path("user_data") / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    original_name = uploaded_file.name or "uploaded_video.mp4"
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", original_name)
    destination = uploads_dir / f"{int(time.time())}_{safe_name}"

    with open(destination, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(destination)


st.set_page_config(page_title="Clip Generator", layout="wide")
st.title("Clip Generator")

st.markdown(
    "Choose an input source, select how many clips you want, and run the pipeline. "
    "The output uses **clip + subtitles** by default."
)

source_mode = st.radio(
    "Input source",
    options=["YouTube URL", "Upload video file"],
    horizontal=True,
)

url = ""
uploaded_video = None
if source_mode == "YouTube URL":
    url = st.text_input("YouTube video URL", placeholder="https://www.youtube.com/watch?v=...")
    st.caption("If YouTube blocks cloud downloads (HTTP 403), use upload mode.")
else:
    uploaded_video = st.file_uploader(
        "Upload a video file",
        type=["mp4", "mov", "mkv", "webm", "avi"],
        accept_multiple_files=False,
    )

clip_count = st.number_input("How many clips?", min_value=1, max_value=40, value=12, step=1)
run_btn = st.button("Generate Clips", type="primary")

if run_btn:
    if source_mode == "YouTube URL" and (not url or not url.strip()):
        st.error("Please enter a valid URL.")
    elif source_mode == "Upload video file" and uploaded_video is None:
        st.error("Please upload a video file.")
    else:
        local_upload_path = None
        try:
            with st.spinner("Running pipeline. This can take a few minutes..."):
                from automatizador import process_url, process_video_file

                if source_mode == "YouTube URL":
                    result = process_url(url.strip(), clip_count=int(clip_count))
                else:
                    local_upload_path = save_uploaded_video(uploaded_video)
                    result = process_video_file(local_upload_path, clip_count=int(clip_count))

            clips_raw = [p for p in result.get("clips_raw", []) if os.path.exists(p)]
            clips_edited = [p for p in result.get("clips_edited", []) if os.path.exists(p)]

            if not clips_raw and not clips_edited:
                st.error("The pipeline finished without generating clips. Check terminal logs.")
                if source_mode == "YouTube URL":
                    st.info("Tip: if logs show YouTube HTTP 403, switch to 'Upload video file'.")
            else:
                if clips_edited:
                    st.success(f"Generated {len(clips_edited)} edited clips.")

                    zip_buf = build_zip_buffer(clips_edited)
                    st.download_button(
                        "Download All Edited Clips (ZIP)",
                        data=zip_buf,
                        file_name="edited_clips.zip",
                        mime="application/zip",
                        use_container_width=True,
                    )

                    st.subheader("Edited Clips - Downloads")
                    render_download_list(clips_edited, key_prefix="edited")

                    with st.expander("Preview one edited clip"):
                        selected_edited = st.selectbox(
                            "Select a clip",
                            options=clips_edited,
                            format_func=lambda p: os.path.basename(p),
                            key="preview_edited",
                        )
                        st.video(selected_edited)
                else:
                    st.warning("No edited clips were generated.")

                if clips_raw:
                    with st.expander("Raw clips (downloads + preview)"):
                        st.subheader("Raw Clips - Downloads")
                        render_download_list(clips_raw, key_prefix="raw")

                        selected_raw = st.selectbox(
                            "Select a raw clip",
                            options=clips_raw,
                            format_func=lambda p: os.path.basename(p),
                            key="preview_raw",
                        )
                        st.video(selected_raw)

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            if source_mode == "YouTube URL":
                st.info("If logs include YouTube HTTP 403, use 'Upload video file' for a reliable run.")
            import traceback

            traceback.print_exc()
        finally:
            if local_upload_path and os.path.exists(local_upload_path):
                try:
                    os.remove(local_upload_path)
                except Exception:
                    pass
