import os
import sys
import cv2
import numpy as np
try:
    from insightface.app import FaceAnalysis
except Exception:
    FaceAnalysis = None
import ffmpeg
from pathlib import Path
from multiprocessing import set_start_method
import traceback
from collections import deque
import subprocess
import random
import re
import json
from fractions import Fraction
from typing import Optional
import shutil
import tempfile

try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int, min_value: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(min_value, int(raw))
    except ValueError:
        return default


CARPETA_BASE = Path(__file__).parent
CARPETA_ENTRADA = CARPETA_BASE / "clips"
CARPETA_SALIDA = CARPETA_BASE / "clips_editados"
RUTA_FUENTE = CARPETA_BASE / "Montserrat-Bold.ttf"
RUTA_GAMEPLAY = CARPETA_BASE / ("gameplay1.mp4" if random.randint(0, 1) == 0 else "gameplay.mp4")
RUTA_MUSICA = CARPETA_BASE / "musica.mp3"

MAX_CLIPS_TO_EDIT = _env_int("MAX_CLIPS_TO_EDIT", 12, min_value=1)
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")
ENABLE_FACE_TRACKING = _env_flag("ENABLE_FACE_TRACKING", False)
USE_GEMINI_METADATA = _env_flag("USE_GEMINI_METADATA", False)
ADD_BACKGROUND_MUSIC = _env_flag("ADD_BACKGROUND_MUSIC", True)
SKIP_EXISTING_OUTPUTS = _env_flag("SKIP_EXISTING_OUTPUTS", True)
INTERVALO_DETECCION = _env_int("INTERVALO_DETECCION", 30, min_value=1)
OUTPUT_LAYOUT = os.getenv("OUTPUT_LAYOUT", "clip_only").strip().lower()
if OUTPUT_LAYOUT not in {"clip_only", "stacked"}:
    print(f"OUTPUT_LAYOUT invalido: {OUTPUT_LAYOUT}. Se usara clip_only.")
    OUTPUT_LAYOUT = "clip_only"


def resolve_ffmpeg_bin() -> str:
    custom = os.getenv("FFMPEG_BIN", "").strip()
    if custom:
        return custom
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    return "ffmpeg"


FFMPEG_BIN = resolve_ffmpeg_bin()


def ensure_ffmpeg_in_path(ffmpeg_bin: str) -> None:
    ffmpeg_path = Path(ffmpeg_bin)
    if not ffmpeg_path.exists():
        return

    current_path = os.environ.get("PATH", "")
    path_items = current_path.split(os.pathsep) if current_path else []

    def prepend(entry: str) -> None:
        if entry and entry not in path_items:
            path_items.insert(0, entry)

    prepend(str(ffmpeg_path.parent))

    expected_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    if ffmpeg_path.name != expected_name:
        shim_dir = Path(tempfile.gettempdir()) / "ffmpeg_shim"
        shim_dir.mkdir(parents=True, exist_ok=True)
        shim_path = shim_dir / expected_name

        if not shim_path.exists():
            try:
                if os.name == "nt":
                    shutil.copy2(ffmpeg_path, shim_path)
                else:
                    os.symlink(ffmpeg_path, shim_path)
            except Exception:
                try:
                    shutil.copy2(ffmpeg_path, shim_path)
                except Exception:
                    pass

        prepend(str(shim_dir))

    os.environ["PATH"] = os.pathsep.join(path_items)


ensure_ffmpeg_in_path(FFMPEG_BIN)


def ffmpeg_has_drawtext() -> bool:
    try:
        result = subprocess.run(
            [FFMPEG_BIN, "-hide_banner", "-filters"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            return False
        return "drawtext" in result.stdout
    except Exception:
        return False


HAS_DRAWTEXT = ffmpeg_has_drawtext()
if not HAS_DRAWTEXT:
    print("Aviso: ffmpeg no tiene filtro drawtext; usando fallback OpenCV para subtitulos.")

ANCHO_SALIDA = 1080
ALTO_SALIDA = 1920
TAMANO_FUENTE_DRAWTEXT = 70
PALABRAS_POR_SUBTITULO = _env_int("PALABRAS_POR_SUBTITULO", 3, min_value=1)
FACTOR_SUAVIZADO_CARA = 25

CENSURA_DICT = {
    "fuck": "f*ck",
    "shit": "sh*t",
    "bitch": "b*tch",
    "asshole": "a**hole",
    "dick": "d*ck",
    "pussy": "p*ssy",
    "cock": "c*ck",
    "cum": "c*m",
    "boobs": "b**bs",
    "sex": "s*x",
    "sexy": "s*xy",
    "faggot": "f*ggot",
    "bastard": "b*stard",
    "slut": "sl*t",
    "whore": "wh*re",
    "balls": "b*lls",
    "dildo": "d*ldo",
    "damn": "d*mn",
    "hell": "h*ll",
    "penis": "p*nis",
    "vagina": "v*gina",
    "orgasm": "org*sm",
    "suck": "s*ck",
    "jerk": "j*rk",
    "kill": "k*ll",
    "murder": "m*rder",
    "suicide": "s*icide",
    "die": "d*e",
    "dead": "d*ad",
    "rape": "r*pe",
    "gun": "g*n",
    "shot": "sh*t",
    "bomb": "b*mb",
    "terror": "t*rror",
    "hitler": "h*tler",
    "nazi": "n*zi",
    "drugs": "dr*gs",
    "weed": "w**d",
    "cocaine": "c*caine",
    "heroin": "h*roin",
    "meth": "m*th",
    "alcohol": "alc*hol",
    "vodka": "v*dka",
    "beer": "b*er",
    "smoke": "sm*ke",
}
CENSURA_REGEX = re.compile(r"\b(" + "|".join(re.escape(k) for k in CENSURA_DICT.keys()) + r")\b", flags=re.IGNORECASE)


# -----------------------------------------------------------------------------
# Model init
# -----------------------------------------------------------------------------
print("Inicializando modelos y configuracion...")

providers = ["CPUExecutionProvider"]
# Cloud-first default: avoid importing torch/cuda just to detect GPU.
DEVICE = "cpu"
FP16 = False
VIDEO_CODEC = "libx264"
print("Usando cpu y libx264 (modo estable para cloud).")

MODELO_WHISPER = None
FACE_APP = None
MODELO_GEMINI = None
FACE_TRACKING_ACTIVE = ENABLE_FACE_TRACKING and FaceAnalysis is not None
if ENABLE_FACE_TRACKING and FaceAnalysis is None:
    print("InsightFace no disponible; desactivando face tracking.")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
if USE_GEMINI_METADATA and genai is not None and GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        MODELO_GEMINI = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
    except Exception as e:
        print(f"No se pudo inicializar Gemini: {e}. Se usaran metadatos locales.")
        MODELO_GEMINI = None
elif USE_GEMINI_METADATA:
    print("Gemini desactivado por falta de libreria o GOOGLE_API_KEY.")

try:
    if not RUTA_FUENTE.exists():
        raise FileNotFoundError(f"No se encuentra la fuente en: {RUTA_FUENTE}")

    if FACE_TRACKING_ACTIVE:
        FACE_APP = FaceAnalysis(name="buffalo_l", providers=providers)
        FACE_APP.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_size=(640, 640))
except Exception as e:
    print(f"Error fatal al cargar modelos/fuente: {e}")
    raise

print("Configuracion base cargada y lista.")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def censurar_texto(texto: str) -> str:
    def reemplazo(match):
        palabra = match.group(0)
        censurada = CENSURA_DICT.get(palabra.lower(), palabra)
        if palabra and palabra[0].isupper():
            censurada = censurada.capitalize()
        return censurada

    return CENSURA_REGEX.sub(reemplazo, texto or "")


def escape_text_for_drawtext(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("\\", "\\\\")
    text = text.replace("%", "\\%").replace(":", "\\:")
    text = text.replace("\n", " ")
    return text


def encontrar_rostro_principal_insightface(frame):
    if FACE_APP is None:
        return None
    faces = FACE_APP.get(frame)
    if not faces:
        return None
    best_face = max(faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
    x1, y1, x2, y2 = [int(coord) for coord in best_face.bbox]
    return (x1, y1, x2 - x1, y2 - y1)


def get_whisper_model():
    global MODELO_WHISPER
    if MODELO_WHISPER is None:
        print(f"Cargando Whisper ({WHISPER_MODEL_SIZE})...")
        try:
            import whisper as whisper_mod
        except Exception as e:
            raise RuntimeError(f"Whisper no disponible: {e}")
        MODELO_WHISPER = whisper_mod.load_model(WHISPER_MODEL_SIZE, device=DEVICE)
    return MODELO_WHISPER


def cargar_segmentos_precalculados(ruta_video: Path) -> list:
    sidecar_path = ruta_video.with_suffix(".json")
    if not sidecar_path.exists():
        return []

    try:
        with open(sidecar_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        segmentos = data.get("segments", []) if isinstance(data, dict) else []
        if isinstance(segmentos, list) and segmentos:
            print(f"[{ruta_video.name}] Usando subtitulos precalculados ({sidecar_path.name}).")
            return segmentos
    except Exception as e:
        print(f"[{ruta_video.name}] No se pudo leer sidecar de subtitulos: {e}")

    return []


def transcribir_audio(ruta_video: Path) -> list:
    precalc = cargar_segmentos_precalculados(ruta_video)
    if precalc:
        return precalc

    print(f"[{ruta_video.name}] Transcribiendo audio...")
    try:
        modelo = get_whisper_model()
        resultado = modelo.transcribe(str(ruta_video), word_timestamps=True, fp16=FP16)
        return resultado.get("segments", [])
    except Exception as e:
        print(f"Error en transcripcion de {ruta_video.name}: {e}")
        return []


def clip_has_audio(ruta_clip: Path) -> bool:
    try:
        probe = ffmpeg.probe(str(ruta_clip))
        return any(s.get("codec_type") == "audio" for s in probe.get("streams", []))
    except Exception:
        return False


def build_word_cues(segmentos) -> list:
    palabras = [p for s in segmentos for p in s.get("words", [])]
    cues = []
    for i in range(0, len(palabras), PALABRAS_POR_SUBTITULO):
        grupo = palabras[i:i + PALABRAS_POR_SUBTITULO]
        if not grupo:
            continue

        start = grupo[0].get("start")
        end = grupo[-1].get("end")
        if start is None or end is None:
            continue

        texto = " ".join(p.get("word", "") for p in grupo).strip().upper()
        texto = censurar_texto(texto)
        if not texto:
            continue
        cues.append((float(start), float(end), texto))

    if cues:
        return cues

    # Fallback when sidecar/Whisper has segment text but no per-word timestamps.
    for seg in segmentos or []:
        start = seg.get("start")
        end = seg.get("end")
        text = str(seg.get("text", "")).strip().upper()
        text = censurar_texto(text)
        if start is None or end is None or not text:
            continue
        cues.append((float(start), float(end), text))

    return cues


def fit_frame_to_canvas(frame, out_w: int, out_h: int):
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    scale = min(out_w / float(w), out_h / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x = (out_w - new_w) // 2
    y = (out_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def draw_subtitle_cv2(frame, text: str):
    if not text:
        return frame

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.7
    thickness = 3
    border = 7
    (text_w, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = max(30, int((ANCHO_SALIDA - text_w) / 2))
    y = ALTO_SALIDA - 140

    cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), border, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def render_clip_only_layout_cv2_fallback(ruta_clip: Path, ruta_salida_video: Path, segmentos, nombre_clip_log: str):
    print(f"[{nombre_clip_log}] Activando fallback OpenCV para render.")
    temp_video = None

    cap_probe = cv2.VideoCapture(str(ruta_clip))
    if not cap_probe.isOpened():
        raise RuntimeError(f"No se pudo abrir clip para fallback: {ruta_clip}")

    fps = cap_probe.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    cap_probe.release()

    cues = build_word_cues(segmentos)

    def render_frames(cap, write_frame):
        cue_idx = 0
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not t or t <= 0:
                t = frame_idx / fps
            while cue_idx < len(cues) and t > cues[cue_idx][1]:
                cue_idx += 1

            frame_out = fit_frame_to_canvas(frame, ANCHO_SALIDA, ALTO_SALIDA)
            if cue_idx < len(cues):
                start, end, text = cues[cue_idx]
                if start <= t <= end:
                    frame_out = draw_subtitle_cv2(frame_out, text)

            write_frame(frame_out)
            frame_idx += 1
        return frame_idx

    def try_ffmpeg_raw_encode(codec_name: str) -> Optional[Path]:
        temp_path = ruta_salida_video.with_suffix(".noaudio.mp4")
        cmd = [
            FFMPEG_BIN,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{ANCHO_SALIDA}x{ALTO_SALIDA}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            codec_name,
            "-pix_fmt",
            "yuv420p",
            str(temp_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        cap = cv2.VideoCapture(str(ruta_clip))
        if not cap.isOpened():
            proc.kill()
            return None

        frame_idx = render_frames(cap, lambda fr: proc.stdin.write(fr.tobytes()))
        cap.release()
        try:
            proc.stdin.close()
        except Exception:
            pass
        _, stderr_output = proc.communicate()

        if frame_idx == 0 or proc.returncode != 0:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            if stderr_output:
                err_text = stderr_output.decode(errors="ignore").strip()
                tail = "\n".join(err_text.splitlines()[-6:])
                print(f"[{nombre_clip_log}] ffmpeg rawvideo stderr ({codec_name}):\n{tail}")
            return None

        return temp_path

    for codec_name in ("libx264", "mpeg4"):
        temp_video = try_ffmpeg_raw_encode(codec_name)
        if temp_video is not None:
            break

    if temp_video is None:
        codec_candidates = [
            ("mp4v", ".noaudio.mp4"),
            ("MJPG", ".noaudio.avi"),
            ("XVID", ".noaudio.avi"),
        ]
        writer = None
        for fourcc, suffix in codec_candidates:
            temp_video = ruta_salida_video.with_suffix(suffix)
            writer = cv2.VideoWriter(
                str(temp_video),
                cv2.VideoWriter_fourcc(*fourcc),
                fps,
                (ANCHO_SALIDA, ALTO_SALIDA),
            )
            if writer.isOpened():
                break
            writer.release()
            writer = None

        if writer is None or not writer.isOpened():
            raise RuntimeError("No se pudo iniciar VideoWriter fallback (mp4v/mjpg/xvid).")

        cap = cv2.VideoCapture(str(ruta_clip))
        if not cap.isOpened():
            writer.release()
            raise RuntimeError(f"No se pudo abrir clip para fallback: {ruta_clip}")

        frame_idx = render_frames(cap, writer.write)
        cap.release()
        writer.release()

        if frame_idx == 0:
            if temp_video and temp_video.exists():
                temp_video.unlink(missing_ok=True)
            raise RuntimeError("Fallback OpenCV no produjo frames.")

    if clip_has_audio(ruta_clip):
        try:
            subprocess.run(
                [
                    FFMPEG_BIN,
                    "-y",
                    "-i",
                    str(temp_video),
                    "-i",
                    str(ruta_clip),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0?",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    str(ruta_salida_video),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            temp_video.unlink(missing_ok=True)
            return
        except Exception as e:
            print(f"[{nombre_clip_log}] No se pudo mezclar audio en fallback: {e}")

    def finalize_temp_video(temp_path: Path) -> None:
        if temp_path.suffix.lower() == ".mp4":
            os.replace(str(temp_path), str(ruta_salida_video))
            return

        for codec_name in ("libx264", "mpeg4"):
            try:
                subprocess.run(
                    [
                        FFMPEG_BIN,
                        "-y",
                        "-i",
                        str(temp_path),
                        "-an",
                        "-c:v",
                        codec_name,
                        "-pix_fmt",
                        "yuv420p",
                        str(ruta_salida_video),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                temp_path.unlink(missing_ok=True)
                return
            except Exception as e:
                print(f"[{nombre_clip_log}] Fallo transcode fallback {codec_name}: {e}")

        print(f"[{nombre_clip_log}] Aviso: dejando contenedor original ({temp_path.suffix}).")
        os.replace(str(temp_path), str(ruta_salida_video))

    if temp_video is not None:
        finalize_temp_video(temp_video)


def obtener_posiciones_suavizadas_cara(ruta_video: Path, w: int, h: int) -> list:
    nombre_clip = ruta_video.name
    print(f"[{nombre_clip}] Analizando posiciones de cara...")
    posiciones_raw = {}

    cap = cv2.VideoCapture(str(ruta_video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    for frame_count in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % INTERVALO_DETECCION == 0:
            rostro = encontrar_rostro_principal_insightface(frame)
            if rostro:
                x, y, rw, rh = rostro
                posiciones_raw[frame_count] = (x + rw / 2, y + rh / 2)

    cap.release()

    if total_frames <= 0:
        return []

    if not posiciones_raw:
        print(f"[{nombre_clip}] No se detectaron caras. Se centrara el video.")
        return [(w / 2, h / 2)] * total_frames

    known_idx = np.array(sorted(posiciones_raw.keys()), dtype=np.int32)
    known_x = np.array([posiciones_raw[i][0] for i in known_idx], dtype=np.float32)
    known_y = np.array([posiciones_raw[i][1] for i in known_idx], dtype=np.float32)
    frame_idx = np.arange(total_frames, dtype=np.float32)

    interp_x = np.interp(frame_idx, known_idx, known_x)
    interp_y = np.interp(frame_idx, known_idx, known_y)

    historial = deque(maxlen=FACTOR_SUAVIZADO_CARA)
    posiciones_suavizadas = []
    for x, y in zip(interp_x, interp_y):
        historial.append((x, y))
        avg_x = float(np.mean([p[0] for p in historial]))
        avg_y = float(np.mean([p[1] for p in historial]))
        posiciones_suavizadas.append((avg_x, avg_y))

    return posiciones_suavizadas


def generar_metadatos_locales(transcripcion_completa: str) -> dict:
    tokens = re.findall(r"[A-Za-z0-9']+", transcripcion_completa or "")
    title_tokens = tokens[:10]
    desc_tokens = tokens[:30]

    title = " ".join(title_tokens).strip().title() if title_tokens else "Clip Interesante"
    description_body = " ".join(desc_tokens).strip() if desc_tokens else "Clip generated automatically."
    description = f"{description_body}\n#shorts #viral #clip"
    return {"title": title, "description": description}


def generar_metadatos(transcripcion_completa: str, nombre_clip: str) -> dict:
    if not transcripcion_completa.strip():
        print(f"[{nombre_clip}] Transcripcion vacia. Metadatos por defecto.")
        return {"title": "Video Interesante", "description": "#video #clip #interesante"}

    if MODELO_GEMINI is None:
        return generar_metadatos_locales(transcripcion_completa)

    print(f"[{nombre_clip}] Generando titulo y descripcion con Gemini...")
    prompt = f"""
You are an expert social media assistant specialized in creating viral content for short videos.
Return valid JSON with keys: title, description.
Use 10-12 words for title and include 3-5 hashtags in description.
Transcription:
{transcripcion_completa}
"""
    try:
        respuesta = MODELO_GEMINI.generate_content(prompt)
        json_text = (respuesta.text or "").strip().replace("```json", "").replace("```", "").strip()
        metadata = json.loads(json_text)
        if not isinstance(metadata, dict) or "title" not in metadata or "description" not in metadata:
            return generar_metadatos_locales(transcripcion_completa)
        return metadata
    except Exception as e:
        print(f"[{nombre_clip}] Error al generar metadatos con Gemini: {e}")
        return generar_metadatos_locales(transcripcion_completa)


def crear_top_video_centrado(ruta_clip: Path, ruta_top_video_temp: Path):
    filtro = f"scale={ANCHO_SALIDA}:{ALTO_SALIDA // 2}:force_original_aspect_ratio=increase,crop={ANCHO_SALIDA}:{ALTO_SALIDA // 2}"
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        str(ruta_clip),
        "-vf",
        filtro,
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        VIDEO_CODEC,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(ruta_top_video_temp),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def crear_top_video_tracking(ruta_clip: Path, ruta_top_video_temp: Path, w: int, h: int, fps: float):
    posiciones = obtener_posiciones_suavizadas_cara(ruta_clip, w, h)
    ancho_recorte = h * (ANCHO_SALIDA / (ALTO_SALIDA / 2))

    command_top = [
        FFMPEG_BIN,
        "-y",
        "-f",
        "rawvideo",
        "-video_size",
        f"{ANCHO_SALIDA}x{ALTO_SALIDA // 2}",
        "-pix_fmt",
        "bgr24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-i",
        str(ruta_clip),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        VIDEO_CODEC,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(ruta_top_video_temp),
    ]

    process_top = subprocess.Popen(command_top, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    cap = cv2.VideoCapture(str(ruta_clip))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(posiciones):
            break

        centro_x, _ = posiciones[frame_idx]
        x1 = int(max(0, centro_x - ancho_recorte / 2))
        x2 = int(min(w, x1 + ancho_recorte))
        frame_recortado = frame[:, x1:x2]
        frame_final_top = cv2.resize(frame_recortado, (ANCHO_SALIDA, ALTO_SALIDA // 2))
        process_top.stdin.write(frame_final_top.tobytes())
        frame_idx += 1

    cap.release()
    _, stderr_output = process_top.communicate()
    if process_top.returncode != 0:
        raise RuntimeError(f"Error en FFmpeg (video superior): {stderr_output.decode(errors='ignore')}")


def mezclar_musica_en_video(video_path: Path, musica_path: Path, vol_db: float = 8.0):
    if not musica_path.exists():
        print(f"No se encontro la musica en {musica_path}. Omitiendo mezcla.")
        return True

    try:
        salida_temp = video_path.with_name(video_path.stem + "_mixtemp.mp4")
        vol_str = f"+{vol_db}dB" if vol_db >= 0 else f"{vol_db}dB"
        cmd = [
            FFMPEG_BIN,
            "-y",
            "-i",
            video_path.as_posix(),
            "-stream_loop",
            "-1",
            "-ss",
            "5",
            "-i",
            musica_path.as_posix(),
            "-filter_complex",
            f"[1:a]volume={vol_str}[m];[0:a][m]amix=inputs=2:duration=first:dropout_transition=2[aout]",
            "-map",
            "0:v",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            salida_temp.as_posix(),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.replace(salida_temp.as_posix(), video_path.as_posix())
        return True
    except Exception as e:
        print(f"Error al mezclar musica: {e}")
        return False


def aplicar_subtitulos_drawtext(video_stream, segmentos, y_offset_px: int):
    if not segmentos:
        return video_stream

    ruta_fuente_posix = Path(RUTA_FUENTE).resolve().as_posix()
    cues = build_word_cues(segmentos)
    subs_y_value = str(ALTO_SALIDA - TAMANO_FUENTE_DRAWTEXT - y_offset_px)

    for start, end, text in cues:
        texto_esc = escape_text_for_drawtext(text)

        video_stream = video_stream.drawtext(
            text=texto_esc,
            fontfile=ruta_fuente_posix,
            fontsize=TAMANO_FUENTE_DRAWTEXT,
            fontcolor="white",
            bordercolor="black",
            borderw=1,
            x="(w-text_w)/2",
            y=subs_y_value,
            enable=f"between(t,{start:.3f},{end:.3f})",
        )

    return video_stream


def render_clip_only_layout(ruta_clip: Path, ruta_salida_video: Path, segmentos, nombre_clip_log: str):
    print(f"[{nombre_clip_log}] Render layout: clip_only")

    if not HAS_DRAWTEXT:
        try:
            render_clip_only_layout_cv2_fallback(ruta_clip, ruta_salida_video, segmentos, nombre_clip_log)
            print(f"[{nombre_clip_log}] Render fallback completado.")
            return
        except Exception as fallback_error:
            print(f"[{nombre_clip_log}] Fallo fallback OpenCV: {fallback_error}")
            raise

    input_clip = ffmpeg.input(str(ruta_clip))
    video_stream = (
        input_clip.video
        .filter("scale", ANCHO_SALIDA, ALTO_SALIDA, force_original_aspect_ratio="decrease")
        .filter("pad", ANCHO_SALIDA, ALTO_SALIDA, "(ow-iw)/2", "(oh-ih)/2", color="black")
    )

    video_stream = aplicar_subtitulos_drawtext(video_stream, segmentos, y_offset_px=220)

    has_audio = clip_has_audio(ruta_clip)

    codec_candidates = [VIDEO_CODEC]
    if VIDEO_CODEC != "mpeg4":
        codec_candidates.append("mpeg4")

    last_error = None
    for codec_name in codec_candidates:
        try:
            output_kwargs = {"c:v": codec_name, "pix_fmt": "yuv420p"}
            if has_audio:
                output_kwargs.update({"c:a": "aac", "b:a": "192k"})
                ffmpeg.output(video_stream, input_clip.audio, str(ruta_salida_video), **output_kwargs).run(
                    overwrite_output=True,
                    capture_stdout=True,
                    capture_stderr=True,
                )
            else:
                ffmpeg.output(video_stream, str(ruta_salida_video), **output_kwargs).run(
                    overwrite_output=True,
                    capture_stdout=True,
                    capture_stderr=True,
                )
            return
        except Exception as e:
            last_error = e
            err_text = ""
            if hasattr(e, "stderr") and e.stderr:
                try:
                    err_text = e.stderr.decode(errors="ignore").strip()
                except Exception:
                    err_text = str(e.stderr)
            if err_text:
                tail = "\n".join(err_text.splitlines()[-6:])
                print(f"[{nombre_clip_log}] ffmpeg stderr ({codec_name}):\n{tail}")
            print(f"[{nombre_clip_log}] Fallo render con codec {codec_name}: {e}")

    try:
        render_clip_only_layout_cv2_fallback(ruta_clip, ruta_salida_video, segmentos, nombre_clip_log)
        print(f"[{nombre_clip_log}] Render fallback completado.")
        return
    except Exception as fallback_error:
        print(f"[{nombre_clip_log}] Fallo fallback OpenCV: {fallback_error}")

    if last_error is not None:
        raise last_error


def enforce_vertical_with_audio(ruta_salida_video: Path, ruta_clip: Path, nombre_clip_log: str):
    temp_final = ruta_salida_video.with_suffix(".final.mp4")
    vf_expr = (
        f"scale={ANCHO_SALIDA}:{ALTO_SALIDA}:force_original_aspect_ratio=decrease,"
        f"pad={ANCHO_SALIDA}:{ALTO_SALIDA}:(ow-iw)/2:(oh-ih)/2:color=black"
    )

    codec_candidates = ["libx264", "mpeg4"]
    last_error = None

    for codec_name in codec_candidates:
        try:
            cmd = [
                FFMPEG_BIN,
                "-y",
                "-i",
                str(ruta_salida_video),
                "-i",
                str(ruta_clip),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-vf",
                vf_expr,
                "-c:v",
                codec_name,
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                str(temp_final),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.replace(str(temp_final), str(ruta_salida_video))
            return
        except Exception as e:
            last_error = e
            print(f"[{nombre_clip_log}] Fallo normalizacion final con {codec_name}: {e}")

    if temp_final.exists():
        temp_final.unlink(missing_ok=True)

    if last_error is not None:
        raise last_error


def ensure_vertical_output(ruta_salida_video: Path, nombre_clip_log: str) -> None:
    try:
        probe = ffmpeg.probe(str(ruta_salida_video))
        stream = next(s for s in probe.get("streams", []) if s.get("codec_type") == "video")
        w = int(stream.get("width", 0) or 0)
        h = int(stream.get("height", 0) or 0)
    except Exception as e:
        print(f"[{nombre_clip_log}] No se pudo verificar dimensiones finales: {e}")
        return

    if w == ANCHO_SALIDA and h == ALTO_SALIDA:
        return

    temp_final = ruta_salida_video.with_suffix(".vert.mp4")
    vf_expr = (
        f"scale={ANCHO_SALIDA}:{ALTO_SALIDA}:force_original_aspect_ratio=decrease,"
        f"pad={ANCHO_SALIDA}:{ALTO_SALIDA}:(ow-iw)/2:(oh-ih)/2:color=black"
    )

    for codec_name in ("libx264", "mpeg4"):
        try:
            subprocess.run(
                [
                    FFMPEG_BIN,
                    "-y",
                    "-i",
                    str(ruta_salida_video),
                    "-vf",
                    vf_expr,
                    "-c:v",
                    codec_name,
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-shortest",
                    str(temp_final),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.replace(str(temp_final), str(ruta_salida_video))
            print(f"[{nombre_clip_log}] Normalizado a 9:16 ({ANCHO_SALIDA}x{ALTO_SALIDA}).")
            return
        except Exception as e:
            print(f"[{nombre_clip_log}] Fallo normalizando dimensiones con {codec_name}: {e}")

    if temp_final.exists():
        temp_final.unlink(missing_ok=True)
def render_stacked_layout(
    ruta_clip: Path,
    ruta_salida_video: Path,
    ruta_top_video_temp: Path,
    segmentos,
    nombre_clip_log: str,
):
    print(f"[{nombre_clip_log}] Render layout: stacked")

    probe_clip = ffmpeg.probe(str(ruta_clip))
    clip_info = next(s for s in probe_clip["streams"] if s["codec_type"] == "video")
    w, h = clip_info["width"], clip_info["height"]
    fps = float(Fraction(clip_info.get("r_frame_rate", "30/1")))

    if FACE_TRACKING_ACTIVE:
        crear_top_video_tracking(ruta_clip, ruta_top_video_temp, w, h, fps)
    else:
        crear_top_video_centrado(ruta_clip, ruta_top_video_temp)

    clip_duration = float(clip_info.get("duration", 0) or probe_clip.get("format", {}).get("duration", 0) or 0)
    if clip_duration <= 0:
        clip_duration = 60.0

    probe_gameplay = ffmpeg.probe(str(RUTA_GAMEPLAY))
    gameplay_duration = float(probe_gameplay["streams"][0].get("duration", 0) or probe_gameplay.get("format", {}).get("duration", 0) or 0)
    max_start_time = max(0.0, gameplay_duration - clip_duration)
    start_time_gameplay = random.uniform(0, max_start_time)

    altura_top = int(ALTO_SALIDA * 0.45)
    altura_gameplay = ALTO_SALIDA - altura_top

    input_top = ffmpeg.input(str(ruta_top_video_temp))
    input_gameplay = ffmpeg.input(RUTA_GAMEPLAY.as_posix(), ss=start_time_gameplay, t=clip_duration)

    gameplay_scaled = (
        input_gameplay.video
        .filter("scale", "iw*1.5", "ih*1.5")
        .filter("crop", f"min(iw,{ANCHO_SALIDA})", f"min(ih,{altura_gameplay})")
        .filter("pad", ANCHO_SALIDA, altura_gameplay, "(ow-iw)/2", "(oh-ih)/2", color="black")
    )

    input_top_scaled = input_top.video.filter("scale", ANCHO_SALIDA, altura_top)
    video_apilado = ffmpeg.filter([input_top_scaled, gameplay_scaled], "vstack", shortest=1)
    video_apilado = aplicar_subtitulos_drawtext(video_apilado, segmentos, y_offset_px=1100)

    ffmpeg.output(
        video_apilado,
        input_top.audio,
        str(ruta_salida_video),
        **{"c:v": VIDEO_CODEC, "c:a": "aac", "b:a": "192k"},
    ).run(overwrite_output=True, quiet=True)

    if ADD_BACKGROUND_MUSIC:
        mezclar_musica_en_video(ruta_salida_video, RUTA_MUSICA, vol_db=8.0)


# -----------------------------------------------------------------------------
# Main clip processing
# -----------------------------------------------------------------------------
def procesar_un_solo_clip(ruta_clip: Path, carpeta_salida_principal: Path):
    nombre_base_clip = ruta_clip.stem
    nombre_clip_log = ruta_clip.name

    carpeta_destino_clip = carpeta_salida_principal / f"editado_{nombre_base_clip}"
    carpeta_destino_clip.mkdir(exist_ok=True)

    ruta_salida_video = carpeta_destino_clip / f"editado_{nombre_base_clip}.mp4"
    ruta_metadata_json = carpeta_destino_clip / "metadata.json"
    ruta_top_video_temp = carpeta_salida_principal / f"temp_top_{nombre_base_clip}.mp4"

    if SKIP_EXISTING_OUTPUTS and ruta_salida_video.exists() and ruta_metadata_json.exists():
        print(f"[{nombre_clip_log}] Ya existe salida final. Saltando.")
        return True

    print(f"--- Empezando procesamiento de: {nombre_clip_log} ---")

    try:
        segmentos = transcribir_audio(ruta_clip)
        transcripcion_completa = " ".join(s.get("text", "") for s in segmentos) if segmentos else ""
        metadatos = generar_metadatos(transcripcion_completa, nombre_clip_log)
        with open(ruta_metadata_json, "w", encoding="utf-8") as f:
            json.dump(metadatos, f, ensure_ascii=False, indent=4)

        if OUTPUT_LAYOUT == "clip_only":
            render_clip_only_layout(ruta_clip, ruta_salida_video, segmentos, nombre_clip_log)
        else:
            render_stacked_layout(
                ruta_clip,
                ruta_salida_video,
                ruta_top_video_temp,
                segmentos,
                nombre_clip_log,
            )

        enforce_vertical_with_audio(ruta_salida_video, ruta_clip, nombre_clip_log)
        ensure_vertical_output(ruta_salida_video, nombre_clip_log)
        return True

    except Exception:
        print(f"Error critico procesando {nombre_clip_log}")
        traceback.print_exc()
        return False
    finally:
        if ruta_top_video_temp.exists():
            try:
                ruta_top_video_temp.unlink()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
def main():
    CARPETA_SALIDA.mkdir(exist_ok=True)

    if not CARPETA_ENTRADA.exists():
        print(f"Error: La carpeta de entrada '{CARPETA_ENTRADA}' no existe.")
        sys.exit(1)

    if OUTPUT_LAYOUT == "stacked" and not RUTA_GAMEPLAY.exists():
        print(f"Error: No se encuentra gameplay en '{RUTA_GAMEPLAY}'.")
        sys.exit(1)

    clips_para_procesar = sorted([f for f in CARPETA_ENTRADA.iterdir() if f.suffix.lower() == ".mp4"])
    if not clips_para_procesar:
        print("No se encontraron clips .mp4 en la carpeta de entrada.")
        sys.exit(1)

    if len(clips_para_procesar) > MAX_CLIPS_TO_EDIT:
        print(f"Limitando edicion a {MAX_CLIPS_TO_EDIT} clips (de {len(clips_para_procesar)}).")
        clips_para_procesar = clips_para_procesar[:MAX_CLIPS_TO_EDIT]

    print(f"Se procesaran {len(clips_para_procesar)} clips en modo secuencial. Layout={OUTPUT_LAYOUT}")
    resultados = [procesar_un_solo_clip(clip, CARPETA_SALIDA) for clip in clips_para_procesar]
    exitosos = sum(1 for r in resultados if r)
    fallidos = len(resultados) - exitosos

    print("Proceso completado.")
    print(f"- Clips exitosos: {exitosos}")
    print(f"- Clips con error: {fallidos}")

    if exitosos == 0:
        sys.exit(1)


if __name__ == "__main__":
    if os.name != "posix":
        set_start_method("spawn", force=True)
    main()







