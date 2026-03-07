# automatizador.py

import os
import glob
import subprocess
import sys
import re
import time
import json

# --- CONFIG ---
CLIP_SELECTOR_SCRIPT = "seleccion_clip.py"
BATCH_EDITOR_SCRIPT = "crear_videos_en_lote.py"
CLIPS_FOLDER = "clips"
PROCESSED_VIDEOS_LOG = "processed_videos.txt"
VIDEOS_SOURCE_FILE = "videos_virales_final.json"

MAX_VIDEOS_TO_PROCESS = int(os.getenv("MAX_VIDEOS_TO_PROCESS", "20"))
WAIT_AFTER_SUCCESS_S = int(os.getenv("WAIT_AFTER_SUCCESS_S", "2"))
WAIT_AFTER_ERROR_S = int(os.getenv("WAIT_AFTER_ERROR_S", "3"))
WAIT_AFTER_EMPTY_S = int(os.getenv("WAIT_AFTER_EMPTY_S", "1"))


def load_processed_videos():
    if not os.path.exists(PROCESSED_VIDEOS_LOG):
        open(PROCESSED_VIDEOS_LOG, "w").close()
        return set()

    with open(PROCESSED_VIDEOS_LOG, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def log_processed_video(video_id):
    with open(PROCESSED_VIDEOS_LOG, "a", encoding="utf-8") as f:
        f.write(video_id + "\n")


def extract_video_id_from_url(url):
    match = re.search(r"(?<=v=)[a-zA-Z0-9_-]+|(?<=be/)[a-zA-Z0-9_-]+", url)
    return match.group(0) if match else None


def remove_processed_url_from_json(url_to_remove, json_file):
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            videos_data = json.load(f)

        if not isinstance(videos_data, list):
            print(f"El archivo {json_file} no contiene una lista de URLs.")
            return False

        if url_to_remove not in videos_data:
            print(f"La URL no se encontro en el archivo {json_file}.")
            return False

        videos_data.remove(url_to_remove)
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(videos_data, f, ensure_ascii=False, indent=2)

        print(f"URL eliminada del archivo {json_file}.")
        return True
    except Exception as e:
        print(f"Error al eliminar URL de {json_file}: {e}")
        return False


def clean_clips_folder():
    print(f"Limpiando carpeta '{CLIPS_FOLDER}'...")
    files_to_delete = glob.glob(os.path.join(CLIPS_FOLDER, "*.mp4"))
    if not files_to_delete:
        print("La carpeta ya estaba limpia.")
        return

    for file_path in files_to_delete:
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Error al eliminar {file_path}: {e}")

    print("Carpeta de clips limpiada.")


def run_script(script_name, *args, env_overrides=None):
    try:
        venv_python = os.path.join(os.getcwd(), "env_clips_act", "Scripts", "python.exe")
        python_executable = venv_python if os.path.exists(venv_python) else sys.executable
        command = [python_executable, script_name] + list(args)
        env = os.environ.copy()
        if env_overrides:
            env.update({k: str(v) for k, v in env_overrides.items()})

        print(f"Ejecutando: {' '.join(command)}")
        subprocess.run(command, check=True, env=env)
        return True
    except FileNotFoundError:
        print(f"Error: no se encontro script '{script_name}'.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error ejecutando '{script_name}'. Exit code: {e.returncode}")
        return False
    except Exception as e:
        print(f"Error inesperado ejecutando '{script_name}': {e}")
        return False


def infer_edited_paths_for_raw(clips_raw):
    clips_editados_folder = os.path.join(os.getcwd(), "clips_editados")
    inferred = []

    for raw_clip in clips_raw:
        clip_stem = os.path.splitext(os.path.basename(raw_clip))[0]
        expected = os.path.join(
            clips_editados_folder,
            f"editado_{clip_stem}",
            f"editado_{clip_stem}.mp4",
        )
        if os.path.exists(expected):
            inferred.append(expected)

    return sorted(inferred)


def build_env_overrides(clip_count=None):
    overrides = {
        "OUTPUT_LAYOUT": "clip_only",
        "ADD_BACKGROUND_MUSIC": "0",
    }

    if clip_count is not None:
        clip_count = max(1, int(clip_count))
        overrides["TOP_N_CLIPS"] = str(clip_count)
        overrides["MAX_CLIPS_TO_EDIT"] = str(clip_count)

    return overrides


def main_orchestrator():
    print("--- INICIO DEL PROCESO DE AUTOMATIZACION ---")

    processed_ids = load_processed_videos()
    print(f"Se cargaron {len(processed_ids)} videos ya procesados.")

    try:
        with open(VIDEOS_SOURCE_FILE, "r", encoding="utf-8") as f:
            videos_to_process = json.load(f)

        video_urls = videos_to_process[:MAX_VIDEOS_TO_PROCESS]
        print(f"Se cargaron {len(videos_to_process)} videos. Se procesaran {len(video_urls)}.")
    except FileNotFoundError:
        print(f"Error: el archivo '{VIDEOS_SOURCE_FILE}' no existe.")
        return
    except (KeyError, json.JSONDecodeError):
        print(f"Error: el archivo '{VIDEOS_SOURCE_FILE}' esta corrupto o invalido.")
        return

    if not video_urls:
        print("No hay videos para procesar.")
        return

    env_overrides = build_env_overrides()
    new_videos_processed_count = 0
    current_index = 0

    while new_videos_processed_count < MAX_VIDEOS_TO_PROCESS and current_index < len(video_urls):
        url = video_urls[current_index]
        print(f"\n--- VIDEO {current_index + 1}/{len(video_urls)} ---")

        video_id = extract_video_id_from_url(url)
        if not video_id:
            print(f"No se pudo extraer ID de URL: {url}")
            current_index += 1
            continue

        if video_id in processed_ids:
            print(f"ID '{video_id}' ya procesado. Saltando.")
            current_index += 1
            continue

        print(f"Procesando ID: {video_id}")
        print(f"URL: {url}")

        clean_clips_folder()

        if not run_script(CLIP_SELECTOR_SCRIPT, url, env_overrides=env_overrides):
            print("Fallo en seleccion_clip.py. Saltando video.")
            time.sleep(WAIT_AFTER_ERROR_S)
            current_index += 1
            continue

        generated_clips = glob.glob(os.path.join(CLIPS_FOLDER, "*.mp4"))
        if not generated_clips:
            print("No se generaron clips. Saltando video.")
            time.sleep(WAIT_AFTER_EMPTY_S)
            current_index += 1
            continue

        print(f"Se encontraron {len(generated_clips)} clips para editar.")

        if not run_script(BATCH_EDITOR_SCRIPT, env_overrides=env_overrides):
            print("Fallo en crear_videos_en_lote.py.")
            time.sleep(WAIT_AFTER_ERROR_S)
            current_index += 1
            continue

        log_processed_video(video_id)
        remove_processed_url_from_json(url, VIDEOS_SOURCE_FILE)
        print(f"Video '{video_id}' procesado y registrado.")

        new_videos_processed_count += 1
        time.sleep(WAIT_AFTER_SUCCESS_S)
        current_index += 1

    print("\n--- PROCESO COMPLETADO ---")
    print(f"Videos nuevos procesados: {new_videos_processed_count}")


def process_url(url: str, clip_count: int = 12):
    if not url or not url.strip():
        raise ValueError("URL vacia o invalida.")

    url = url.strip()
    env_overrides = build_env_overrides(clip_count=clip_count)
    clean_clips_folder()

    # Para Streamlit: siempre procesa la URL recibida.
    if not run_script(CLIP_SELECTOR_SCRIPT, url, env_overrides=env_overrides):
        return {"clips_raw": [], "clips_edited": []}

    clips_raw = sorted(glob.glob(os.path.join(CLIPS_FOLDER, "*.mp4")))
    if not clips_raw:
        return {"clips_raw": [], "clips_edited": []}

    if not run_script(BATCH_EDITOR_SCRIPT, env_overrides=env_overrides):
        return {"clips_raw": clips_raw, "clips_edited": []}

    clips_edited = infer_edited_paths_for_raw(clips_raw)
    return {"clips_raw": clips_raw, "clips_edited": clips_edited}


if __name__ == "__main__":
    main_orchestrator()
