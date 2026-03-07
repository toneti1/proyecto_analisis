#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seleccion_clip.py
VersiÃ³n mejorada y compatible con automatizador.py:
 - mantiene mejoras (segmentaciÃ³n dinÃ¡mica, prosodia, normalizaciÃ³n, ffmpeg extract, batching, dedupe)
 - CLIPS_OUTPUT_FOLDER apunta a ./clips para que el orquestador encuentre los archivos
 - sale con cÃ³digo != 0 en errores crÃ­ticos para que el orquestador detecte fallos
"""

import os
import sys
import time
import math
import subprocess
import tempfile
from bisect import bisect_left
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from safetensors.torch import load_file

# Optional libs
try:
    import whisper
except Exception:
    whisper = None

try:
    import librosa
except Exception:
    librosa = None

try:
    import parselmouth
except Exception:
    parselmouth = None

# CONFIG
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_N_CLIPS = _env_int("TOP_N_CLIPS", 12, min_value=1)
BASE_FOLDER = r"C:\Users\gruiz\OneDrive\Desktop\clip_generator"
MODEL_PATH = os.path.join(BASE_FOLDER, "viral_clip_detector")
OUTPUT_FOLDER = os.path.join(BASE_FOLDER, "proyecto_analisis")
# <-- Cambio mínimo para compatibilidad: la carpeta de salida de clips es ./clips
CLIPS_OUTPUT_FOLDER = os.path.join(os.getcwd(), "clips")   # compatible con automatizador.py (CLIPS_FOLDER = "clips")
CLIP_MIN_S = 5
CLIP_MAX_S = _env_int("CLIP_MAX_S", 60, min_value=5)
SLIDING_STEP_S = _env_int("SLIDING_STEP_S", 5, min_value=1)
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "tiny")
ENABLE_PROSODY = _env_flag("ENABLE_PROSODY", False)
INFERENCE_BATCH_SIZE = _env_int("INFERENCE_BATCH_SIZE", 24 if DEVICE == "cuda" else 8, min_value=1)
FFMPEG_BIN = "ffmpeg"

# Feature normalization placeholders (ajusta con stats reales)
FEATURE_MEANS = np.array([130.0, 0.25, 0.0, 0.0, 0.02, 0.005], dtype=np.float32)
FEATURE_STDS  = np.array([40.0, 0.20, 0.02, 0.02, 0.03, 0.01], dtype=np.float32)

# -----------------------------
# 1) MODELO: arquitectura
# -----------------------------
class ViralClipModel(nn.Module):
    def __init__(self, num_labels=2):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.2)
        self.feature_layer = nn.Linear(6, 32)
        self.classifier = nn.Linear(self.bert.config.hidden_size + 32, num_labels)

    def forward(self, input_ids, attention_mask, features):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled_output = bert_output.pooler_output
        features_out = torch.relu(self.feature_layer(features))
        combined = torch.cat([pooled_output, features_out], dim=1)
        logits = self.classifier(combined)
        return {'logits': logits, 'pooled_output': pooled_output}

# -----------------------------
# 2) UTIL: descarga video
# -----------------------------
def download_youtube_video_with_yt_dlp(url: str, output_folder: str, filename: str) -> str:
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, filename)
    print(f"[+] Descargando video: {url} -> {output_path}")
    try:
        command = [
            'yt-dlp', '-f', 'bestvideo+bestaudio/best',
            '--merge-output-format', 'mp4', url, '-o', output_path
        ]
        subprocess.run(command, check=True)
        if os.path.exists(output_path):
            print("[âœ“] Descarga completada.")
            return output_path
        else:
            raise FileNotFoundError("yt-dlp no produjo el archivo esperado.")
    except FileNotFoundError:
        print("ERROR: 'yt-dlp' no encontrado. InstÃ¡lalo: pip install yt-dlp")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print("ERROR: yt-dlp fallÃ³:", e)
        sys.exit(1)

# -----------------------------
# 3) TranscripciÃ³n con Whisper
# -----------------------------
def transcribe_audio_with_timestamps(video_path: str):
    if whisper is None:
        print("ERROR: Whisper no estÃ¡ instalado. Instala: pip install -U openai-whisper")
        sys.exit(1)
    print("[+] Cargando modelo Whisper:", WHISPER_SIZE)
    model = whisper.load_model(WHISPER_SIZE, device=DEVICE)
    print("[+] Transcribiendo... esto puede tardar.")
    res = model.transcribe(video_path, word_timestamps=True)
    segments = res.get('segments', [])
    print(f"[âœ“] TranscripciÃ³n completada. {len(segments)} segmentos.")
    return segments

# -----------------------------
# 4) SegmentaciÃ³n dinÃ¡mica por energÃ­a (librosa)
# -----------------------------
def compute_energy_segments(audio_path: str, hop_length=512, energy_thresh_ratio=0.3, min_s=CLIP_MIN_S, max_s=CLIP_MAX_S):
    if librosa is None:
        print("[!] librosa no disponible: no se podrÃ¡n generar segmentos dinÃ¡micos.")
        return []

    y, sr = librosa.load(audio_path, sr=16000)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    max_r = float(np.max(rms)) if len(rms) else 0.0
    thresh = max_r * energy_thresh_ratio
    segments = []
    in_seg = False
    seg_start = 0.0

    for t, r in zip(times, rms):
        if r >= thresh and not in_seg:
            in_seg = True
            seg_start = t
        elif r < thresh and in_seg:
            seg_end = t
            dur = seg_end - seg_start
            if dur >= min_s:
                cur = seg_start
                while cur + max_s < seg_end:
                    segments.append((cur, cur + max_s))
                    cur += max_s
                segments.append((cur, seg_end))
            in_seg = False

    if in_seg:
        seg_end = times[-1]
        dur = seg_end - seg_start
        if dur >= min_s:
            cur = seg_start
            while cur + max_s < seg_end:
                segments.append((cur, cur + max_s))
                cur += max_s
            segments.append((cur, seg_end))

    print(f"[+] SegmentaciÃ³n dinÃ¡mica generÃ³ {len(segments)} segmentos.")
    return segments

# -----------------------------
# 5) Prosodia (parselmouth) + fallback
# -----------------------------
def extract_prosody_from_audio_segment(
    audio_path: str,
    start: float,
    end: float,
    preloaded_sound=None,
) -> Dict[str, float]:
    try:
        if not ENABLE_PROSODY or parselmouth is None:
            raise ImportError("parselmouth no instalado")
        snd = preloaded_sound if preloaded_sound is not None else parselmouth.Sound(audio_path)
        seg = snd.extract_part(from_time=start, to_time=end, preserve_times=True)
        pitch = seg.to_pitch()
        pitch_vals = pitch.selected_array['frequency']
        pitch_vals = pitch_vals[pitch_vals > 0]
        pitch_mean = float(np.mean(pitch_vals)) if len(pitch_vals) else 0.0
        pitch_std = float(np.std(pitch_vals)) if len(pitch_vals) else 0.0
        intensity = seg.to_intensity().values.mean()
        return {'pitch_mean': pitch_mean, 'pitch_std': pitch_std, 'intensity': float(intensity)}
    except Exception:
        return {'pitch_mean': 0.0, 'pitch_std': 0.0, 'intensity': 0.0}

# -----------------------------
# 6) Prepare features y tokenizaciÃ³n
# -----------------------------
def punctuation_restore_simple(text: str) -> str:
    qwords = ("who", "what", "when", "where", "why", "how", "is", "are", "do", "does", "did", "could", "would", "should")
    t = text.strip()
    if not t:
        return t
    if t[-1] not in ".!?":
        t = t + "."
    first_word = t.split()[0].lower().strip()
    if first_word in qwords:
        t = t[:-1] + "?"
    return t

def prepare_single_clip_for_inference(clip_data: dict, tokenizer: BertTokenizer) -> dict:
    text = punctuation_restore_simple(clip_data['transcription'])
    encoding = tokenizer.encode_plus(
        text, add_special_tokens=True, max_length=256,
        padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
    )
    wpm = clip_data.get('speech_rate_wpm', 0.0)
    emotions = clip_data.get('emotions', {})
    emotion_intensity = emotions.get('emotion_intensity', 1.0 - emotions.get('neutral', 0.0)) if emotions else clip_data.get('heur_emotion_intensity', 0.1)
    surprise_joy = emotions.get('surprise', 0.0) * emotions.get('joy', 0.0) if emotions else 0.0
    surprise_anger = emotions.get('surprise', 0.0) * emotions.get('anger', 0.0) if emotions else 0.0
    word_count = len(text.split())
    question_density = text.count('?') / max(word_count, 1)
    exclamation_density = text.count('!') / max(word_count, 1)
    raw = np.array([wpm, emotion_intensity, surprise_joy, surprise_anger, question_density, exclamation_density], dtype=np.float32)
    normalized = (raw - FEATURE_MEANS) / (FEATURE_STDS + 1e-9)
    feature_vector = torch.tensor(normalized.reshape(1, -1), dtype=torch.float).to(DEVICE)
    return {
        'input_ids': encoding['input_ids'].to(DEVICE),
        'attention_mask': encoding['attention_mask'].to(DEVICE),
        'features': feature_vector
    }

# -----------------------------
# 7) Dedupe semÃ¡ntica simple (cosine greedy)
# -----------------------------
def dedupe_by_embeddings(clips: List[dict], embeddings: List[np.ndarray], threshold: float = 0.85) -> List[dict]:
    picked = []
    picked_embs = []
    for clip, emb in sorted(zip(clips, embeddings), key=lambda x: x[0]['viral_score'], reverse=True):
        keep = True
        for p_emb in picked_embs:
            cos = float(np.dot(emb, p_emb) / (np.linalg.norm(emb) * np.linalg.norm(p_emb) + 1e-9))
            if cos >= threshold:
                keep = False
                break
        if keep:
            picked.append(clip)
            picked_embs.append(emb)
        if len(picked) >= TOP_N_CLIPS:
            break
    return picked

# -----------------------------
# 8) ffmpeg extraction rÃ¡pido
# -----------------------------
def ffmpeg_extract(input_path: str, output_path: str, start: float, end: float, recode=False):
    cmd = [FFMPEG_BIN, '-y', '-ss', str(start), '-to', str(end), '-i', input_path]
    if not recode:
        cmd += ['-c', 'copy', output_path]
    else:
        cmd += ['-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-c:a', 'aac', output_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        subprocess.run([FFMPEG_BIN, '-y', '-ss', str(start), '-to', str(end), '-i', input_path, '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-c:a', 'aac', output_path], check=True)

# -----------------------------
# 9) Carga del modelo y tokenizer con robustez
# -----------------------------
def load_model_and_tokenizer(model_path: str):
    print("[+] Cargando modelo y tokenizer...")
    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        print("[+] Tokenizer cargado desde", model_path)
    except Exception:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        print("[!] Tokenizer no encontrado en model_path; usando 'bert-base-uncased'")

    model = ViralClipModel()
    weights_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(weights_path):
        try:
            state_dict = load_file(weights_path, device=DEVICE)
            model.load_state_dict(state_dict, strict=False)
            print("[+] Pesos cargados desde model.safetensors")
        except Exception as e:
            print("[!] Error cargando pesos:", e)
            print("[!] Continuando con pesos aleatorios (usa esto solo para pruebas).")
    else:
        print("[!] No se encontrÃ³ model.safetensors en", model_path, "; usando pesos por defecto.")

    model.to(DEVICE)
    model.eval()
    return model, tokenizer

# -----------------------------
# 10) Filtrado de solapamientos por interval overlap
# -----------------------------
def filter_overlapping_clips(ranked_clips: List[dict], num_clips: int, min_distance_s: float = 15.0) -> List[dict]:
    final = []
    for clip in ranked_clips:
        if len(final) >= num_clips:
            break
        s1, e1 = clip['start_time'], clip['end_time']
        overlapping = False
        for sel in final:
            s2, e2 = sel['start_time'], sel['end_time']
            if not (e1 + min_distance_s < s2 or s1 - min_distance_s > e2):
                overlapping = True
                break
        if not overlapping:
            final.append(clip)
    return final

# -----------------------------
# 11) MAIN: pipeline
# -----------------------------
def main(YOUTUBE_URL: str):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(CLIPS_OUTPUT_FOLDER, exist_ok=True)

    ts = time.strftime('%Y%m%d_%H%M%S')
    downloaded_filename = f"{ts}.mp4"

    video_path = download_youtube_video_with_yt_dlp(YOUTUBE_URL, OUTPUT_FOLDER, downloaded_filename)

    if not video_path or not os.path.exists(video_path):
        print("ERROR: no se pudo descargar el video.")
        sys.exit(1)

    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    if model is None or tokenizer is None:
        print("ERROR cargando modelo/tokenizer.")
        sys.exit(1)

    segments = transcribe_audio_with_timestamps(video_path)
    if not segments:
        print("ERROR: No se pudieron obtener segmentos de transcripciÃ³n.")
        sys.exit(1)

    tmp_wav = os.path.join(tempfile.gettempdir(), f"audio_{ts}.wav")
    try:
        subprocess.run([FFMPEG_BIN, '-y', '-i', video_path, '-ar', '16000', '-ac', '1', tmp_wav], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print("ERROR extrayendo audio:", e)
        sys.exit(1)

    dynamic_segments = compute_energy_segments(tmp_wav) if librosa is not None else []

    candidate_intervals = []
    if dynamic_segments:
        candidate_intervals = dynamic_segments
    else:
        duration = segments[-1]['end'] if segments else 0
        if duration <= 0:
            print("ERROR: duraciÃ³n del audio desconocida.")
            sys.exit(1)
        cur = 0.0
        while cur < duration:
            end = min(cur + CLIP_MAX_S, duration)
            candidate_intervals.append((cur, end))
            cur += SLIDING_STEP_S

    print("[+] Construyendo clips candidatos a partir de timestamps de palabras...")
    clips_candidates = []
    words_flat = []
    for seg in segments:
        for w in seg.get('words', []):
            ws = w.get('start', w.get('t', None))
            we = w.get('end', w.get('end', None))
            word_text = w.get('word', w.get('text', ''))
            if ws is not None and we is not None:
                words_flat.append({'start': ws, 'end': we, 'word': word_text})

    preloaded_sound = None
    if ENABLE_PROSODY and parselmouth is not None:
        try:
            preloaded_sound = parselmouth.Sound(tmp_wav)
        except Exception:
            preloaded_sound = None

    word_starts = [w['start'] for w in words_flat]
    for (s, e) in candidate_intervals:
        i0 = bisect_left(word_starts, s)
        i1 = bisect_left(word_starts, e, lo=i0)
        words = [words_flat[i]['word'] for i in range(i0, i1)]
        if not words:
            continue
        text = ' '.join(words).strip()
        word_count = len(text.split())
        duration = max(1.0, e - s)
        wpm = (word_count / duration) * 60.0
        pros = extract_prosody_from_audio_segment(tmp_wav, s, e, preloaded_sound=preloaded_sound)
        heur_emotion_intensity = min(1.0, pros['intensity'] / 100.0 + (pros['pitch_std'] / 50.0))
        clip = {
            'start_time': float(s),
            'end_time': float(e),
            'transcription': text,
            'speech_rate_wpm': float(wpm),
            'prosody': pros,
            'heur_emotion_intensity': float(heur_emotion_intensity)
        }
        clips_candidates.append(clip)

    print(f"[+] {len(clips_candidates)} clips candidatos generados.")
    if not clips_candidates:
        print("ERROR: No se generaron clips candidatos.")
        sys.exit(1)

    print("[+] Evaluando candidatos con el modelo...")
    results = []
    for i in range(0, len(clips_candidates), INFERENCE_BATCH_SIZE):
        batch = clips_candidates[i:i + INFERENCE_BATCH_SIZE]
        inputs = [prepare_single_clip_for_inference(c, tokenizer) for c in batch]
        input_ids = torch.cat([x['input_ids'] for x in inputs], dim=0)
        attention_mask = torch.cat([x['attention_mask'] for x in inputs], dim=0)
        features = torch.cat([x['features'] for x in inputs], dim=0)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
            logits = out['logits']
            pooled = out.get('pooled_output', None)
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            pooled_np = pooled.cpu().numpy() if pooled is not None else None
        for j, c in enumerate(batch):
            viral_score = float(probs[j][1]) if probs.shape[1] > 1 else float(probs[j][0])
            entry = {**c, 'viral_score': viral_score}
            entry['embedding'] = pooled_np[j] if pooled_np is not None else None
            results.append(entry)

    if not results:
        print("ERROR: No hubo resultados tras inference.")
        sys.exit(1)

    results_sorted = sorted(results, key=lambda x: x['viral_score'], reverse=True)
    embeddings = [r['embedding'] for r in results_sorted if r.get('embedding') is not None]
    if embeddings:
        emb_norms = [e / (np.linalg.norm(e) + 1e-9) for e in embeddings]
        filtered = dedupe_by_embeddings([r for r in results_sorted if r.get('embedding') is not None], emb_norms, threshold=0.88)
        filtered_ids = set(id(x) for x in filtered)
        for r in results_sorted:
            if len(filtered) >= TOP_N_CLIPS:
                break
            if id(r) not in filtered_ids:
                filtered.append(r)
        top_clips = filtered[:TOP_N_CLIPS]
    else:
        top_clips = filter_overlapping_clips(results_sorted, TOP_N_CLIPS, min_distance_s=15.0)

    print(f"[+] Se seleccionaron {len(top_clips)} clips finales.")
    if not top_clips:
        print("ERROR: no se seleccionaron clips finales.")
        sys.exit(1)

    run_ts = time.strftime('%Y%m%d_%H%M%S')

    for idx, clip in enumerate(top_clips):
        start = clip['start_time']
        end = clip['end_time']
        score = clip['viral_score']
        dur = end - start
        if dur > CLIP_MAX_S:
            end = start + CLIP_MAX_S
        out_name = f"clip_{run_ts}_{idx+1}_s{int(start)}_e{int(end)}_score_{score:.3f}.mp4"
        out_path = os.path.join(CLIPS_OUTPUT_FOLDER, out_name)
        print(f"[>] Exportando clip {idx+1}: {start:.1f}s - {end:.1f}s  (score={score:.3f}) -> {out_path}")
        try:
            ffmpeg_extract(video_path, out_path, start, end, recode=False)
            print("[âœ“] Exportado:", out_path)
        except Exception as e:
            print("[!] Error exportando con -c copy, intentando recodificar:", e)
            ffmpeg_extract(video_path, out_path, start, end, recode=True)
            print("[âœ“] Exportado (recode):", out_path)

    try:
        os.remove(video_path)
    except Exception:
        pass
    try:
        os.remove(tmp_wav)
    except Exception:
        pass

    print("[FIN] Pipeline completado.")
    # Salida exitosa
    sys.exit(0)

# -----------------------------
# entrypoint
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        url = sys.argv[1]
        print("[*] Iniciando pipeline para:", url)
        main(url)
    else:
        print("Uso: python seleccion_clip.py \"URL_DE_YOUTUBE\"")
        sys.exit(1)




