import librosa
import numpy as np
import soundfile as sf
import tempfile
import os
import pandas as pd
from pathlib import Path
from pyannote.audio import Pipeline
from huggingface_hub import get_token

AUDIOS_DIR = Path(__file__).parent.parent / "audios_originales"
OUTPUT_CSV = Path(__file__).parent.parent / "datos_entrenamiento" / "diarizacion_reporte.csv"
SAMPLE_RATE = 16000
TARGET_DBFS = -20.0


def normalizar_audio(ruta_audio, ruta_wav_temp):
    y, sr = librosa.load(ruta_audio, sr=SAMPLE_RATE, mono=True)
    rms = np.sqrt(np.mean(y**2))
    if rms > 0:
        rms_objetivo = 10 ** (TARGET_DBFS / 20.0)
        y = y * (rms_objetivo / rms)
        pico = np.max(np.abs(y))
        if pico > 1.0:
            y = y / pico
    sf.write(ruta_wav_temp, y, SAMPLE_RATE)


def diarizar_audio(pipeline, ruta_wav):
    resultado = pipeline(ruta_wav, min_speakers=1, max_speakers=5)
    return resultado.exclusive_speaker_diarization


def procesar_audio(pipeline, ruta_audio):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        ruta_wav_temp = f.name
    try:
        normalizar_audio(ruta_audio, ruta_wav_temp)
        annotation = diarizar_audio(pipeline, ruta_wav_temp)
    finally:
        os.unlink(ruta_wav_temp)

    segmentos = []
    duraciones_por_speaker = {}
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        duracion = turn.end - turn.start
        segmentos.append({"speaker": speaker, "t_inicio": round(turn.start, 2), "t_fin": round(turn.end, 2), "duracion_s": round(duracion, 2)})
        duraciones_por_speaker[speaker] = duraciones_por_speaker.get(speaker, 0) + duracion

    total = sum(duraciones_por_speaker.values())
    filas = []
    for seg in segmentos:
        sp = seg["speaker"]
        filas.append({
            "audio_file": ruta_audio.name,
            "speaker": sp,
            "t_inicio": seg["t_inicio"],
            "t_fin": seg["t_fin"],
            "duracion_s": seg["duracion_s"],
            "duracion_total_speaker_s": round(duraciones_por_speaker[sp], 2),
            "porcentaje": round(100 * duraciones_por_speaker[sp] / total, 1) if total > 0 else 0,
            "es_paciente": "",
        })
    return filas


def main():
    token = get_token()
    print("Cargando pipeline de diarización...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=token)

    audios = sorted(AUDIOS_DIR.glob("*.m4a"))
    print(f"Encontrados {len(audios)} audios en {AUDIOS_DIR}\n")

    todas_las_filas = []
    for i, ruta_audio in enumerate(audios, 1):
        print(f"[{i}/{len(audios)}] {ruta_audio.name}")
        try:
            filas = procesar_audio(pipeline, ruta_audio)
            todas_las_filas.extend(filas)
            speakers = set(f["speaker"] for f in filas)
            for sp in sorted(speakers):
                t = next(f["duracion_total_speaker_s"] for f in filas if f["speaker"] == sp)
                pct = next(f["porcentaje"] for f in filas if f["speaker"] == sp)
                print(f"    {sp}: {t:.1f}s ({pct:.1f}%)")
        except Exception as e:
            print(f"    ERROR: {e}")

    df = pd.DataFrame(todas_las_filas)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nReporte guardado en: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
