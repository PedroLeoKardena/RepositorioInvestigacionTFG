import librosa
import numpy as np
import soundfile as sf
import tempfile
import os
from pathlib import Path
from pyannote.audio import Pipeline

AUDIO_PATH = r"C:\Users\pedro\git\InvestigacionFinGrado\audios_originales\1-Alto flujo sequedad nasal neo pulmonar 56 años.m4a"
SAMPLE_RATE = 16000
TARGET_DBFS = -20.0


def cargar_y_convertir_a_wav(ruta_audio, ruta_wav_temp):
    y, sr = librosa.load(ruta_audio, sr=SAMPLE_RATE, mono=True)
    rms = np.sqrt(np.mean(y**2))
    if rms > 0:
        rms_objetivo = 10 ** (TARGET_DBFS / 20.0)
        y = y * (rms_objetivo / rms)
        pico = np.max(np.abs(y))
        if pico > 1.0:
            y = y / pico
    sf.write(ruta_wav_temp, y, SAMPLE_RATE)


def main():
    from huggingface_hub import get_token
    token = get_token()
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=token)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        ruta_wav_temp = f.name

    try:
        print(f"Cargando y convirtiendo audio: {Path(AUDIO_PATH).name}")
        cargar_y_convertir_a_wav(AUDIO_PATH, ruta_wav_temp)

        print("Ejecutando diarización (puede tardar unos minutos)...")
        diarization = pipeline(ruta_wav_temp, min_speakers=2, max_speakers=5)

        annotation = diarization.exclusive_speaker_diarization

        print("\n--- Resultado de la diarización ---")
        duraciones = {}
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            duracion = turn.end - turn.start
            print(f"  {turn.start:6.1f}s - {turn.end:6.1f}s  [{duracion:5.1f}s]  {speaker}")
            duraciones[speaker] = duraciones.get(speaker, 0) + duracion

        print("\n--- Tiempo total por hablante ---")
        total = sum(duraciones.values())
        for speaker, t in sorted(duraciones.items(), key=lambda x: -x[1]):
            print(f"  {speaker}: {t:.1f}s ({100*t/total:.1f}%)")

    finally:
        os.unlink(ruta_wav_temp)


if __name__ == "__main__":
    main()
