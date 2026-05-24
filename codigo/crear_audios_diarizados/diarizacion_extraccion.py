import librosa
import numpy as np
import soundfile as sf
import pandas as pd
from pathlib import Path

AUDIOS_DIR = Path(__file__).parent.parent.parent / "audios_originales"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "audios_diarizados"
REPORTE_CSV = Path(__file__).parent.parent.parent / "datos_entrenamiento" / "diarizacion_reporte_completo.csv"
SAMPLE_RATE = 16000
TARGET_DBFS = -20.0


def normalizar_audio(y):
    rms = np.sqrt(np.mean(y**2))
    if rms == 0:
        return y
    rms_objetivo = 10 ** (TARGET_DBFS / 20.0)
    y = y * (rms_objetivo / rms)
    pico = np.max(np.abs(y))
    if pico > 1.0:
        y = y / pico
    return y


def extraer_segmentos_paciente(ruta_audio, segmentos):
    y, _ = librosa.load(ruta_audio, sr=SAMPLE_RATE, mono=True)
    y = normalizar_audio(y)

    fragmentos = []
    for _, row in segmentos.iterrows():
        inicio = int(row["t_inicio"] * SAMPLE_RATE)
        fin = int(row["t_fin"] * SAMPLE_RATE)
        fragmentos.append(y[inicio:fin])

    if not fragmentos:
        return None
    return np.concatenate(fragmentos)


def main():
    df = pd.read_csv(REPORTE_CSV, encoding="utf-8")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audios_utiles = df[df["es_paciente"] != -1]["audio_file"].unique()
    print(f"Audios a procesar: {len(audios_utiles)} (descartados: {df[df['es_paciente'] == -1]['audio_file'].nunique()})\n")

    ok, sin_segmentos, error = 0, 0, 0

    for audio_file in sorted(audios_utiles):
        ruta_audio = AUDIOS_DIR / audio_file
        segmentos = df[(df["audio_file"] == audio_file) & (df["es_paciente"] == 1)]

        if segmentos.empty:
            print(f"  [SIN SEGMENTOS] {audio_file}")
            sin_segmentos += 1
            continue

        try:
            audio_paciente = extraer_segmentos_paciente(ruta_audio, segmentos)
            if audio_paciente is None or len(audio_paciente) == 0:
                print(f"  [VACÍO] {audio_file}")
                sin_segmentos += 1
                continue

            nombre_salida = Path(audio_file).stem + ".wav"
            ruta_salida = OUTPUT_DIR / nombre_salida
            sf.write(ruta_salida, audio_paciente, SAMPLE_RATE)

            duracion = len(audio_paciente) / SAMPLE_RATE
            print(f"  [OK] {audio_file} → {nombre_salida} ({duracion:.1f}s)")
            ok += 1

        except Exception as e:
            print(f"  [ERROR] {audio_file}: {e}")
            error += 1

    print(f"\nResumen: {ok} extraídos, {sin_segmentos} sin segmentos, {error} errores")
    print(f"Audios guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
