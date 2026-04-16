import librosa
import numpy as np
import soundfile as sf
import pandas as pd
import shutil
import gc
from pathlib import Path


CHUNK_DURATION_S = 10
MIN_CHUNK_DURATION_S = 5
SAMPLE_RATE = 16000
TARGET_DBFS = -20.0

CHUNK_SAMPLES = CHUNK_DURATION_S * SAMPLE_RATE
MIN_CHUNK_SAMPLES = MIN_CHUNK_DURATION_S * SAMPLE_RATE


def preprocesado_basico(ruta_audio, target_dbfs=TARGET_DBFS):
    y, sr = librosa.load(ruta_audio, sr=SAMPLE_RATE, mono=True)

    rms = np.sqrt(np.mean(y**2))
    if rms == 0:
        return y, sr

    rms_objetivo = 10 ** (target_dbfs / 20.0)
    y_norm = y * (rms_objetivo / rms)

    pico = np.max(np.abs(y_norm))
    if pico > 1.0:
        y_norm = y_norm / pico

    return y_norm, sr


def dividir_en_chunks(y, chunk_samples=CHUNK_SAMPLES, min_samples=MIN_CHUNK_SAMPLES):
    chunks = []
    inicio = 0
    while inicio < len(y):
        fin = inicio + chunk_samples
        chunk = y[inicio:fin]
        if len(chunk) >= min_samples:
            # Rellenar con ceros si el chunk es válido pero más corto que chunk_samples
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            chunks.append(chunk)
        inicio = fin
    return chunks


def procesar_y_chunkear(ruta_audios_originales, ruta_salida_chunks):
    if ruta_salida_chunks.exists():
        shutil.rmtree(ruta_salida_chunks)
        print(f"Carpeta existente eliminada: {ruta_salida_chunks}")
    ruta_salida_chunks.mkdir(parents=True)

    archivos = sorted([
        f for f in ruta_audios_originales.iterdir()
        if f.suffix.lower() in {'.wav', '.m4a', '.mp3'}
    ])
    print(f"Audios encontrados: {len(archivos)}\n")

    resumen = []
    for idx, ruta_audio in enumerate(archivos):
        nombre_base = ruta_audio.stem
        print(f"[{idx+1}/{len(archivos)}] {ruta_audio.name}")

        try:
            y, sr = preprocesado_basico(ruta_audio)
            duracion_s = len(y) / sr
            chunks = dividir_en_chunks(y)

            for i, chunk in enumerate(chunks):
                nombre_chunk = f"{nombre_base}_chunk{i:03d}.wav"
                sf.write(ruta_salida_chunks / nombre_chunk, chunk, sr)

            print(f"   -> {duracion_s:.1f}s  |  {len(chunks)} chunks generados")
            resumen.append({'nombre_original': ruta_audio.name, 'num_chunks': len(chunks), 'duracion_s': round(duracion_s, 1)})

            del y, chunks
            gc.collect()

        except Exception as e:
            print(f"   -> ERROR: {e}")
            resumen.append({'nombre_original': ruta_audio.name, 'num_chunks': 0, 'duracion_s': 0})

    return resumen


def generar_metadata_chunkeada(ruta_csv_original, ruta_salida_chunks, resumen_chunks):
    df = pd.read_csv(ruta_csv_original, sep=";", encoding="utf-8")

    mapa_chunks = {r['nombre_original']: r['num_chunks'] for r in resumen_chunks}

    filas = []
    for _, row in df.iterrows():
        nombre_orig = row['nombre_archivo']
        nombre_base = Path(nombre_orig).stem
        num_chunks = mapa_chunks.get(nombre_orig, 0)

        for i in range(num_chunks):
            nueva_fila = row.copy()
            nueva_fila['nombre_archivo'] = f"{nombre_base}_chunk{i:03d}.wav"
            nueva_fila['chunk_id'] = i
            nueva_fila['audio_original'] = nombre_orig
            filas.append(nueva_fila)

    return pd.DataFrame(filas)


if __name__ == "__main__":
    ruta_base = Path(__file__).resolve().parent.parent
    ruta_originales = ruta_base / "audios_originales"
    ruta_chunks = ruta_base / "audios_chunks"
    ruta_entrenamiento = ruta_base / "datos_entrenamiento"

    print("=== PASO 1: CHUNKING DE AUDIOS ORIGINALES ===")
    resumen = procesar_y_chunkear(ruta_originales, ruta_chunks)

    print("\n=== PASO 2: GENERANDO METADATOS CHUNKEADOS ===")
    for split, csv_entrada, csv_salida in [
        ("train", "metadata_train.csv", "metadata_train_chunked.csv"),
        ("test",  "metadata_test.csv",  "metadata_test_chunked.csv"),
    ]:
        ruta_csv_in = ruta_entrenamiento / csv_entrada
        if not ruta_csv_in.exists():
            print(f"No encontrado: {csv_entrada}, se omite.")
            continue

        df_chunked = generar_metadata_chunkeada(ruta_csv_in, ruta_chunks, resumen)
        ruta_csv_out = ruta_entrenamiento / csv_salida
        df_chunked.to_csv(ruta_csv_out, index=False, sep=";", encoding="utf-8")
        print(f"{split}: {len(df_chunked)} chunks -> {csv_salida}")

    print("\n=== RESUMEN ===")
    total_chunks = sum(r['num_chunks'] for r in resumen)
    total_audios = sum(1 for r in resumen if r['num_chunks'] > 0)
    print(f"Audios procesados: {total_audios}")
    print(f"Chunks totales generados: {total_chunks}")
    print(f"Directorio de salida: {ruta_chunks}")
    print("\nPROCESO COMPLETADO.")
