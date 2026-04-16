import librosa
import pickle
import pandas as pd
from pathlib import Path
import numpy as np
import warnings


def extraccion_mfccs(y, sr, n_mfcc=30):
    mfccs = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=512,
        win_length=400,
        hop_length=160,
        window='hann'
    )
    return mfccs


def procesar_dataset(ruta_csv, ruta_audios):
    try:
        df = pd.read_csv(ruta_csv, encoding="utf-8", sep=";")
        print(f"Archivo CSV '{ruta_csv.name}' leído correctamente. ({len(df)} chunks)")
    except FileNotFoundError:
        print(f"Error: no se encontró '{ruta_csv.name}'")
        return []

    dataset_procesado = []
    for _, fila in df.iterrows():
        nombre_archivo = fila['nombre_archivo']
        ruta_audio = ruta_audios / nombre_archivo

        if ruta_audio.exists():
            try:
                # Los chunks ya están preprocesados: carga directa sin normalizar
                y, sr = librosa.load(ruta_audio, sr=None)

                mfccs = extraccion_mfccs(y, sr)

                dataset_procesado.append({
                    'nombre_archivo': nombre_archivo,
                    'audio_original': fila.get('audio_original', ''),
                    'chunk_id': fila.get('chunk_id', -1),
                    'mfccs': mfccs,
                    'grupo': fila['grupo'],
                    'caja_toracica': fila['caja_toracica'],
                    'fold': fila.get('fold', -1),
                })

                print(f"  -> OK. MFCC: {mfccs.shape} | {nombre_archivo}")
            except Exception as e:
                print(f"  -> ERROR en {nombre_archivo}: {e}")
        else:
            print(f"  -> No encontrado: {nombre_archivo}")

    print(f"\nFinalizado. Chunks procesados: {len(dataset_procesado)}")
    return dataset_procesado


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ruta_base = Path(__file__).resolve().parent.parent
    ruta_entrenamiento = ruta_base / "datos_entrenamiento"
    ruta_audios = ruta_base / "audios_chunks"

    print("=== EXTRACCIÓN DE MFCCs — AUDIOS CHUNKEADOS ===\n")

    print("--- Train ---")
    datos_train = procesar_dataset(ruta_entrenamiento / "metadata_train_chunked.csv", ruta_audios)
    if datos_train:
        ruta_out = ruta_entrenamiento / "train_mfcc_chunked.pkl"
        with open(ruta_out, 'wb') as f:
            pickle.dump(datos_train, f)
        print(f"Guardado: {ruta_out}")

    print("\n--- Test ---")
    datos_test = procesar_dataset(ruta_entrenamiento / "metadata_test_chunked.csv", ruta_audios)
    if datos_test:
        ruta_out = ruta_entrenamiento / "test_mfcc_chunked.pkl"
        with open(ruta_out, 'wb') as f:
            pickle.dump(datos_test, f)
        print(f"Guardado: {ruta_out}")

    print("\nPROCESO COMPLETADO.")
