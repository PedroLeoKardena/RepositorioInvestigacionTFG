import librosa

import pickle
import pandas as pd
from pathlib import Path
import numpy as np
import warnings

TARGET_FRAMES = 1001
def extraccion_melSpectrograrm(y, sr, mel_bands=128, n_fft=512, hop_length=160, win_length=400):
    mels = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=mel_bands,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window='hann'
    )

    mels_db = librosa.power_to_db(mels, ref=np.max)
    n_frames = mels_db.shape[1]
    if n_frames < TARGET_FRAMES:
        #Cuanto hay que rellenar
        pad_width = TARGET_FRAMES - n_frames

        #Hacemos padding con silencio, no con 0s.
        mels_db = np.pad(
            mels_db, 
            pad_width=((0, 0), (0, pad_width)), 
            mode='constant', 
            constant_values=mels_db.min()
        )
    elif n_frames > TARGET_FRAMES:
        mels_db = mels_db[:, :TARGET_FRAMES]

    return mels_db


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

                mels = extraccion_melSpectrograrm(y, sr)

                dataset_procesado.append({
                    'nombre_archivo': nombre_archivo,
                    'audio_original': fila.get('audio_original', ''),
                    'chunk_id': fila.get('chunk_id', -1),
                    'mels': mels,
                    'grupo': fila['grupo'],
                    'caja_toracica': fila['caja_toracica'],
                    'fold': fila.get('fold', -1),
                })

                print(f"  -> OK. Mel Spectrogram: {mels.shape} | {nombre_archivo}")

            except Exception as e:
                print(f"  -> ERROR en {nombre_archivo}: {e}")
        else:
            print(f"  -> No encontrado: {nombre_archivo}")

    print(f"\nFinalizado. Chunks procesados: {len(dataset_procesado)}")
    return dataset_procesado


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ruta_base = Path(__file__).resolve().parent.parent.parent
    ruta_entrenamiento = ruta_base / "datos_entrenamiento"
    ruta_audios = ruta_base / "audios_aumentados"

    print("=== EXTRACCIÓN DE MEL SPECTROGRAMS — AUDIOS CHUNKEADOS ===\n")

    print("--- Train ---")
    datos_train = procesar_dataset(ruta_entrenamiento / "metadata_train_aumentado.csv", ruta_audios)
    if datos_train:
        ruta_out = ruta_entrenamiento / "train_mels_chunked_aumentado.pkl"
        with open(ruta_out, 'wb') as f:
            pickle.dump(datos_train, f)
        print(f"Guardado: {ruta_out}")

    print("\n--- Test ---")
    datos_test = procesar_dataset(ruta_entrenamiento / "metadata_test_aumentado.csv", ruta_audios)
    if datos_test:
        ruta_out = ruta_entrenamiento / "test_mels_chunked_aumentado.pkl"
        with open(ruta_out, 'wb') as f:
            pickle.dump(datos_test, f)
        print(f"Guardado: {ruta_out}")

    print("\nPROCESO COMPLETADO.")
