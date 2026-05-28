import librosa

import pickle
import pandas as pd
from pathlib import Path
import numpy as np
import warnings

#Tal y como se dice en la siguiente conversación de stackoverflow: https://stackoverflow.com/questions/62584184/understanding-the-shape-of-spectrograms-and-n-mels

#Nosotros podemos llegar a conocer las dimensiones de nuestro array espectrograma,
#conociendo el tiempo de audios, la frecuencia de sampling y el hop_length.

#Nosotros trabajamos con audios de 10 segundos, sampleados a 16000 Hz. Esto nos da un total de 160000 muestras por audio 
#(10 segundos * 16000 muestras/segundo).

#Las dimensiones entonces del array del espectrograma las calcularemos dividiendo el numero de muestras por el hop_legth.
#Si nosotros establecemos el hop_length de base (512) segun https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html 
#Obtendremos un array de 313 columnas.
#Estos espectrogramas los usaremos para entrenar modelos de Deep Learning (redes neuronales).

#Todo modelo de Deep Learning puede trabajar con entradas que no sean de un tamaño (nxn). Pueden ser de cualquier tamaño.
#En muchos modelos secuenciales de Deep Learning (RNN, LSTM, GRU) y transformers como whisper se suele elegir 
#un valor de hop_length dependiendo en sí de los milisegundos que quiere que dure cada salto, al igual que para window.

#El modelo whisper: https://openwhispr.com/blog/how-whisper-ai-works crea saltos de 10 ms y ventanas de 25 ms.

#Para obtener estos valores estableceremos el hop_length a 160 muestras (10 ms) y el window a 400 muestras (25 ms). Como con los mfcc
#n_fft los estableceremos a 512 muestras

#Esto nos permitirá compararlo directamente con MFCC, al haber establecido mismos parametros.
#Con esto tendremos entonces un array de espectrograma con 128 filas (mel_bands) y 160000/160 = 1000 columnas (frames). Esto nos da un array de 128x1000 por cada audio chunkeado.

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
    ruta_audios = ruta_base / "audios_chunks"

    print("=== EXTRACCIÓN DE MEL SPECTROGRAMS — AUDIOS CHUNKEADOS ===\n")

    print("--- Train ---")
    datos_train = procesar_dataset(ruta_entrenamiento / "metadata_train_chunked.csv", ruta_audios)
    if datos_train:
        ruta_out = ruta_entrenamiento / "train_mels_chunked.pkl"
        with open(ruta_out, 'wb') as f:
            pickle.dump(datos_train, f)
        print(f"Guardado: {ruta_out}")

    print("\n--- Test ---")
    datos_test = procesar_dataset(ruta_entrenamiento / "metadata_test_chunked.csv", ruta_audios)
    if datos_test:
        ruta_out = ruta_entrenamiento / "test_mels_chunked.pkl"
        with open(ruta_out, 'wb') as f:
            pickle.dump(datos_test, f)
        print(f"Guardado: {ruta_out}")

    print("\nPROCESO COMPLETADO.")
