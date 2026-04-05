import librosa
import pickle
import numpy as np
import pandas as pd
import torch
import os

from transformers import Wav2Vec2Processor, Wav2Vec2Model
from pathlib import Path

nombre_modelo = "facebook/wav2vec2-base-960h"

processor = Wav2Vec2Processor.from_pretrained(nombre_modelo)
modelo = Wav2Vec2Model.from_pretrained(nombre_modelo)

modelo.eval()

def preprocesado_basico(ruta_audio, target_dfbs=-20.0):
    y, sr = librosa.load(ruta_audio, sr=16000, mono=True)

    rms = np.sqrt(np.mean(y**2))

    rms_objetivo = 10 ** (target_dfbs / 20.0)

    y_normalizado = y * (rms_objetivo / rms)

    pico_maximo = np.max(np.abs(y_normalizado))
    if pico_maximo > 1.0:
        y_normalizado = y_normalizado / pico_maximo

    return y_normalizado, sr

def embedding_wav2vec2(ruta_audio):
    y, sr = preprocesado_basico(ruta_audio)
    inputs = processor(y, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        outputs = modelo(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

def procesar_dataset(ruta_csv, ruta_audios, ruta_salida):
    df = pd.read_csv(ruta_csv, sep=";")
    dataset = []
    for index, fila in df.iterrows():
        nombre_archivo = fila['nombre_archivo']
        ruta_audio = os.path.join(ruta_audios, nombre_archivo)

        try:
            features = embedding_wav2vec2(ruta_audio)
            
            paciente_columnas = {
                'nombre_archivo': nombre_archivo,
                'embedding': features,
                'caja_toracica': fila['caja_toracica'],
                'grupo': fila['grupo'],
                'fold': fila.get('fold', -1)
            }
            dataset.append(paciente_columnas)
            print(f"  -> OK. Dimensión Embedding: {features.shape} en {nombre_archivo}")
        except Exception as e:
            print(f"  -> ERROR al procesar {nombre_archivo}: {e}")
    
    with open(ruta_salida, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"\nProceso finalizado. Se han extraído características de {len(dataset)} audios.")

if __name__ == "__main__":
    ruta_base = Path(__file__).resolve().parent.parent
    ruta_entrenamiento = ruta_base / "datos_entrenamiento"
    ruta_audios = ruta_base / "audios"

    ruta_salida_train = ruta_base / "datos_entrenamiento" / "train_embeddings_wav2vec2.pkl"
    ruta_salida_test = ruta_base / "datos_entrenamiento" / "test_embeddings_wav2vec2.pkl"

    ruta_csv_train = ruta_entrenamiento / "metadata_train.csv"
    procesar_dataset(ruta_csv_train, ruta_audios, ruta_salida_train)

    ruta_csv_test = ruta_entrenamiento / "metadata_test.csv"
    procesar_dataset(ruta_csv_test, ruta_audios, ruta_salida_test)