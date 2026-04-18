import librosa
import pickle
import pandas as pd
from pathlib import Path
import numpy as np
import warnings

# Número de frames objetivo: chunk de 10s a sr=16000, hop_length=160, center=True
# n_frames = 1 + (10 * 16000) // 160 = 1001
TARGET_FRAMES = 1001

def extraccion_mfccs(y, sr, n_mfcc=30):
    """
    Para la extracción de los coeficientes MFCCS, utilizamos la función librosa.feature.mfcc.
    Esta función toma el audio preprocesado y devuelve un array de coeficientes MFCCS.
    Utilizaremos los siguientes parámetros:
        -30 coeficientes MFCCS (n_mfcc=30)
        -Ventanas de 25 ms
        -Saltos de 10 ms
    """
    mfccs = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=512,
        win_length=400,
        hop_length=160,
        window='hann'
    )

    # Padding/truncado para garantizar dimensionalidad uniforme entre augmentations.
    # El time stretch reduce la duración del audio, generando menos frames que el original.
    n_frames = mfccs.shape[1]
    if n_frames < TARGET_FRAMES:
        mfccs = np.pad(mfccs, ((0, 0), (0, TARGET_FRAMES - n_frames)), mode='constant')
    elif n_frames > TARGET_FRAMES:
        mfccs = mfccs[:, :TARGET_FRAMES]

    return mfccs

def procesar_dataset(ruta_csv, ruta_audios):
    try:
        df = pd.read_csv(ruta_csv, encoding="utf-8", sep=";")
        print(f"Archivo CSV '{ruta_csv.name}' leído correctamente.")
    except FileNotFoundError:    
        print(f"Error: no se encontró la base de datos '{ruta_csv.name}'")
        return []
    

    dataset_procesado = []
    for index, fila in df.iterrows():
        nombre_archivo = fila['nombre_archivo']
        grupo = fila['grupo']
        
        caja_toracica = fila['caja_toracica']
        fold = fila.get('fold', -1)

        ruta_audio_especifico = ruta_audios / nombre_archivo
        
        if ruta_audio_especifico.exists():
            try:
                # Los audios ya están preprocesados
                y_norm, sr = librosa.load(ruta_audio_especifico, sr=None)
                
                mfccs = extraccion_mfccs(y_norm, sr)
                
                datos_paciente = {
                    'nombre_archivo': nombre_archivo,
                    'mfccs': mfccs,
                    'caja_toracica': caja_toracica,
                    'grupo': grupo,
                    'fold': fold
                }
                dataset_procesado.append(datos_paciente)
                
                print(f"  -> OK. Dimensión MFCC: {mfccs.shape} en {nombre_archivo}")
            except Exception as e:
                print(f"  -> ERROR al procesar {nombre_archivo}: {e}")
        else:
            print(f"Archivo no encontrado: {nombre_archivo}")
        
    print(f"\nProceso finalizado. Se han extraído características de {len(dataset_procesado)} audios.")

    return dataset_procesado
    
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ruta_base = Path(__file__).resolve().parent.parent
    ruta_entrenamiento = ruta_base / "datos_entrenamiento"
    ruta_audios = ruta_base / "audios_aumentados"
    
    ruta_csv_train = ruta_entrenamiento / "metadata_train_aumentado.csv"
    datos_mfcc_train = procesar_dataset(ruta_csv_train, ruta_audios)

    if datos_mfcc_train:
        ruta_dest_train = ruta_entrenamiento / "train_mfcc_chunked_aumentado.pkl"
        with open(ruta_dest_train, 'wb') as f:
            pickle.dump(datos_mfcc_train, f)
        print(f"Datos MFCC entrenamiento guardados en: {ruta_dest_train}")
    
    ruta_csv_test = ruta_entrenamiento / "metadata_test_aumentado.csv"
    datos_mfcc_test = procesar_dataset(ruta_csv_test, ruta_audios)

    if datos_mfcc_test:
        ruta_dest_test = ruta_entrenamiento / "test_mfcc_chunked_aumentado.pkl"
        with open(ruta_dest_test, 'wb') as f:
            pickle.dump(datos_mfcc_test, f)
        print(f"Datos MFCC test guardados en: {ruta_dest_test}")
