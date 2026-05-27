import librosa
import pandas as pd
from pathlib import Path
import numpy as np

def preprocesado_basico(ruta_audio, target_dfbs=-20.0):
    """
    Para preprocesar el audio formato .m4a, vamos a llevar a cabo lo 4 siguientes pasos:
        -Cargar el audio
        -Convertir a mono
        -Remuestrear a 16kHz
        -Aplicar normalización RMS
    Con esto ya podremos posteriormente extraer los coeficientes MFCCS.
    """

    #Conversión a mono y remuestreo a 16kHz.
    #librosa.load delega la decodificación del.m4a a FFmpeg por debajo, lo que nos permite cargar el audio directamente en el formato deseado.
    y, sr = librosa.load(ruta_audio, sr=16000, mono=True)

    #Normalización RMS
    #Para la normalización RMS, calculamos la energía media global usando NumPy:
    rms = np.sqrt(np.mean(y**2))

    # Convertimos nuestro objetivo en decibelios (dBFS) a una escala de amplitud lineal
    rms_objetivo = 10 ** (target_dfbs / 20.0)

    # Escalamos el tensor de audio original para que coincida con el RMS objetivo
    y_normalizado = y * (rms_objetivo / rms)

    #Si la ganancia RMS es demasiado alta, podríamos saturar el audio. Para evitar esto, limitamos la amplitud máxima a 1.0 (o -1.0 para el lado negativo):
    pico_maximo = np.max(np.abs(y_normalizado))
    if pico_maximo > 1.0:
        y_normalizado = y_normalizado / pico_maximo

    return y_normalizado, sr


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

    return mfccs

def procesar_dataset():
    #Cargamos base de datos
    ruta_base = Path(__file__).resolve().parent.parent 
    ruta_csv = ruta_base / "datos_entrada.csv"
    ruta_audios = ruta_base / "audios"

    try:
        df = pd.read_csv(ruta_csv, encoding = "latin-1", sep="\t")
        print(f"Archivo CSV '{ruta_csv}' leído correctamente.")
    except FileNotFoundError:    
        print(f"Error: no se encontró la base de datos 'datos_entrada.csv'")
        return
    
    dataset_procesado = []
    for index, fila in df.iterrows():
        nombre_archivo = fila['nombre_archivo']
        caja_toracica = fila['caja_toracica']
        grupo = fila['grupo']
        
        ruta_audio_especifico = ruta_audios / nombre_archivo
        
        if ruta_audio_especifico.exists():
            try:
                # 1. Aplicamos tu preprocesado
                y_norm, sr = preprocesado_basico(ruta_audio_especifico)
                
                # 2. Extraemos los MFCCs
                mfccs = extraccion_mfccs(y_norm, sr)
                
                # 3. Guardamos todo junto en un diccionario
                datos_paciente = {
                    'nombre_archivo': nombre_archivo,
                    'mfccs': mfccs,
                    'caja_toracica': caja_toracica,
                    'grupo': grupo
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
    datos_mfcc = procesar_dataset()
