import os
import librosa
import numpy as np
import gc
import soundfile as sf
import pandas as pd
from pathlib import Path

def apply_augmentations(y, sr):
    augmentations = {}
    
    # 1. Ruido Gaussiano
    ruido = np.random.randn(len(y))
    y_noise = y + 0.005 * ruido 
    augmentations['noise'] = y_noise
    
    # 2. Pitch Shift (+/- 2 semitonos, simula distinta fisionomía)
    print("   -> Aplicando Pitch Shift...")
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2.0)
    augmentations['pitch'] = y_pitch
    
    # 3. Time Stretch (Simula taquipnea un 15% más rápida)
    print("   -> Aplicando Time Stretch...")
    y_stretch = librosa.effects.time_stretch(y, rate=1.15)
    augmentations['stretch'] = y_stretch
    
    return augmentations

def procesar_dataset_completo(ruta_input, ruta_output):
    os.makedirs(ruta_output, exist_ok=True)
    
    archivos = [f for f in os.listdir(ruta_input) if f.endswith(('.wav', '.m4a', '.mp3'))]
    total_archivos = len(archivos)
    print(f"Se encontraron {total_archivos} audios para procesar.\n")
    
    for idx, archivo in enumerate(archivos):
        print(f"[{idx+1}/{total_archivos}] Procesando: {archivo}")
        ruta_completa = os.path.join(ruta_input, archivo)
        nombre_base = os.path.splitext(archivo)[0]
        
        try:
            # Los chunks ya están preprocesados: carga directa
            y_base, sr = librosa.load(ruta_completa, sr=None)
            
            # Guardar la versión Original Preprocesada
            nombre_orig = f"{nombre_base}_original.wav"
            sf.write(os.path.join(ruta_output, nombre_orig), y_base, sr)
            
            # 2. Generar Augmentations
            augs = apply_augmentations(y_base, sr)
            
            # 3. Guardar las versiones aumentadas
            for tipo, y_aug in augs.items():
                pico = np.max(np.abs(y_aug))
                if pico > 1.0:
                    y_aug = y_aug / pico
                    
                nombre_aug = f"{nombre_base}_{tipo}.wav"
                sf.write(os.path.join(ruta_output, nombre_aug), y_aug, sr)

            print(f"Generados 4 archivos para {nombre_base}")

            del y_base, augs
            gc.collect()
            
        except Exception as e:
            print(f"[ERROR] Fallo procesando {archivo}: {e}")

def expandir_metadata_aumentada(ruta_base):
    ruta_entrenamiento = ruta_base / "datos_entrenamiento"
    ruta_csv_train = ruta_entrenamiento / "metadata_train_chunked.csv"
    ruta_csv_test = ruta_entrenamiento / "metadata_test_chunked.csv"

    try:
        df_train_orig = pd.read_csv(ruta_csv_train, sep=";")
        df_test_orig = pd.read_csv(ruta_csv_test, sep=";")
    except FileNotFoundError:
        print("Error: No se encontraron los archivos metadata_train.csv o metadata_test.csv.")
        return

    sufijos_train = ["original", "noise", "pitch", "stretch"]
    sufijos_test = ["original"]

    def expandir_df(df_original, listado_sufijos):
        filas_expandidas = []
        for _, row in df_original.iterrows():
            nombre_archivo_completo = str(row['nombre_archivo'])
            nombre_base = nombre_archivo_completo.rsplit('.', 1)[0]
            
            for sufijo in listado_sufijos:
                nueva_fila = row.copy()
                nueva_fila['nombre_archivo'] = f"{nombre_base}_{sufijo}.wav"
                filas_expandidas.append(nueva_fila)
                
        return pd.DataFrame(filas_expandidas)

    df_train_aug = expandir_df(df_train_orig, sufijos_train)
    df_test_aug = expandir_df(df_test_orig, sufijos_test)

    # Guardamos
    df_train_aug.to_csv(ruta_entrenamiento / "metadata_train_aumentado.csv", index=False, sep=";")
    df_test_aug.to_csv(ruta_entrenamiento / "metadata_test_aumentado.csv", index=False, sep=";")

    print("\n¡Metadatos expandidos correctamente!")
    print(f"Train original: {len(df_train_orig)} audios -> Train expandido (Aumentado): {len(df_train_aug)} audios.")
    print(f"Test original:  {len(df_test_orig)} audios -> Test expandido (Solo Original): {len(df_test_aug)} audios.")

if __name__ == "__main__":
    ruta_base = Path(__file__).resolve().parent.parent
    RUTA_ENTRADA = ruta_base / "audios_chunks"
    RUTA_SALIDA = ruta_base / "audios_aumentados"
    
    print("--- 1. GENERANDO ARCHIVOS DE AUDIO AUMENTADOS ---")
    procesar_dataset_completo(str(RUTA_ENTRADA), str(RUTA_SALIDA))
    
    print("\n--- 2. ACTUALIZANDO METADATOS ---")
    expandir_metadata_aumentada(ruta_base)
    
    print("\nPROCESO DE AUGMENTATION COMPLETADO.")
