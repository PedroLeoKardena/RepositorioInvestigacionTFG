import pandas as pd
from pathlib import Path

def verificar_lectura_audios():
    # Ruta del archivo CSV
    ruta_base = Path(__file__).resolve().parent.parent 
    ruta_csv = ruta_base / "datos_entrada.csv"
    ruta_audios = ruta_base / "audios_originales"

    try:
        df = pd.read_csv(ruta_csv, encoding = "latin-1", sep="\t")
        print(f"Archivo CSV '{ruta_csv}' leído correctamente.")
    except FileNotFoundError:    
        print(f"Error: no se encontró la base de datos 'datos_entrada.csv'")
        return
    
    encontrados = 0
    faltantes = []
    
    for index, row in df.iterrows():
        nombre_audio = row['nombre_archivo']
        ruta_audio = ruta_audios / nombre_audio

        if ruta_audio.is_file():
            encontrados += 1
        else:
            faltantes.append(nombre_audio)
    
    print(df.head())
    print(f"Archivos encontrados: {encontrados}")
    print(f"Archivos faltantes: {len(faltantes)}")
    if faltantes:
        print("Archivos faltantes:")
        for nombre in faltantes:
            print(f" - {nombre}")
    

if __name__ == "__main__":
    verificar_lectura_audios()