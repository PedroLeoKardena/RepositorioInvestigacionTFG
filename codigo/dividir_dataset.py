import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from pathlib import Path

def crear_divisiones():
    ruta_base = Path(__file__).resolve().parent.parent
    ruta_csv = ruta_base / "datos_entrada.csv"
    ruta_salida = ruta_base / "datos_entrenamiento"

    ruta_salida.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(ruta_csv, encoding = "latin-1", sep=";")
        print(f"Archivo CSV '{ruta_csv}' leído correctamente.")
    except FileNotFoundError:
        print(f"Error: no se encontró la base de datos 'datos_entrada.csv'")
        return
    
    #Separamos las variables predictoras (X) de la variable objetivo (y)
    df_train, df_test = train_test_split(
        df, 
        test_size=0.2, 
        stratify=df["grupo"], 
        random_state=42
    )

    # Aplicamos el 5-Fold Cross Validation al conjunto de Train
    df_train = df_train.reset_index(drop=True)
    df_train['fold'] = -1

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  
    # Asignamos a cada audio de entrenamiento un fold
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df_train, df_train["grupo"])):
        df_train.loc[val_idx, 'fold'] = fold_idx

    #Guardamos los resultados
    df_train.to_csv(ruta_salida / "metadata_train.csv", index=False, sep = ";")
    df_test.to_csv(ruta_salida / "metadata_test.csv", index=False, sep = ";")
    
    print(f"\nDivisión del dataset realizada correctamente.")
    print(f"Tamaño del conjunto de entrenamiento: {len(df_train)}")
    print(f"Tamaño del conjunto de prueba: {len(df_test)}")
    
    return df_train, df_test

if __name__ == "__main__":
    crear_divisiones()