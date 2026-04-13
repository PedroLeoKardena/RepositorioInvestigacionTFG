import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold
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
    

    y_strat = pd.get_dummies(df[['grupo', 'caja_toracica']]).values

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    train_idx, test_idx = next(msss.split(df, y_strat))
    
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    df_train['fold'] = -1
    
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  
    # Volvemos a hacer las dummies solo para el conjunto de entrenamiento
    y_train_strat = pd.get_dummies(df_train[['grupo', 'caja_toracica']]).values
    
    # Asignamos a cada audio de entrenamiento un fold
    for fold_idx, (_, val_idx) in enumerate(mskf.split(df_train, y_train_strat)):
        df_train.loc[val_idx, 'fold'] = fold_idx

    #Guardamos los resultados
    df_train.to_csv(ruta_salida / "metadata_train.csv", index=False, sep=";")
    df_test.to_csv(ruta_salida / "metadata_test.csv", index=False, sep=";")
    
    print(f"\nDivisión del dataset realizada correctamente (Estratificación Iterativa).")
    print(f"Tamaño del conjunto de entrenamiento: {len(df_train)}")
    print(f"Tamaño del conjunto de prueba: {len(df_test)}")
    
    return df_train, df_test

if __name__ == "__main__":
    crear_divisiones()