import pandas as pd
import numpy as np
from pathlib import Path


def analizar_distribucion(df, nombre_split):
    print(f"\n{'='*60}")
    print(f"  {nombre_split}")
    print(f"{'='*60}")
    print(f"Total chunks: {len(df)}")
    print(f"Pacientes únicos: {df['audio_original'].nunique()}")

    for tarea in ['grupo', 'caja_toracica']:
        print(f"\n--- Por {tarea} ---")
        resumen = (
            df.groupby(tarea)
            .agg(
                chunks=('nombre_archivo', 'count'),
                pacientes=('audio_original', 'nunique')
            )
            .assign(
                pct_chunks=lambda x: (x['chunks'] / x['chunks'].sum() * 100).round(1),
                chunks_por_paciente=lambda x: (x['chunks'] / x['pacientes']).round(1)
            )
            .sort_values('chunks', ascending=False)
        )
        print(resumen.to_string())

    print(f"\n--- Chunks por paciente (top 10 más largos) ---")
    por_paciente = (
        df.groupby(['audio_original', 'grupo', 'caja_toracica'])
        .size()
        .reset_index(name='num_chunks')
        .sort_values('num_chunks', ascending=False)
        .head(10)
    )
    print(por_paciente.to_string(index=False))


def analizar_folds(df_train):
    print(f"\n{'='*60}")
    print(f"  DISTRIBUCIÓN POR FOLDS (train)")
    print(f"{'='*60}")
    for fold in sorted(df_train['fold'].unique()):
        fold_df = df_train[df_train['fold'] == fold]
        print(f"\nFold {fold}: {len(fold_df)} chunks ({fold_df['audio_original'].nunique()} pacientes)")
        dist = fold_df['grupo'].value_counts()
        for clase, n in dist.items():
            print(f"  {clase:<35} {n:>4} chunks")


if __name__ == "__main__":
    ruta_base = Path(__file__).resolve().parent.parent
    ruta_entrenamiento = ruta_base / "datos_entrenamiento"

    ruta_train = ruta_entrenamiento / "metadata_train_chunked.csv"
    ruta_test = ruta_entrenamiento / "metadata_test_chunked.csv"

    try:
        df_train = pd.read_csv(ruta_train, sep=";", encoding="utf-8")
        df_test = pd.read_csv(ruta_test, sep=";", encoding="utf-8")
    except FileNotFoundError as e:
        print(f"Error: {e}\nAsegúrate de haber ejecutado chunking_audios.py primero.")
        exit(1)

    analizar_distribucion(df_train, "TRAIN")
    analizar_distribucion(df_test, "TEST")
    analizar_folds(df_train)

    print(f"\n{'='*60}")
    print("  COMPARATIVA TRAIN vs TEST (% chunks por clase)")
    print(f"{'='*60}")
    for tarea in ['grupo', 'caja_toracica']:
        print(f"\n--- {tarea} ---")
        pct_train = (df_train[tarea].value_counts(normalize=True) * 100).round(1).rename("train %")
        pct_test = (df_test[tarea].value_counts(normalize=True) * 100).round(1).rename("test %")
        comparativa = pd.concat([pct_train, pct_test], axis=1).fillna(0)
        comparativa['diferencia'] = (comparativa['train %'] - comparativa['test %']).round(1)
        print(comparativa.to_string())
