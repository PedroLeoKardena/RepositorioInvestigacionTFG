import librosa
import pickle
import numpy as np
import pandas as pd
import torch
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from datasets import Dataset
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer
)
from pathlib import Path

nombre_modelo = "facebook/wav2vec2-base-960h"


def preprocesado_basico(ruta_audio, target_dfbs=-20.0):
    y, sr = librosa.load(ruta_audio, sr=16000, mono=True)

    rms = np.sqrt(np.mean(y**2))

    rms_objetivo = 10 ** (target_dfbs / 20.0)

    y_normalizado = y * (rms_objetivo / rms)

    pico_maximo = np.max(np.abs(y_normalizado))
    if pico_maximo > 1.0:
        y_normalizado = y_normalizado / pico_maximo

    return y_normalizado, sr

# Inicializamos el feature extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(nombre_modelo)

def preprocesar_batch(batch, ruta_audios):
    audio_arrays = []
    
    for nombre_archivo in batch["nombre_archivo"]:
        ruta_audio = os.path.join(ruta_audios, nombre_archivo)
        # Reutilizamos el preprocesado básico (RMS normalization)
        y_norm, _ = preprocesado_basico(ruta_audio)
        audio_arrays.append(y_norm)

    # El feature extractor rellena o trunca a la longitud máxima especificada.
    # 160000 muestras equivalen a 10 segundos a 16kHz. 
    # Es vital truncar porque audios de 10 minutos provocarían OutOfMemory en la RAM de la GPU.
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=16000, 
        padding="max_length", 
        max_length=160000, # 10 segundos
        truncation=True
    )
    
    # HF Dataset requiere las etiquetas bajo la clave 'labels'
    inputs["labels"] = batch["label"]
    return inputs

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

def entrenar_modelo():
    ruta_base = Path(__file__).resolve().parent.parent
    ruta_entrenamiento = ruta_base / "datos_entrenamiento"
    ruta_audios = ruta_base / "audios"
    ruta_modelos = ruta_base / "modelos_entrenados"
    
    ruta_csv_train = ruta_entrenamiento / "metadata_train.csv"
    ruta_csv_test = ruta_entrenamiento / "metadata_test.csv"

    try:
        df_train = pd.read_csv(ruta_csv_train, sep=";")
        df_test = pd.read_csv(ruta_csv_test, sep=";")
    except FileNotFoundError:
        print("Archivos de metadata no encontrados. Por favor, asegúrate de haber ejecutado previamente dividir_dataset.py")
        return

    df_train['clase_combinada'] = df_train['grupo'].astype(str) + "_" + df_train['caja_toracica'].astype(str)
    df_test['clase_combinada'] = df_test['grupo'].astype(str) + "_" + df_test['caja_toracica'].astype(str)

    le = LabelEncoder()
    df_train['label'] = le.fit_transform(df_train['clase_combinada'])
    df_test['label'] = le.transform(df_test['clase_combinada'])
    
    num_labels = len(le.classes_)
    print(f"Clases detectadas ({num_labels}):", list(le.classes_))
    
    train_dataset = Dataset.from_pandas(df_train[['nombre_archivo', 'label']])
    test_dataset = Dataset.from_pandas(df_test[['nombre_archivo', 'label']])

    print("Preprocesando conjunto de entrenamiento (esto podría tardar unos minutos)...")
    train_dataset = train_dataset.map(
        lambda batch: preprocesar_batch(batch, ruta_audios),
        batched=True,
        batch_size=8,
        remove_columns=['nombre_archivo']
    )

    print("Preprocesando conjunto de test...")
    test_dataset = test_dataset.map(
        lambda batch: preprocesar_batch(batch, ruta_audios),
        batched=True,
        batch_size=8,
        remove_columns=['nombre_archivo']
    )

    print("Cargando el modelo...")
    modelo = Wav2Vec2ForSequenceClassification.from_pretrained(
        nombre_modelo,
        num_labels=num_labels
    )
    
    training_args = TrainingArguments(
        output_dir=str(ruta_modelos),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4, 
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=modelo,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    print("Iniciando el entrenamiento End2End...")
    trainer.train()

    print("Evaluando sobre el conjunto de test...")
    resultados = trainer.evaluate()
    print("Resultados de la evaluación:", resultados)

    preds_output = trainer.predict(test_dataset)
    predicciones_etiquetas = np.argmax(preds_output.predictions, axis=1)
    
    todas_las_etiquetas = np.arange(len(le.classes_))
    print("\nReporte Final de Clasificación:\n")
    print(classification_report(
        test_dataset['labels'], 
        predicciones_etiquetas, 
        labels=todas_las_etiquetas,     
        target_names=le.classes_,       
        zero_division=0                 
    ))

    ruta_guardado_final = ruta_modelos / "modelo_final"
    trainer.save_model(str(ruta_guardado_final))
    feature_extractor.save_pretrained(str(ruta_guardado_final))
    
    os.makedirs(ruta_guardado_final, exist_ok=True)
    np.save(ruta_guardado_final / "label_classes.npy", le.classes_)
    print(f"\nModelo final y Feature Extractor guardados con éxito en '{ruta_guardado_final}'")


if __name__ == "__main__":
    entrenar_modelo()