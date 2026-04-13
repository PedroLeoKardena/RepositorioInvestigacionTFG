import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import gc

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from datasets import Dataset
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    TrainingArguments,
    Trainer
)
from pathlib import Path

nombre_modelo = "facebook/wav2vec2-base-960h"

class Wav2Vec2MultiTask(nn.Module):
    def __init__(self, nombre_modelo, num_labels_grupo, num_labels_caja):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(nombre_modelo)
        
        hidden_size = self.wav2vec2.config.hidden_size
        
        self.classifier_grupo = nn.Linear(hidden_size, num_labels_grupo)
        self.classifier_caja = nn.Linear(hidden_size, num_labels_caja)

    def forward(self, input_values, **kwargs):
        outputs = self.wav2vec2(input_values)
        
        hidden_states = outputs.last_hidden_state
        pooled_output = hidden_states.mean(dim=1) 
        
        logits_grupo = self.classifier_grupo(pooled_output)
        logits_caja = self.classifier_caja(pooled_output)
        
        return {"logits_grupo": logits_grupo, "logits_caja": logits_caja}

class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_grupo = inputs.pop("labels_grupo")
        labels_caja = inputs.pop("labels_caja")
        
        outputs = model(inputs["input_values"])
        
        loss_fct = nn.CrossEntropyLoss()
        loss_grupo = loss_fct(outputs["logits_grupo"], labels_grupo)
        loss_caja = loss_fct(outputs["logits_caja"], labels_caja)
        
        loss = loss_grupo + loss_caja
        
        return (loss, outputs) if return_outputs else loss

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
        y_norm, _ = preprocesado_basico(ruta_audio)
        audio_arrays.append(y_norm)

    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=16000, 
        padding="max_length", 
        max_length=160000, # 10 segundos
        truncation=True
    )
    
    inputs["labels_grupo"] = batch["label_grupo"]
    inputs["labels_caja"] = batch["label_caja"]
    return inputs

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

    le_grupo = LabelEncoder()
    le_caja = LabelEncoder()

    df_train['label_grupo'] = le_grupo.fit_transform(df_train['grupo'])
    df_test['label_grupo'] = le_grupo.transform(df_test['grupo'])
    
    df_train['label_caja'] = le_caja.fit_transform(df_train['caja_toracica'])
    df_test['label_caja'] = le_caja.transform(df_test['caja_toracica'])
    
    num_labels_grupo = len(le_grupo.classes_)
    num_labels_caja = len(le_caja.classes_)
    
    print(f"Clases Grupo ({num_labels_grupo}):", list(le_grupo.classes_))
    print(f"Clases Caja Torácica ({num_labels_caja}):", list(le_caja.classes_))
    
    train_dataset = Dataset.from_pandas(df_train[['nombre_archivo', 'label_grupo', 'label_caja', 'fold']])
    test_dataset = Dataset.from_pandas(df_test[['nombre_archivo', 'label_grupo', 'label_caja']])

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

    cv_accuracies_grupo = []
    cv_accuracies_caja = []
    
    for fold_val in range(5):
        print(f"\n--- Iniciando Entrenamiento Fold {fold_val}/4 ---")
        
        train_fold_ds = train_dataset.filter(lambda example: example['fold'] != fold_val)
        val_fold_ds = train_dataset.filter(lambda example: example['fold'] == fold_val)
        
        train_fold_ds = train_fold_ds.remove_columns(['fold'])
        val_fold_ds = val_fold_ds.remove_columns(['fold'])
        
        modelo_cv = Wav2Vec2MultiTask(nombre_modelo, num_labels_grupo, num_labels_caja)
        
        training_args_cv = TrainingArguments(
            output_dir=str(ruta_modelos / f"fold_{fold_val}"),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=3e-5,
            per_device_train_batch_size=4, 
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            num_train_epochs=3,
            logging_steps=10,
            remove_unused_columns=False,
        )

        trainer_cv = MultiTaskTrainer(
            model=modelo_cv,
            args=training_args_cv,
            train_dataset=train_fold_ds,
            eval_dataset=val_fold_ds,
        )

        trainer_cv.train()

        modelo_cv.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        modelo_cv.to(device)
        
        correctos_grupo, correctos_caja = 0, 0
        total = len(val_fold_ds)

        with torch.no_grad():
            for i in range(total):
                inputs = torch.tensor(val_fold_ds[i]['input_values']).unsqueeze(0).to(device)
                outputs = modelo_cv(inputs)
                
                pred_grupo = torch.argmax(outputs['logits_grupo'], dim=-1).item()
                pred_caja = torch.argmax(outputs['logits_caja'], dim=-1).item()
                
                if pred_grupo == val_fold_ds[i]['label_grupo']: correctos_grupo += 1
                if pred_caja == val_fold_ds[i]['label_caja']: correctos_caja += 1

        acc_grupo = correctos_grupo / total
        acc_caja = correctos_caja / total
        cv_accuracies_grupo.append(acc_grupo)
        cv_accuracies_caja.append(acc_caja)
        
        print(f"Resultados Fold {fold_val} -> Accuracy Grupo: {acc_grupo:.4f} | Accuracy Caja: {acc_caja:.4f}")
        
        # Limpieza de memoria GPU
        del trainer_cv, modelo_cv
        torch.cuda.empty_cache()
        gc.collect()

    print(f"Precisión Media Grupo: {np.mean(cv_accuracies_grupo):.4f} (+/- {np.std(cv_accuracies_grupo):.4f})")
    print(f"Precisión Media Caja: {np.mean(cv_accuracies_caja):.4f} (+/- {np.std(cv_accuracies_caja):.4f})")

    print("Iniciando Entrenamiento Final del Modelo con TODOS los datos de Train...")
    train_final_ds = train_dataset.remove_columns(['fold'])
    
    modelo_final = Wav2Vec2MultiTask(nombre_modelo, num_labels_grupo, num_labels_caja)

    training_args_final = TrainingArguments(
        output_dir=str(ruta_modelos / "entrenamiento_final_multitask"),
        eval_strategy="no",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=2,
        num_train_epochs=3, #Modificable numero de epochs
        logging_steps=10,
        remove_unused_columns=False,
    )

    trainer_final = MultiTaskTrainer(
        model=modelo_final,
        args=training_args_final,
        train_dataset=train_final_ds,
    )

    trainer_final.train()

    print("\nEvaluando Modelo Final sobre el conjunto de TEST...")
    modelo_final.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    modelo_final.to(device)
    
    preds_grupo_list, preds_caja_list = [], []
    real_grupo_list, real_caja_list = test_dataset['label_grupo'], test_dataset['label_caja']

    with torch.no_grad():
        for i in range(len(test_dataset)):
            inputs = torch.tensor(test_dataset[i]['input_values']).unsqueeze(0).to(device)
            outputs = modelo_final(inputs)
            
            pred_grupo = torch.argmax(outputs['logits_grupo'], dim=-1).item()
            pred_caja = torch.argmax(outputs['logits_caja'], dim=-1).item()
            
            preds_grupo_list.append(pred_grupo)
            preds_caja_list.append(pred_caja)

    print("\nReporte Final - GRUPO:\n")
    print(classification_report(real_grupo_list, preds_grupo_list, target_names=le_grupo.classes_, zero_division=0))

    print("\nReporte Final - CAJA TORÁCICA:\n")
    etiquetas_caja = np.arange(len(le_caja.classes_))
    
    print(classification_report(
        real_caja_list, 
        preds_caja_list, 
        labels=etiquetas_caja,           
        target_names=le_caja.classes_, 
        zero_division=0
    ))
    
    ruta_guardado_final = ruta_modelos / "modelo_multitask_wav2vec2"
    os.makedirs(ruta_guardado_final, exist_ok=True)
    
    torch.save(modelo_final.state_dict(), ruta_guardado_final / "pytorch_model.bin")
    feature_extractor.save_pretrained(str(ruta_guardado_final))
    
    np.save(ruta_guardado_final / "label_classes_grupo.npy", le_grupo.classes_)
    np.save(ruta_guardado_final / "label_classes_caja.npy", le_caja.classes_)
    
    print(f"\nProceso completado. Modelo final guardado en '{ruta_guardado_final}'")


if __name__ == "__main__":
    #Evitamos advertencias
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    entrenar_modelo()