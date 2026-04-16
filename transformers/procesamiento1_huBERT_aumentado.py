import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import gc
import json
from datetime import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from datasets import Dataset
from transformers import (
    AutoFeatureExtractor, 
    HubertModel,          
    TrainingArguments,
    Trainer
)
from pathlib import Path

# Usamos el modelo base de HuBERT de HuggingFace
nombre_modelo = "facebook/hubert-base-ls960"

class HubertMultiTask(nn.Module):
    def __init__(self, nombre_modelo, num_labels_grupo, num_labels_caja):
        super().__init__()
        # Usamos HubertModel en lugar de Wav2Vec2Model
        self.hubert = HubertModel.from_pretrained(nombre_modelo, use_safetensors=True)        
        hidden_size = self.hubert.config.hidden_size
        
        self.classifier_grupo = nn.Linear(hidden_size, num_labels_grupo)
        self.classifier_caja = nn.Linear(hidden_size, num_labels_caja)

    def forward(self, input_values, **kwargs):
        outputs = self.hubert(input_values)
        
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
feature_extractor = AutoFeatureExtractor.from_pretrained(nombre_modelo)

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
        
        modelo_cv = HubertMultiTask(nombre_modelo, num_labels_grupo, num_labels_caja)
        
        training_args_cv = TrainingArguments(
            output_dir=str(ruta_modelos / f"hubert_fold_{fold_val}"),
            eval_strategy="epoch",
            save_strategy="no",
            learning_rate=5e-5,
            per_device_train_batch_size=4, 
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            num_train_epochs=10,
            weight_decay=0.01,
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
    
    modelo_final = HubertMultiTask(nombre_modelo, num_labels_grupo, num_labels_caja)

    training_args_final = TrainingArguments(
        output_dir=str(ruta_modelos / "entrenamiento_final_multitask_hubert"),
        eval_strategy="no",
        save_strategy="no",
        learning_rate=5e-5,
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=2,
        num_train_epochs=10, #Modificable numero de epochs
        weight_decay=0.01,
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
    reporte_grupo_str = classification_report(real_grupo_list, preds_grupo_list, target_names=le_grupo.classes_, zero_division=0)
    reporte_grupo_dict = classification_report(real_grupo_list, preds_grupo_list, target_names=le_grupo.classes_, zero_division=0, output_dict=True)
    print(reporte_grupo_str)

    print("\nReporte Final - CAJA TORÁCICA:\n")
    etiquetas_caja = np.arange(len(le_caja.classes_))
    
    reporte_caja_str = classification_report(
        real_caja_list, 
        preds_caja_list, 
        labels=etiquetas_caja,           
        target_names=le_caja.classes_, 
        zero_division=0
    )
    reporte_caja_dict = classification_report(
        real_caja_list, 
        preds_caja_list, 
        labels=etiquetas_caja,           
        target_names=le_caja.classes_, 
        zero_division=0,
        output_dict=True
    )
    print(reporte_caja_str)
    
    resultados = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modelo": nombre_modelo,
        "hiperparametros": {
            "learning_rate": training_args_final.learning_rate,
            "per_device_train_batch_size": training_args_final.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args_final.gradient_accumulation_steps,
            "num_train_epochs": training_args_final.num_train_epochs,
            "weight_decay": training_args_final.weight_decay
        },
        "resultados_cv": {
            "media_grupo": float(np.mean(cv_accuracies_grupo)),
            "std_grupo": float(np.std(cv_accuracies_grupo)),
            "media_caja": float(np.mean(cv_accuracies_caja)),
            "std_caja": float(np.std(cv_accuracies_caja))
        },
        "reporte_final_grupo": reporte_grupo_dict,
        "reporte_final_caja": reporte_caja_dict
    }

    ruta_log_dir = ruta_modelos / "resultados_json"
    os.makedirs(ruta_log_dir, exist_ok=True)
    ruta_log = ruta_log_dir / "registro_resultados.json"
    
    if ruta_log.exists():
        with open(ruta_log, 'r', encoding='utf-8') as f:
            log_historico = json.load(f)
    else:
        log_historico = []
        
    log_historico.append(resultados)
    
    with open(ruta_log, 'w', encoding='utf-8') as f:
        json.dump(log_historico, f, indent=4, ensure_ascii=False)
        
    print(f"\nResultados e hiperparámetros guardados en: {ruta_log}")
    # ---------------------------------------------------------
    
    ruta_guardado_final = ruta_modelos / "modelo_multitask_hubert"
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
