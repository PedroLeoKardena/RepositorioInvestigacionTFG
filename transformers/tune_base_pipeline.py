import librosa
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import gc

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from abc import ABC, abstractmethod
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoFeatureExtractor,
    HubertModel,
    Wav2Vec2Model,
    TrainingArguments,
    Trainer
)

class HubertMultiTask(nn.Module):
    def __init__(self, nombre_modelo, num_labels_grupo, num_labels_caja):
        super().__init__()
        self.hubert = HubertModel.from_pretrained(nombre_modelo, use_safetensors=True)
        for param in self.hubert.feature_extractor.parameters():
            param.requires_grad = False

        hidden_size = self.hubert.config.hidden_size
        self.classifier_grupo = nn.Linear(hidden_size, num_labels_grupo)
        self.classifier_caja = nn.Linear(hidden_size, num_labels_caja)

    def forward(self, input_values, attention_mask=None, **kwargs):
        outputs = self.hubert(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled_output = hidden_states.mean(dim=1)
        logits_grupo = self.classifier_grupo(pooled_output)
        logits_caja = self.classifier_caja(pooled_output)
        return {"logits_grupo": logits_grupo, "logits_caja": logits_caja}


class Wav2Vec2MultiTask(nn.Module):
    def __init__(self, nombre_modelo, num_labels_grupo, num_labels_caja):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(nombre_modelo, use_safetensors=True)
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False

        hidden_size = self.wav2vec2.config.hidden_size
        self.classifier_grupo = nn.Linear(hidden_size, num_labels_grupo)
        self.classifier_caja = nn.Linear(hidden_size, num_labels_caja)

    def forward(self, input_values, attention_mask=None, **kwargs):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled_output = hidden_states.mean(dim=1)
        logits_grupo = self.classifier_grupo(pooled_output)
        logits_caja = self.classifier_caja(pooled_output)
        return {"logits_grupo": logits_grupo, "logits_caja": logits_caja}


class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_grupo = inputs.pop("labels_grupo")
        labels_caja = inputs.pop("labels_caja")
        outputs = model(inputs["input_values"], attention_mask=inputs.get("attention_mask"))
        loss_fct = nn.CrossEntropyLoss()
        loss_grupo = loss_fct(outputs["logits_grupo"], labels_grupo)
        loss_caja = loss_fct(outputs["logits_caja"], labels_caja)
        loss = loss_grupo + loss_caja
        return (loss, outputs) if return_outputs else loss


class BaseTransformerPipeline(ABC):
    @property
    @abstractmethod
    def nombre_dataset(self) -> str:
        pass

    @property
    @abstractmethod
    def nombre_modelo(self) -> str:
        pass

    @property
    @abstractmethod
    def ruta_audios(self) -> str:
        pass

    @property
    @abstractmethod
    def learning_rate(self) -> float:
        pass

    @property
    @abstractmethod
    def batch_size(self) -> int:
        pass

    @property
    @abstractmethod
    def grad_steps(self) -> int:
        pass

    @property
    @abstractmethod
    def epochs(self) -> int:
        pass

    @property
    @abstractmethod
    def weight_decay(self) -> float:
        pass

    @property
    @abstractmethod
    def warmup_ratio(self) -> float:
        pass

    @property
    @abstractmethod
    def csv_train(self) -> str:
        pass

    @property
    @abstractmethod
    def csv_test(self) -> str:
        pass

    @property
    @abstractmethod
    def nombre_run(self) -> str:
        pass

    @property
    @abstractmethod
    def nombre_modelo_guardado(self) -> str:
        pass

    @abstractmethod
    def get_multitask_model(self, num_labels_grupo, num_labels_caja) -> nn.Module:
        pass

    def get_feature_extractor(self):
        return AutoFeatureExtractor.from_pretrained(self.nombre_modelo)

    def preprocesar_batch(self, batch, ruta_audios, feature_extractor):
        audio_arrays = []
        for nombre_archivo in batch["nombre_archivo"]:
            ruta_audio = os.path.join(ruta_audios, nombre_archivo)
            y, _ = librosa.load(ruta_audio, sr=None)
            audio_arrays.append(y)

        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            padding="max_length",
            max_length=160000,
            truncation=True
        )

        inputs["labels_grupo"] = batch["label_grupo"]
        inputs["labels_caja"] = batch["label_caja"]
        return inputs

    def evaluar_por_batches(self, modelo, dataset, batch_size, device):
        dataset.set_format('torch')
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        preds_grupo, preds_caja = [], []

        with torch.no_grad():
            for batch in dataloader:
                input_values = batch['input_values'].to(device)
                attention_mask = batch['attention_mask'].to(device) if 'attention_mask' in batch else None
                outputs = modelo(input_values, attention_mask=attention_mask)
                preds_grupo.extend(torch.argmax(outputs['logits_grupo'], dim=-1).tolist())
                preds_caja.extend(torch.argmax(outputs['logits_caja'], dim=-1).tolist())

        real_grupo = [int(x) for x in dataset['label_grupo']]
        real_caja = [int(x) for x in dataset['label_caja']]

        return preds_grupo, preds_caja, real_grupo, real_caja

    def plot_confusion_matrix(self, real_grupo, preds_grupo, real_caja, preds_caja, clases_grupo, clases_caja):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Matrices de Confusión - {self.nombre_modelo_guardado}", fontsize=16, fontweight="bold")
        
        cm_grupo = confusion_matrix(real_grupo, preds_grupo)
        sns.heatmap(cm_grupo, annot=True, fmt="d", cmap="Blues", ax=axes[0],xticklabels=clases_grupo, yticklabels=clases_grupo)
        axes[0].set_title("Predicción: Grupo Clínico")
        axes[0].set_ylabel("Etiqueta Real")
        axes[0].set_xlabel("Predicción")
        axes[0].tick_params(axis="x", rotation=45)
        
        cm_caja = confusion_matrix(real_caja, preds_caja)
        sns.heatmap(cm_caja, annot=True, fmt="d", cmap="Greens", ax=axes[1],xticklabels=clases_caja, yticklabels=clases_caja)
        axes[1].set_title("Predicción: Caja Torácica")
        axes[1].set_ylabel("Etiqueta Real")
        axes[1].set_xlabel("Predicción")
        axes[1].tick_params(axis="x", rotation=45)
        
        plt.tight_layout()
        
        nombre_archivo_fig = f"confusion_matrix_{self.nombre_modelo_guardado}.png"
        mlflow.log_figure(fig, nombre_archivo_fig)
        
        plt.close(fig)
    
    def ejecutar(self):
        ruta_base = Path(__file__).resolve().parent.parent
        ruta_resultados = ruta_base / "resultados"
        ruta_db = ruta_resultados / "resultados_voces.db"
        ruta_mlruns = ruta_resultados / "mlruns"
        
        os.makedirs(ruta_resultados, exist_ok=True)

        mlflow.set_tracking_uri(f"sqlite:///{ruta_db.as_posix()}")
        nombre_experimento = "Clasificacion_Transformers"
        
        experimento = mlflow.get_experiment_by_name(nombre_experimento)
        if experimento is None:
            mlflow.create_experiment(nombre_experimento, artifact_location=ruta_mlruns.as_uri())
        mlflow.set_experiment(nombre_experimento)
    

        ruta_entrenamiento = ruta_base / "datos_entrenamiento"
        ruta_audios = str(ruta_base / self.ruta_audios)
        ruta_modelos = ruta_base / "modelos_entrenados"

        ruta_csv_train = ruta_entrenamiento / self.csv_train
        ruta_csv_test = ruta_entrenamiento / self.csv_test
        
        try:
            df_train = pd.read_csv(ruta_csv_train, sep=";")
            df_test = pd.read_csv(ruta_csv_test, sep=";")
        except FileNotFoundError:
            print(f"Archivos de metadata no encontrados: {self.csv_train} o {self.csv_test}")
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

    
        feature_extractor = self.get_feature_extractor()

        print("Preprocesando conjunto de entrenamiento...")
        train_dataset = train_dataset.map(
            lambda batch: self.preprocesar_batch(batch, ruta_audios, feature_extractor),
            batched=True,
            batch_size=8,
            remove_columns=['nombre_archivo']
        )

        print("Preprocesando conjunto de test...")
        test_dataset = test_dataset.map(
            lambda batch: self.preprocesar_batch(batch, ruta_audios, feature_extractor),
            batched=True,
            batch_size=8,
            remove_columns=['nombre_archivo']
        )

        

        with mlflow.start_run(run_name=self.nombre_run):
            

            mlflow.log_param("modelo", self.nombre_modelo)
            mlflow.log_param("learning_rate", self.learning_rate)
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("gradient_acc_steps", self.grad_steps)
            mlflow.log_param("num_epochs", self.epochs)
            mlflow.log_param("weight_decay", self.weight_decay)
            mlflow.log_param("warmup_ratio", self.warmup_ratio)

            cv_accuracies_grupo = []
            cv_accuracies_caja = []

            for fold_val in range(5):
                print(f"\n--- Iniciando Entrenamiento Fold {fold_val}/4 ---")

                train_fold_ds = train_dataset.filter(lambda example: example['fold'] != fold_val)
                val_fold_ds = train_dataset.filter(lambda example: example['fold'] == fold_val)

                train_fold_ds = train_fold_ds.remove_columns(['fold'])
                val_fold_ds = val_fold_ds.remove_columns(['fold'])

                modelo_cv = self.get_multitask_model(num_labels_grupo, num_labels_caja)

                output_dir_cv = str(ruta_modelos / f"{self.nombre_modelo_guardado}_fold_{fold_val}")
                
                training_args_cv = TrainingArguments(
                    output_dir=output_dir_cv,
                    eval_strategy="epoch",
                    save_strategy="no",
                    learning_rate=self.learning_rate,
                    per_device_train_batch_size=self.batch_size,
                    per_device_eval_batch_size=self.batch_size,
                    gradient_accumulation_steps=self.grad_steps,
                    num_train_epochs=self.epochs,
                    weight_decay=self.weight_decay,
                    warmup_ratio=self.warmup_ratio,
                    logging_steps=10,
                    remove_unused_columns=False,
                    bf16=True
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

                preds_grupo, preds_caja, real_grupo, real_caja = self.evaluar_por_batches(modelo_cv, val_fold_ds, batch_size=4, device=device)

                acc_grupo = sum(p == r for p, r in zip(preds_grupo, real_grupo)) / len(real_grupo)
                acc_caja = sum(p == r for p, r in zip(preds_caja, real_caja)) / len(real_caja)
                cv_accuracies_grupo.append(acc_grupo)
                cv_accuracies_caja.append(acc_caja)
                mlflow.log_metric(f"fold_{fold_val}_acc_grupo", acc_grupo)
                mlflow.log_metric(f"fold_{fold_val}_acc_caja", acc_caja)

                print(f"Resultados Fold {fold_val} -> Accuracy Grupo: {acc_grupo:.4f} | Accuracy Caja: {acc_caja:.4f}")
                
                #TODO: eliminar delete cuando tengamos hiperparámetros finales
                del trainer_cv, modelo_cv
                torch.cuda.empty_cache()
                gc.collect()

            print(f"Precisión Media Grupo: {np.mean(cv_accuracies_grupo):.4f} (+/- {np.std(cv_accuracies_grupo):.4f})")
            print(f"Precisión Media Caja: {np.mean(cv_accuracies_caja):.4f} (+/- {np.std(cv_accuracies_caja):.4f})")

            print("Iniciando Entrenamiento Final del Modelo con TODOS los datos de Train...")
            train_final_ds = train_dataset.remove_columns(['fold'])

            modelo_final = self.get_multitask_model(num_labels_grupo, num_labels_caja)
            output_dir_final = str(ruta_modelos / f"entrenamiento_final_{self.nombre_modelo_guardado}")

            training_args_final = TrainingArguments(
                output_dir=output_dir_final,
                eval_strategy="no",
                save_strategy="no",
                learning_rate=self.learning_rate,
                per_device_train_batch_size=self.batch_size,
                gradient_accumulation_steps=self.grad_steps,
                num_train_epochs=self.epochs,
                weight_decay=self.weight_decay,
                warmup_ratio=self.warmup_ratio,
                logging_steps=10,
                remove_unused_columns=False,
                bf16=True
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

            preds_grupo_list, preds_caja_list, real_grupo_list, real_caja_list = self.evaluar_por_batches(modelo_final, test_dataset, batch_size=4, device=device)

            print("\nReporte Final - GRUPO:\n")
            etiquetas_grupo = np.arange(len(le_grupo.classes_))
            reporte_grupo_str = classification_report(real_grupo_list, preds_grupo_list, labels=etiquetas_grupo, target_names=le_grupo.classes_, zero_division=0)
            reporte_grupo_dict = classification_report(real_grupo_list, preds_grupo_list, labels=etiquetas_grupo, target_names=le_grupo.classes_, zero_division=0, output_dict=True)
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

            mlflow.log_metric("cv_mean_acc_grupo", float(np.mean(cv_accuracies_grupo)))
            mlflow.log_metric("cv_std_acc_grupo", float(np.std(cv_accuracies_grupo)))
            mlflow.log_metric("cv_mean_acc_caja", float(np.mean(cv_accuracies_caja)))
            mlflow.log_metric("cv_std_acc_caja", float(np.std(cv_accuracies_caja)))
            mlflow.log_metric("test_acc_grupo", reporte_grupo_dict["accuracy"])
            mlflow.log_metric("test_acc_caja", reporte_caja_dict["accuracy"])

            mlflow.log_dict(reporte_grupo_dict, "reporte_clasificacion_grupo.json")
            mlflow.log_dict(reporte_caja_dict, "reporte_clasificacion_caja.json")

            self.plot_confusion_matrix(
                real_grupo=real_grupo_list, 
                preds_grupo=preds_grupo_list, 
                clases_grupo=le_grupo.classes_,
                real_caja=real_caja_list, 
                preds_caja=preds_caja_list, 
                clases_caja=le_caja.classes_
            )

            ruta_guardado_final = ruta_modelos / self.nombre_modelo_guardado
            os.makedirs(ruta_guardado_final, exist_ok=True)

            torch.save(modelo_final.state_dict(), ruta_guardado_final / "pytorch_model.bin")
            feature_extractor.save_pretrained(str(ruta_guardado_final))

            np.save(ruta_guardado_final / "label_classes_grupo.npy", le_grupo.classes_)
            np.save(ruta_guardado_final / "label_classes_caja.npy", le_caja.classes_)

            print(f"\nProceso completado. Modelo final guardado en '{ruta_guardado_final}'")
