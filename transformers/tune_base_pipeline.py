import os
import shutil
import librosa
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gc
import random

if torch.cuda.is_available():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from abc import ABC, abstractmethod
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import classification_report
from datasets import Dataset

SEED = 42
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
        for layer in self.hubert.encoder.layers[:6]:
            for param in layer.parameters():
                param.requires_grad = False

        hidden_size = self.hubert.config.hidden_size
        self.classifier_grupo = nn.Linear(hidden_size, num_labels_grupo)
        self.classifier_caja = nn.Linear(hidden_size, num_labels_caja)

    #def gradient_checkpointing_enable(self, **kwargs):
    #    self.hubert.gradient_checkpointing_enable(**kwargs)

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
        for layer in self.wav2vec2.encoder.layers[:6]:
            for param in layer.parameters():
                param.requires_grad = False

        hidden_size = self.wav2vec2.config.hidden_size
        self.classifier_grupo = nn.Linear(hidden_size, num_labels_grupo)
        self.classifier_caja = nn.Linear(hidden_size, num_labels_caja)

    #def gradient_checkpointing_enable(self, **kwargs):
    #    self.wav2vec2.gradient_checkpointing_enable(**kwargs)

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

        audio_input = inputs.get("input_values") if "input_values" in inputs else inputs.get("input_features")
        attention_mask = inputs.get("attention_mask")

        outputs = model(audio_input, attention_mask=attention_mask)

        loss_fct = nn.CrossEntropyLoss()
        loss_grupo = loss_fct(outputs["logits_grupo"], labels_grupo)
        loss_caja = loss_fct(outputs["logits_caja"], labels_caja)
        loss = loss_grupo + loss_caja

        return (loss, outputs) if return_outputs else loss


class BaseTransformerPipeline(ABC):
    
    #Este metodo lo creamos para manejar Whisper, el cual espera una entrada con 30000 pasos temporales, 
    #mientras que nosotros le otorgamos una con 10000 pasos. Al devolver None en los pipelines de Whisper, el modelo aplicará padding
    #de manera interna hasta alcanzar los pasos necesarios. En los demás dejamos el valor que tenemos: 16000 (Hz).  
    @property
    @abstractmethod
    def max_audio_length(self):
        pass

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
    def warmup_steps(self) -> int:
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

    def get_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    
    def clear_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def preprocesar_batch(self, batch, ruta_audios, feature_extractor):
        audio_arrays = []
        for nombre_archivo in batch["nombre_archivo"]:
            ruta_audio = os.path.join(ruta_audios, nombre_archivo)
            y, _ = librosa.load(ruta_audio, sr=16000)
            audio_arrays.append(y)

        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            padding="max_length",
            max_length=self.max_audio_length,
            truncation=True
        )

        inputs["labels_grupo"] = batch["label_grupo"]
        inputs["labels_caja"] = batch["label_caja"]
        return inputs

    def evaluar_por_batches(self, modelo, dataset, batch_size, device):
        dataset.set_format('torch')
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        preds_grupo, preds_caja = [], []
        probs_grupo, probs_caja = [], []

        with torch.no_grad():
            for batch in dataloader:
                if 'input_values' in batch:
                    audio_input = batch['input_values'].to(device)
                else:
                    audio_input = batch['input_features'].to(device)
                
                attention_mask = batch['attention_mask'].to(device) if 'attention_mask' in batch else None
                outputs = modelo(audio_input, attention_mask=attention_mask)
                logits_g = outputs['logits_grupo']
                logits_c = outputs['logits_caja']
                
                # Calculamos probabilidades usando Softmax
                probs_g = torch.softmax(logits_g, dim=-1)
                probs_c = torch.softmax(logits_c, dim=-1)
                
                preds_grupo.extend(torch.argmax(logits_g, dim=-1).cpu().tolist())
                preds_caja.extend(torch.argmax(logits_c, dim=-1).cpu().tolist())
                
                probs_grupo.extend(probs_g.cpu().tolist())
                probs_caja.extend(probs_c.cpu().tolist())

        real_grupo = [int(x) for x in dataset['label_grupo']]
        real_caja = [int(x) for x in dataset['label_caja']]

        return preds_grupo, preds_caja, real_grupo, real_caja, np.array(probs_grupo), np.array(probs_caja)
    
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

    def plot_roc_curve(self, real_grupo, probs_grupo, real_caja, probs_caja, clases_grupo, clases_caja):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Curvas ROC - {self.nombre_modelo_guardado}", fontsize=16, fontweight="bold")

        for real, probs, clases, ax, title in zip(
            [real_grupo, real_caja], 
            [probs_grupo, probs_caja], 
            [clases_grupo, clases_caja], 
            axes, 
            ["Grupo Clínico", "Caja Torácica"]
        ):
            real_bin = label_binarize(real, classes=np.arange(len(clases)))
            if len(clases) == 2:
                real_bin = np.hstack((1 - real_bin, real_bin))

            for j, clase in enumerate(clases):
                if real_bin[:, j].sum() == 0:
                    continue
                fpr, tpr, _ = roc_curve(real_bin[:, j], probs[:, j])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2, label=f'{clase} (AUC = {roc_auc:.2f})')

            ax.plot([0, 1], [0, 1], 'k--', lw=2)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Tasa de Falsos Positivos')
            ax.set_ylabel('Tasa de Verdaderos Positivos')
            ax.set_title(f'ROC: {title}')
            ax.legend(loc="lower right")

        plt.tight_layout()
        nombre_archivo_fig = f"roc_curve_{self.nombre_modelo_guardado}.png"
        mlflow.log_figure(fig, nombre_archivo_fig)
        plt.close(fig)
    
    def ejecutar(self, modo_tuning=False):
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

        

        self.set_seed(SEED)
        device = self.get_device()
        use_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
        use_fp16 = device.type == 'mps'
        print(f"Dispositivo: {device} | bf16={use_bf16} | fp16={use_fp16}")

        with mlflow.start_run(run_name=f"{self.nombre_run}_{self.nombre_dataset}"):

            mlflow.log_param("modelo", self.nombre_modelo)
            mlflow.log_param("dataset", self.nombre_dataset)
            mlflow.log_param("learning_rate", self.learning_rate)
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("gradient_acc_steps", self.grad_steps)
            mlflow.log_param("num_epochs", self.epochs)
            mlflow.log_param("weight_decay", self.weight_decay)
            mlflow.log_param("warmup_steps", self.warmup_steps)

            cv_accuracies_grupo = []
            cv_accuracies_caja = []
            cv_f1_macro_grupo = []
            cv_f1_macro_caja = []


            for fold_val in range(5):
                print(f"\n--- Iniciando Entrenamiento Fold {fold_val}/4 ---")

                train_fold_ds = train_dataset.filter(lambda example: example['fold'] != fold_val)
                val_fold_ds = train_dataset.filter(lambda example: example['fold'] == fold_val)

                train_fold_ds = train_fold_ds.remove_columns(['fold'])
                val_fold_ds = val_fold_ds.remove_columns(['fold'])

                self.set_seed(SEED + fold_val)
                modelo_cv = self.get_multitask_model(num_labels_grupo, num_labels_caja)

                output_dir_cv = str(ruta_modelos / f"{self.nombre_modelo_guardado}_fold_{fold_val}")

                training_args_cv = TrainingArguments(
                    output_dir=output_dir_cv,
                    eval_strategy="no",
                    save_strategy="no",
                    learning_rate=self.learning_rate,
                    per_device_train_batch_size=self.batch_size,
                    per_device_eval_batch_size=self.batch_size,
                    gradient_accumulation_steps=self.grad_steps,
                    num_train_epochs=self.epochs,
                    weight_decay=self.weight_decay,
                    warmup_steps=self.warmup_steps,
                    logging_steps=10,
                    remove_unused_columns=False,
                    bf16=use_bf16,
                    fp16=use_fp16,
                    seed=SEED + fold_val,
                )

                trainer_cv = MultiTaskTrainer(
                    model=modelo_cv,
                    args=training_args_cv,
                    train_dataset=train_fold_ds,
                    eval_dataset=val_fold_ds,
                )

                #print("Ejecutando .train().")
                trainer_cv.train()

                modelo_cv.eval()
                modelo_cv.to(device)

                preds_grupo, preds_caja, real_grupo, real_caja, _, _ = self.evaluar_por_batches(modelo_cv, val_fold_ds, batch_size=4, device=device)
                val_fold_ds.reset_format()

                acc_grupo = sum(p == r for p, r in zip(preds_grupo, real_grupo)) / len(real_grupo)
                acc_caja = sum(p == r for p, r in zip(preds_caja, real_caja)) / len(real_caja)
                f1_grupo = f1_score(real_grupo, preds_grupo, average='macro', zero_division=0)
                f1_caja = f1_score(real_caja, preds_caja, average='macro', zero_division=0)

                cv_accuracies_grupo.append(acc_grupo)
                cv_accuracies_caja.append(acc_caja)
                cv_f1_macro_grupo.append(f1_grupo)
                cv_f1_macro_caja.append(f1_caja)

                mlflow.log_metric(f"fold_{fold_val}_acc_grupo", acc_grupo)
                mlflow.log_metric(f"fold_{fold_val}_acc_caja", acc_caja)
                mlflow.log_metric(f"fold_{fold_val}_f1_macro_grupo", f1_grupo)
                mlflow.log_metric(f"fold_{fold_val}_f1_macro_caja", f1_caja)

                print(f"Resultados Fold {fold_val} -> Accuracy Grupo: {acc_grupo:.4f} | Accuracy Caja: {acc_caja:.4f}")
                
                modelo_cv.cpu()
                del trainer_cv, modelo_cv
                self.clear_memory()
                gc.collect()
                if os.path.exists(output_dir_cv):
                    shutil.rmtree(output_dir_cv)

            print(f"Precisión Media Grupo: {np.mean(cv_accuracies_grupo):.4f} (+/- {np.std(cv_accuracies_grupo):.4f})")
            print(f"Precisión Media Caja: {np.mean(cv_accuracies_caja):.4f} (+/- {np.std(cv_accuracies_caja):.4f})")
            print(f"F1-Macro Medio Grupo: {np.mean(cv_f1_macro_grupo):.4f}")
            print(f"F1-Macro Medio Caja: {np.mean(cv_f1_macro_caja):.4f}")

            mlflow.log_metric("cv_mean_acc_grupo", float(np.mean(cv_accuracies_grupo)))
            mlflow.log_metric("cv_std_acc_grupo", float(np.std(cv_accuracies_grupo, ddof=1)))
            mlflow.log_metric("cv_mean_acc_caja", float(np.mean(cv_accuracies_caja)))
            mlflow.log_metric("cv_std_acc_caja", float(np.std(cv_accuracies_caja, ddof=1)))
            mlflow.log_metric("cv_mean_f1_macro_grupo", float(np.mean(cv_f1_macro_grupo)))
            mlflow.log_metric("cv_mean_f1_macro_caja", float(np.mean(cv_f1_macro_caja)))

            if modo_tuning:
                print(f"MODO TUNING FINALIZADO")
                print("El conjunto Test no ha sido evaluado.")
                return

            print("Iniciando Entrenamiento Final del Modelo con TODOS los datos de Train...")
            train_final_ds = train_dataset.remove_columns(['fold'])

            self.set_seed(SEED)
            modelo_final = self.get_multitask_model(num_labels_grupo, num_labels_caja)
            output_dir_final = str(ruta_modelos / f"entrenamiento_final_{self.nombre_modelo_guardado}")

            training_args_final = TrainingArguments(
                output_dir=output_dir_final,
                eval_strategy="no",
                save_strategy="no",
                learning_rate=self.learning_rate,
                per_device_train_batch_size=self.batch_size,
                gradient_accumulation_steps=self.grad_steps,
                #gradient_checkpointing=True,
                num_train_epochs=self.epochs,
                weight_decay=self.weight_decay,
                warmup_steps=self.warmup_steps,
                logging_steps=10,
                remove_unused_columns=False,
                bf16=use_bf16,
                fp16=use_fp16,
                seed=SEED,
            )

            trainer_final = MultiTaskTrainer(
                model=modelo_final,
                args=training_args_final,
                train_dataset=train_final_ds,
            )

            trainer_final.train()

            print("\nEvaluando Modelo Final sobre el conjunto de TEST (Hold-out Validation / Preproducción)...")
            modelo_final.eval()
            print(f"Trabajando con {device}")
            modelo_final.to(device)

            preds_grupo_list, preds_caja_list, real_grupo_list, real_caja_list, probs_grupo_arr, probs_caja_arr = self.evaluar_por_batches(modelo_final, test_dataset, batch_size=4, device=device)
            
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

            clases_presentes_grupo = np.unique(np.array(real_grupo_list))
            clases_presentes_caja = np.unique(np.array(real_caja_list))
            probs_grupo_filtradas = probs_grupo_arr[:, clases_presentes_grupo]
            probs_grupo_filtradas = probs_grupo_filtradas / probs_grupo_filtradas.sum(axis=1, keepdims=True)
            probs_caja_filtradas = probs_caja_arr[:, clases_presentes_caja]
            probs_caja_filtradas = probs_caja_filtradas / probs_caja_filtradas.sum(axis=1, keepdims=True)
            try:
                auc_grupo = roc_auc_score(real_grupo_list, probs_grupo_filtradas, multi_class='ovr', average='macro', labels=clases_presentes_grupo)
                auc_caja = roc_auc_score(real_caja_list, probs_caja_filtradas, multi_class='ovr', average='macro', labels=clases_presentes_caja)
            except Exception as e:
                print(f"[WARN] No se pudo calcular AUC-ROC: {e}")
                auc_grupo = float('nan')
                auc_caja = float('nan')

            # Registrar en MLflow (Hold-out Validation)
            mlflow.log_metric("test_acc_grupo", reporte_grupo_dict["accuracy"])
            mlflow.log_metric("test_f1_macro_grupo", reporte_grupo_dict["macro avg"]["f1-score"])
            mlflow.log_metric("test_f1_weighted_grupo", reporte_grupo_dict["weighted avg"]["f1-score"])
            mlflow.log_metric("test_recall_grupo", reporte_grupo_dict["weighted avg"]["recall"])
            mlflow.log_metric("test_precision_grupo", reporte_grupo_dict["weighted avg"]["precision"])
            mlflow.log_metric("test_auc_roc_grupo", auc_grupo)
            
            mlflow.log_metric("test_acc_caja", reporte_caja_dict["accuracy"])
            mlflow.log_metric("test_f1_macro_caja", reporte_caja_dict["macro avg"]["f1-score"])
            mlflow.log_metric("test_f1_weighted_caja", reporte_caja_dict["weighted avg"]["f1-score"])
            mlflow.log_metric("test_recall_caja", reporte_caja_dict["weighted avg"]["recall"])
            mlflow.log_metric("test_precision_caja", reporte_caja_dict["weighted avg"]["precision"])
            mlflow.log_metric("test_auc_roc_caja", auc_caja)

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
            self.plot_roc_curve(
                real_grupo=real_grupo_list,
                probs_grupo=probs_grupo_arr,
                real_caja=real_caja_list,
                probs_caja=probs_caja_arr,
                clases_grupo=le_grupo.classes_,
                clases_caja=le_caja.classes_
            )

            ruta_guardado_final = ruta_modelos / self.nombre_modelo_guardado
            os.makedirs(ruta_guardado_final, exist_ok=True)

            torch.save(modelo_final.state_dict(), ruta_guardado_final / "pytorch_model.bin")
            feature_extractor.save_pretrained(str(ruta_guardado_final))

            np.save(ruta_guardado_final / "label_classes_grupo.npy", le_grupo.classes_)
            np.save(ruta_guardado_final / "label_classes_caja.npy", le_caja.classes_)

            print(f"\nProceso completado. Modelo final guardado en '{ruta_guardado_final}'")
