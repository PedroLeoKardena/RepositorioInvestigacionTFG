import os
import pickle
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from abc import ABC, abstractmethod
from pathlib import Path
from sklearn.metrics import confusion_matrix

#pytorch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

#sklearn
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve, auc


from modelo_crnn import CRNN

class DatasetMFCC(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fila = self.df.iloc[idx]

        mfcc = fila['mfccs'] 
        etiqueta_grupo = fila['label_grupo']
        etiqueta_caja = fila['label_caja']
       
        tensor_mfcc = torch.tensor(mfcc, dtype=torch.float32)
        tensor_grupo = torch.tensor(etiqueta_grupo, dtype=torch.long)
        tensor_caja = torch.tensor(etiqueta_caja, dtype=torch.long)

        if tensor_mfcc.shape[1] > 1000:
            #Hay que tener en cuenta que librosa inserta, para un chunk de 10 segundos, con un hop_length de 160 y frecuencia de 16000 Hz, genera 1001 muestras temporales, ya que incluye el instante 0.
            #Para este código no lo contamos
            tensor_mfcc = tensor_mfcc[:, :1000]
        elif tensor_mfcc.shape[1] < 1000:
            tensor_mfcc = torch.nn.functional.pad(tensor_mfcc, (0, 1000 - tensor_mfcc.shape[1]))

        #Esto lo hacemos es pasar de un tensor = [30,1000] a un tensor = [1,30,1000], donde 1 = canal_inicial
        tensor_mfcc = tensor_mfcc.unsqueeze(0)
        
        return tensor_mfcc, tensor_grupo, tensor_caja

class PipelineComunCRNN(ABC):

    def __init__(self, pkl_train, pkl_test, nombre_dataset, hidden_size, batch_size, num_capas_ocultas_lstm, alpha_leaky_relu, is_bidirectional, dropout, lr_adam, num_epochs):
        self.ruta_base = Path(__file__).resolve().parent.parent.parent
        self.pkl_train = pkl_train
        self.pkl_test = pkl_test
        self.nombre_dataset = nombre_dataset
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_capas_ocultas_lstm = num_capas_ocultas_lstm
        self.alpha_leaky_relu = alpha_leaky_relu
        self.is_bidirectional = is_bidirectional
        self.dropout = dropout
        self.lr_adam = lr_adam
        self.num_epochs = num_epochs

        tiempo_actual = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        self.nombre_modelo_guardado = f"CRNN_MFCC_{self.nombre_dataset}_{tiempo_actual}"

    def get_device():
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    
    def clear_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def obtener_datasets(self):
        ruta_datos = self.ruta_base / "datos_entrenamiento"
        ruta_train = ruta_datos / self.pkl_train
        ruta_test = ruta_datos / self.pkl_test

        datos_train = pickle.load(open(ruta_train, 'rb'))
        datos_test = pickle.load(open(ruta_test, 'rb'))

        df_train = pd.DataFrame(datos_train)
        df_test = pd.DataFrame(datos_test)

        return df_train, df_test

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
            # Binarizar etiquetas para ROC multiclase
            real_bin = label_binarize(real, classes=np.arange(len(clases)))
            if len(clases) == 2:
                # Si es clasificación binaria, label_binarize devuelve 1 columna, necesitamos 2 para iterar igual
                real_bin = np.hstack((1 - real_bin, real_bin))

            for j, clase in enumerate(clases):
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

    def ejecutar(self, modo_tuning = True):
        
        ruta_resultados = self.ruta_base / "resultados"
        ruta_mlruns = ruta_resultados / "mlruns"
        ruta_db = ruta_resultados / "resultados_voces.db"
        os.makedirs(ruta_resultados, exist_ok=True)

        device = self.get_device()
        print(f"Dispositivo detectado: {device}")

        mlflow.set_tracking_uri(f"sqlite:///{ruta_db.as_posix()}")
        nombre_experimento = "Clasificacion_CRNN_MFCC"

        experimento = mlflow.get_experiment_by_name(nombre_experimento)
        if experimento is None:
            mlflow.create_experiment(nombre_experimento, artifact_location=ruta_mlruns.as_uri())
        mlflow.set_experiment(nombre_experimento)

        train_dataset, test_dataset = self.obtener_datasets()
        le_grupo = LabelEncoder()
        le_caja = LabelEncoder()
        train_dataset['label_grupo'] = le_grupo.fit_transform(train_dataset['grupo'])
        test_dataset['label_grupo'] = le_grupo.transform(test_dataset['grupo'])

        train_dataset['label_caja'] = le_caja.fit_transform(train_dataset['caja_toracica'])
        test_dataset['label_caja'] = le_caja.transform(test_dataset['caja_toracica'])

        #Utilizamos crossEntropyLoss ya que se trata del estándar para clasificación multiclase.
        criterio_grupo = nn.CrossEntropyLoss()
        criterio_caja = nn.CrossEntropyLoss()

        nombre_run = f"CRNN_MFCC_{self.nombre_dataset}_hidden{self.hidden_size}"

        cv_val_losses = []
        cv_val_f1_grupo = []
        cv_val_f1_caja = []

        with mlflow.start_run(run_name = nombre_run):

            mlflow.log_param("hidden_size", self.hidden_size)
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("num_capas_ocultas_lstm", self.num_capas_ocultas_lstm)
            mlflow.log_param("alpha_leaky_relu", self.alpha_leaky_relu)
            mlflow.log_param("is_bidirectional", self.is_bidirectional)
            mlflow.log_param("dropout", self.dropout)
            mlflow.log_param("num_epochs", self.num_epochs)
            mlflow.log_param("lr_adam", self.lr_adam)

            for fold_val in range(5):
                print(f"\n--- Iniciando Entrenamiento Fold {fold_val}/4 ---")

                df_train_fold = train_dataset[train_dataset['fold'] != fold_val]
                df_val_fold = train_dataset[train_dataset['fold'] == fold_val]

                train_loader = DataLoader(DatasetMFCC(df_train_fold), batch_size=self.batch_size, shuffle=True)
                val_loader = DataLoader(DatasetMFCC(df_val_fold), batch_size=self.batch_size, shuffle=False)

                modelo = CRNN(num_features=30, num_time_steps=1000, hidden_size=self.hidden_size, num_capas_ocultas_lstm=self.num_capas_ocultas_lstm, alpha_leaky_relu=self.alpha_leaky_relu, is_bidirectional=self.is_bidirectional, dropout=self.dropout)
                modelo.to(device)

                #Utilizaremos optimizer Adam.
                optimizador = torch.optim.Adam(modelo.parameters(), lr=self.lr_adam)

                mejor_val_loss_fold = float('inf')

                for epoca in range(self.num_epochs):
                    modelo.train()
                    loss_train_total = 0.0
                    loss_actual = 0.0

                    for i, data in enumerate(train_loader, 0):
                        batch_mfcc, batch_grupo, batch_caja = data
                        batch_mfcc, batch_grupo, batch_caja = batch_mfcc.to(device), batch_grupo.to(device), batch_caja.to(device)

                        optimizador.zero_grad()

                        pred_grupo, pred_caja = modelo(batch_mfcc)
                        
                       
                        #if pred_grupo.dim() == 3:
                        #    pred_grupo = pred_grupo[:, -1, :]
                        #    pred_caja = pred_caja[:, -1, :]

                        loss_grupo = criterio_grupo(pred_grupo, batch_grupo)
                        loss_caja = criterio_caja(pred_caja, batch_caja)
                        loss_total = loss_grupo + loss_caja

                        loss_total.backward()
                        optimizador.step()
                        loss_actual += loss_total.item()
                        loss_train_total += loss_total.item()

                        if i % 50 == 0:
                            print(f"Fold actual: {fold_val}, Epoca: {epoca + 1}, Batch: {i + 1}, Loss Actual: {loss_actual / 50:.3f}")
                            loss_actual = 0.0
                    
                    loss_train_media = loss_train_total / len(train_loader)

                    #Validacion
                    modelo.eval()
                    loss_val_total = 0.0
                    
                    preds_val_grupo, preds_val_caja = [], []
                    reales_val_grupo, reales_val_caja = [], []

                    with torch.no_grad():
                        for data in val_loader:
                            batch_mfcc, batch_grupo, batch_caja = data
                            batch_mfcc, batch_grupo, batch_caja = batch_mfcc.to(device), batch_grupo.to(device), batch_caja.to(device)

                            pred_grupo, pred_caja = modelo(batch_mfcc)

                            #if pred_grupo.dim() == 3:
                            #    pred_grupo = pred_grupo[:, -1, :]
                            #    pred_caja = pred_caja[:, -1, :]

                            loss_grupo = criterio_grupo(pred_grupo, batch_grupo)
                            loss_caja = criterio_caja(pred_caja, batch_caja)
                            loss_total = loss_grupo + loss_caja
                            
                            loss_val_total += loss_total.item()
                            
                            clases_pred_grupo = torch.argmax(pred_grupo, dim=1)
                            clases_pred_caja = torch.argmax(pred_caja, dim=1)
                            
                            preds_val_grupo.extend(clases_pred_grupo.cpu().numpy())
                            preds_val_caja.extend(clases_pred_caja.cpu().numpy())
                            reales_val_grupo.extend(batch_grupo.cpu().numpy())
                            reales_val_caja.extend(batch_caja.cpu().numpy())

                    acc_val_grupo = sum(p == r for p, r in zip(preds_val_grupo, reales_val_grupo)) / len(reales_val_grupo)
                    acc_val_caja = sum(p == r for p, r in zip(preds_val_caja, reales_val_caja)) / len(reales_val_caja)
                    f1_val_grupo = f1_score(reales_val_grupo, preds_val_grupo, average='macro', zero_division=0)
                    f1_val_caja = f1_score(reales_val_caja, preds_val_caja, average='macro', zero_division=0)
                    
                    mlflow.log_metric(f"fold_{fold_val}_val_acc_grupo", acc_val_grupo, step=epoca)
                    mlflow.log_metric(f"fold_{fold_val}_val_acc_caja", acc_val_caja, step=epoca)
                    mlflow.log_metric(f"fold_{fold_val}_val_f1_macro_grupo", f1_val_grupo, step=epoca)
                    mlflow.log_metric(f"fold_{fold_val}_val_f1_macro_caja", f1_val_caja, step=epoca)

                    loss_val_media = loss_val_total / len(val_loader)
                    if loss_val_media < mejor_val_loss_fold:
                        mejor_val_loss_fold = loss_val_media
                        mejor_f1_grupo_fold = f1_val_grupo
                        mejor_f1_caja_fold = f1_val_caja
                    
                    print(f"Época {epoca+1}/{self.num_epochs} | Train Loss: {loss_train_media:.4f} | Val Loss: {loss_val_media:.4f}")
                    mlflow.log_metric(f"fold_{fold_val}_train_loss", loss_train_media, step=epoca)
                    mlflow.log_metric(f"fold_{fold_val}_val_loss", loss_val_media, step=epoca)
                
                cv_val_losses.append(mejor_val_loss_fold)
                cv_val_f1_grupo.append(mejor_f1_grupo_fold)
                cv_val_f1_caja.append(mejor_f1_caja_fold)

                del modelo, optimizador
                self.clear_memory()
                gc.collect()

            mlflow.log_metric("cv_mean_val_loss", np.mean(cv_val_losses))
            mlflow.log_metric("cv_std_val_loss", float(np.std(cv_val_losses)))
            mlflow.log_metric("cv_mean_val_f1_grupo", np.mean(cv_val_f1_grupo))
            mlflow.log_metric("cv_mean_val_f1_caja", np.mean(cv_val_f1_caja))

            if modo_tuning:
                print(f"MODO TUNING FINALIZADO")
                print(f"Loss Validación Media: {np.mean(cv_val_losses):.4f}")
                print(f"F1-Macro Grupo Medio:  {np.mean(cv_val_f1_grupo):.4f}")
                print(f"F1-Macro Caja Medio:   {np.mean(cv_val_f1_caja):.4f}")
                print("El conjunto Test no ha sido evaluado.")
                return

            train_final_ds = train_dataset.drop(columns=['fold'])
            train_final_loader = DataLoader(DatasetMFCC(train_final_ds), batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(DatasetMFCC(test_dataset), batch_size=self.batch_size, shuffle=False)

            modelo_final = CRNN(num_features=30, num_time_steps=1000, hidden_size=self.hidden_size, num_capas_ocultas_lstm=self.num_capas_ocultas_lstm, alpha_leaky_relu=self.alpha_leaky_relu, is_bidirectional=self.is_bidirectional, dropout=self.dropout)
            modelo_final.to(device)
            optimizador_final = torch.optim.Adam(modelo_final.parameters(), lr=self.lr_adam)

            for epoca in range(self.num_epochs):
                modelo_final.train()
                loss_train_total = 0.0

                for i, data in enumerate(train_final_loader, 0):
                    batch_mfcc, batch_grupo, batch_caja = data
                    batch_mfcc, batch_grupo, batch_caja = batch_mfcc.to(device), batch_grupo.to(device), batch_caja.to(device)

                    optimizador_final.zero_grad()
                    pred_grupo, pred_caja = modelo_final(batch_mfcc)

                    #if pred_grupo.dim() == 3:
                    #    pred_grupo = pred_grupo[:, -1, :]
                    #    pred_caja = pred_caja[:, -1, :]

                    loss_grupo = criterio_grupo(pred_grupo, batch_grupo)
                    loss_caja = criterio_caja(pred_caja, batch_caja)
                    loss_total = loss_grupo + loss_caja

                    loss_total.backward()
                    optimizador_final.step()
                    loss_train_total += loss_total.item()

                loss_train_media = loss_train_total / len(train_final_loader)
                print(f"Época {epoca+1}/{self.num_epochs} | Train Loss: {loss_train_media:.4f}")
                mlflow.log_metric("train_loss_final", loss_train_media, step=epoca)
            
            modelo_final.eval()

            preds_grupo, preds_caja = [], []
            etiquetas_grupo, etiquetas_caja = [], []
            probs_grupo_list, probs_caja_list = [], []

            with torch.no_grad():
                for data in test_loader:
                    batch_mfcc, batch_grupo, batch_caja = data
                    batch_mfcc, batch_grupo, batch_caja = batch_mfcc.to(device), batch_grupo.to(device), batch_caja.to(device)

                    pred_grupo, pred_caja = modelo_final(batch_mfcc)

                    #if pred_grupo.dim() == 3:
                    #    pred_grupo = pred_grupo[:, -1, :]
                    #    pred_caja = pred_caja[:, -1, :]
                    
                    probs_grupo = torch.softmax(pred_grupo, dim=1)
                    probs_caja = torch.softmax(pred_caja, dim=1)

                    clases_pred_grupo = torch.argmax(pred_grupo, dim=1)
                    clases_pred_caja = torch.argmax(pred_caja, dim=1)

                    preds_grupo.extend(clases_pred_grupo.cpu().numpy())
                    preds_caja.extend(clases_pred_caja.cpu().numpy())
                    probs_grupo_list.append(probs_grupo.cpu().numpy())
                    probs_caja_list.append(probs_caja.cpu().numpy())
                    etiquetas_grupo.extend(batch_grupo.cpu().numpy())
                    etiquetas_caja.extend(batch_caja.cpu().numpy())
            
            
            etiquetas_grupo = np.array(etiquetas_grupo)
            etiquetas_caja = np.array(etiquetas_caja)
            probs_grupo_arr = np.vstack(probs_grupo_list)
            probs_caja_arr = np.vstack(probs_caja_list)
            
            labels_grupo = np.arange(len(le_grupo.classes_))
            labels_caja = np.arange(len(le_caja.classes_))

            print("\n--- Reporte de Clasificación para Grupo ---")
            reporte_grupo_str = classification_report(etiquetas_grupo, preds_grupo, labels=labels_grupo, target_names=le_grupo.classes_, zero_division=0)
            reporte_grupo_dict = classification_report(etiquetas_grupo, preds_grupo, labels=labels_grupo, target_names=le_grupo.classes_, zero_division=0, output_dict=True)
            print(reporte_grupo_str)

            print("\n--- Reporte de Clasificación para Caja Torácica ---")
            reporte_caja_str = classification_report(etiquetas_caja, preds_caja, labels=labels_caja, target_names=le_caja.classes_, zero_division=0)
            reporte_caja_dict = classification_report(etiquetas_caja, preds_caja, labels=labels_caja, target_names=le_caja.classes_, zero_division=0, output_dict=True)
            print(reporte_caja_str)

            auc_grupo = roc_auc_score(etiquetas_grupo, probs_grupo_arr, multi_class='ovr', average='macro')
            auc_caja = roc_auc_score(etiquetas_caja, probs_caja_arr, multi_class='ovr', average='macro')

            mlflow.log_metric("test_acc_grupo", reporte_grupo_dict["accuracy"])
            mlflow.log_metric("test_f1_macro_grupo", reporte_grupo_dict["macro avg"]["f1-score"])
            mlflow.log_metric("test_f1_weighted_grupo", reporte_grupo_dict["weighted avg"]["f1-score"])
            mlflow.log_metric("test_auc_roc_grupo", auc_grupo)


            mlflow.log_metric("test_acc_caja", reporte_caja_dict["accuracy"])
            mlflow.log_metric("test_f1_macro_caja", reporte_caja_dict["macro avg"]["f1-score"])
            mlflow.log_metric("test_f1_weighted_caja", reporte_caja_dict["weighted avg"]["f1-score"])
            mlflow.log_metric("test_auc_roc_caja", auc_caja)

            mlflow.log_dict(reporte_grupo_dict, "reporte_clasificacion_grupo.json")
            mlflow.log_dict(reporte_caja_dict, "reporte_clasificacion_caja.json")

            self.plot_confusion_matrix(etiquetas_grupo, preds_grupo, etiquetas_caja, preds_caja, le_grupo.classes_, le_caja.classes_)
            self.plot_roc_curve(etiquetas_grupo, probs_grupo_arr, etiquetas_caja, probs_caja_arr, le_grupo.classes_, le_caja.classes_)

            nombre_ruta_modelo = f"modelo_crnn_mfcc_{self.nombre_modelo_guardado}.pth"
            ruta_modelos = self.ruta_base / "modelos_entrenados" / "modelos_CRNN_MFCC"
            os.makedirs(ruta_modelos, exist_ok=True)

            ruta_guardado_final = ruta_modelos / self.nombre_modelo_guardado
            os.makedirs(ruta_guardado_final, exist_ok=True)

            torch.save(modelo_final.state_dict(), ruta_guardado_final / "pytorch_model.bin")
            
            np.save(ruta_guardado_final / "label_classes_grupo.npy", le_grupo.classes_)
            np.save(ruta_guardado_final / "label_classes_caja.npy", le_caja.classes_)

            print(f"\nProceso completado. Modelo final guardado en '{ruta_guardado_final}'")




                    



                









    
    

    



