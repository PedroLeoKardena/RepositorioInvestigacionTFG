from tune_pipeline_comun import PipelineComunCRNN

pkl_train = "train_mels_chunked_aumentado_diarizado.pkl"
pkl_test = "test_mels_chunked_aumentado_diarizado.pkl"

"""
Después de los tuneos básicos, el mejor tuneo es el siguiente:
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.1
NUM_EPOCHS = 20
LR_ADAM = 0.001

En mlflow = 758167ec18b2408ab87ecf5686258fd4
Presenta el mejor f1_macro para caja o grupo y un val_loss equivalente a los demás.
cv_mean_loss = 2.1955857948272945
cv_mean_val_f1_grupo = 0.31711531315519337
cv_mean_val_f1_caja = 0.26078112745853255

Siguiente tuneo a probar: Probamos a quitar el dropout y poner lr a 0.0005

HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 20
LR_ADAM = 0.0005

RunID = 80b7df7f5d2b4d5daf4a4c428424968f
Presentan más o menos valores similares de loss, pero este último presenta f1_macros peores.
Vamos a probar a dejar el LR_ADAM de 0.001 y vamos a bajar el dropout a 0.0 con respecto al primer tuneo

Seguimos tuneando la primera run:
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 20
LR_ADAM = 0.001


"""

#Hiperparámetros
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.001



if __name__ == "__main__":
    pipeline = PipelineComunCRNN(
        nombre_dataset="chunked_diarizado_aumentado",
        pkl_train=pkl_train,
        pkl_test=pkl_test,
        batch_size=BATCH_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_capas_ocultas_lstm=NUM_LAYERS_LSTM,
        alpha_leaky_relu=ALPHA_LEAKY_RELU,
        is_bidirectional=IS_BIDIRECTIONAL,
        dropout=DROPOUT,
        lr_adam=LR_ADAM,
        num_epochs=NUM_EPOCHS
    )

    pipeline.ejecutar()
    
