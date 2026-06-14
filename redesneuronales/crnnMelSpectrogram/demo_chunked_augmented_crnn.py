from tune_pipeline_comun import PipelineComunCRNN

pkl_train = "train_mels_chunked_aumentado.pkl"
pkl_test = "test_mels_chunked_aumentado.pkl"

"""
Después de los tuneos básicos, el mejor tuneo es el siguiente:
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 1
ALPHA_LEAKY_RELU = 0.01
IS_BIDIRECTIONAL = False
DROPOUT = 0.0
NUM_EPOCHS = 20
LR_ADAM = 0.001

En mlflow = CRNN_MelSpectrogram_chunked_aumentado_hidden128. RunID = 3c3fb3ceabc045c7b948adae3b9b9552

Buscar run = attributes.run_id IN ("3c3fb3ceabc045c7b948adae3b9b9552")

Aun que no presenta el absoluto mejor f1_macro para grupo, si que presenta un mucho menor val_loss que los demás
que presentan un f1_macro equiparable. Si presenta el mejor de caja.
cv_mean_loss = 2.423198068672027
cv_mean_val_f1_grupo = 0.2820297311359166
cv_mean_val_f1_caja = 0.20612217438134217

Siguiente tuneo a probar: Ponemos 2 layers, probamos un leaky rely de 0.0 y un n_epochs de 50
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 50
LR_ADAM = 0.001

Este tuneo era peor que el anterior. Presentaba un mayor loss y encima, un peor rendimiento. Run ID = 1b22700833d143f79ae3eb158ee07c91 
Vamos a probar a bajar numero de epochs y dejar el leaky relu a 0.0:
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 20
LR_ADAM = 0.001
"""

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
        nombre_dataset="chunked_aumentado",
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
    
