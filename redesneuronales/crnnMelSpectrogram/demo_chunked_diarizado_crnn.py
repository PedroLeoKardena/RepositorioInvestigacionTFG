from tune_pipeline_comun import PipelineComunCRNN

pkl_train = "train_mels_chunked_diarizado.pkl"
pkl_test = "test_mels_chunked_diarizado.pkl"
#Hiperparámetros
#BATCH_SIZE = 32 #El batch size de los data loaders. https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
#HIDDEN_SIZE = 128 #El numero de neuronas en la capa oculta de la LSTM: https://docs.pytorch.org/docs/2.12/generated/torch.nn.LSTM.html
#NUM_LAYERS_LSTM = 1 #El numero de capas ocultas en la LSTM. En este caso, al ser 1, no habrá apilamiento de capas LSTM: https://docs.pytorch.org/docs/2.12/generated/torch.nn.LSTM.html
#ALPHA_LEAKY_RELU = 0.01 #El valor de alpha para la función de activación Leaky ReLU. https://docs.pytorch.org/docs/2.12/generated/torch.nn.LeakyReLU.html
#IS_BIDIRECTIONAL = False #Establecemos si la LSTM es bidireccional o no: https://docs.pytorch.org/docs/2.12/generated/torch.nn.LSTM.html
#DROPOUT = 0.0 #Añade una capa de dropout entra las capas del LSTM (DROPOUT = REGULARIZACION). https://docs.pytorch.org/docs/2.12/generated/torch.nn.LSTM.html
#LR_ADAM = 0.001 #La tasa de aprendizaje para el optimizador Adam. https://docs.pytorch.org/docs/2.12/generated/torch.optim.Adam.html
#NUM_EPOCHS = 20 #Numero de epocas de entrenamiento.
#Mantenemos Hidden_size y aumentamos num_epochs. Se podría probar a la siguiente a aumentar el LR_ADAM a 0.0005 o 0.001

#Tambien aumentamos el dropout, ya que el val loss sube mientras que el train loss baja. Dropout es lo que permite trabajar en contra
#del overfitting. Ponemos a 0.3

"""
Después de los tuneos básicos, el mejor tuneo es el siguiente:
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 1
ALPHA_LEAKY_RELU = 0.01
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 20
LR_ADAM = 0.001

En mlflow = CRNN_MelSpectrogram_chunked_hidden128 = 4f7c5ed022d1453a9f627a383337f962
Presenta el mejor f1_macro para caja o grupo y un  menor val_loss que los demás.
cv_mean_loss = 1.957083404858907
cv_mean_val_f1_grupo = 0.33874221723643955
cv_mean_val_f1_caja = 0.23628785436830518

Siguiente tuneo a probar: Ponemos 2 layers, probamos un leaky rely de 0.0 y un n_epochs de 50
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 50
LR_ADAM = 0.001

run_id = 7726e2ebb4fd41dc88fe93d955c9f097
Esta ultima es una peor run. Probamos la siguiente config a continuacion:
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
        nombre_dataset="chunked_diarizado",
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
    
