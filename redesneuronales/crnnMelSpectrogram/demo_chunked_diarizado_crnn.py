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

"""
Hiperparametros tuneo grupo: f1 = 0.8589392943239096
hidden_size 256
batch_size 32
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.01
is_bidirectional True
dropout 0.0
num_epochs 50


Hiperparametros tuneo 2: f1 = 0.7337250554323725
hidden_size 256
batch_size 32
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.01
is_bidirectional True
dropout 0.3
num_epochs 75
lr_adam 0.0001

#Vemos que ha empeorado al subir el número de epochs y establecer cierto valor de dropout. Vamos a probar a dejar el numero de epochs a 50 y solo con dropout a 0.3 y aumentando el LR a 0.0005.

Post-tuneo 3: f1 = 0.824388327721661. Ha mejorado pero sigue siendo peor que el primer tuneo.
hidden_size 256
batch_size 32
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.01
is_bidirectional True
dropout 0.3
num_epochs 50
lr_adam 0.0005

Incluso si analizamos el val loss, el primero es mejor que este post-tuneo 3.
Vamo a probar entonces a no poner dropout, dejar el numero de epochs a 50 y bajar el LR a 0.0005 (el primero lo tenia a 0.001).


Post-tuneo 4: f1 = 0.8292121212121212. Sigue siendo peor que el primer tuneo.
hidden_size 256
batch_size 32
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.01
is_bidirectional True
dropout 0.0
num_epochs 50
lr_adam 0.0005

Vamos a probar ahora con un lr mayor a 0.001 como 0.005, y un leaky_relu a 0.0 Lo demás se mantiene igual.

Post-tuneo 5: f1 = 0.8586666666666667. Ha mejorado bastante. Es levemente peor al tuneo inicial. Vamos a probar a dejar el lr en 0.001
hidden_size 256
batch_size 32
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.01
is_bidirectional True
dropout 0.0
num_epochs 50
lr_adam 0.005

Post-tuneo 6: 0.7941287878787878. Ha empeorado.
hidden_size 256
batch_size 32
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.0
is_bidirectional True
dropout 0.0
num_epochs 50
lr_adam 0.001

Vamos a probar el tuneo inicial pero con un leaky_relu mayor a 0.01, como de 0.05.

Post-tuneo 7: f1 grupo = 0.6982001150086257. Ha empeorado bastante.
hidden_size 256
batch_size 32
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.05
is_bidirectional True
dropout 0.0
num_epochs 50
lr_adam 0.001

Entonces vamos a dejar todo igual al mejor tuneo, y ahora vamos a probar a subir el batch_size a 64.

Post-tuneo 8: f1 grupo = 0.6532828282828282 Sigue siendo peor
hidden_size 256
batch_size 64
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.05
is_bidirectional True
dropout 0.0
num_epochs 50
lr_adam 0.001


A la proxima probaremos a poner hidden_size 512 con dicho batch_size 64.

Post-tuneo 9: f1 grupo = 0.7912524983344437. No mejora la mejor configuración.
hidden_size 512
batch_size 64
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.05
is_bidirectional True
dropout 0.0
num_epochs 50
lr_adam 0.001

Ahora probamos con leaky_relu = 0.01.

Post-tuneo 10: f1 grupo = 0.7735053763440861. No mejor ni al de antes
hidden_size 512
batch_size 64
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.05
is_bidirectional True
dropout 0.0
num_epochs 50
lr_adam 0.001

De este modo podemos establecer que la configuración final será:
hidden_size 256
batch_size 32
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.01
is_bidirectional True
dropout 0.0
num_epochs 50
lr_adam 0.001
"""

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

En mlflow = CRNN_MelSpectrogram_chunked_hidden128
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

"""

HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 50
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
    
