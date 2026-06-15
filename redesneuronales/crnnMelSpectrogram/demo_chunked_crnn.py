from tune_pipeline_comun import PipelineComunCRNN

pkl_train = "train_mels_chunked.pkl"
pkl_test = "test_mels_chunked.pkl"

"""

MEJOR CONFIGURACIÓN:
Después de los tuneos básicos, el mejor tuneo es el siguiente:
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.001


En mlflow  = 69fc580605134e28903adb395b1f3a18
cv_mean_loss = 2.2548182936509447
cv_std_loss = 0.32366897305701176
cv_mean_val_f1_grupo = 0.25929679300592073
cv_mean_val_f1_caja = 0.2082380620949693

Tiene buenas performances para algunos folds, pero para otros se queda corto, creemos que podemos aumentar el dropout para mejorar
la capacidad de generalización.


Siguiente tuneo a probar: Aumentar dropout
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.3
NUM_EPOCHS = 25
LR_ADAM = 0.001

Run_id = e9c94ae624bc48b1b54c275d3b0a2e17

cv_mean_loss = 2.341840048631032
cv_std_loss = 0.31931970447131824
cv_mean_val_f1_grupo = 0.23865180984473855
cv_mean_val_f1_caja = 0.19501263732361956

Ha empeorado, asi que vamos a dejar el dropout a 0.15 y vamos a probar con un lr_adam de 0.0005.
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.0005

run id =  8dffdff3c0c54219b05c885bc0e395db

cv_mean_loss = 2.2219053041934966
cv_std_loss = 0.27321503478616566
cv_mean_val_f1_grupo = 0.2366485759213024
cv_mean_val_f1_caja = 0.19607297256970913

Ha empeorado un poco. 
Vamos entonces a probar lo contrario, ha subir el lr adam.
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.003

run id = f4f383bbba294cb7a888f01e89eded0a
cv_mean_loss = 2.291663714647293
cv_std_loss = 0.2777058585556113
cv_mean_val_f1_grupo = 0.2056978244984736
cv_mean_val_f1_caja = 0.1954439264959921

Vamos a probar con un valor un poco mas bajo pero superior a 0.001:
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.0015

run id = fd4976617ec44baea11249023d31133b
cv_mean_loss = 2.259644584655762
cv_std_loss = 0.28891504467435
cv_mean_val_f1_grupo = 0.25579496248413686
cv_mean_val_f1_caja = 0.19805493112385622

Es un poco peor que el mejor, por lo que vamos a dejar el lr_adam a 0.001. Vamos a probar a configurar por ultimo el alpha leaky relu.

HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.025
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.001
"""

#Hiperparámetros
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.025
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.001



if __name__ == "__main__":
    pipeline = PipelineComunCRNN(
        nombre_dataset="chunked",
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
    
