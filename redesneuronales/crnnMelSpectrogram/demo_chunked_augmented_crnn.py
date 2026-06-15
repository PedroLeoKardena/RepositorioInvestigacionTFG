from tune_pipeline_comun import PipelineComunCRNN

pkl_train = "train_mels_chunked_aumentado.pkl"
pkl_test = "test_mels_chunked_aumentado.pkl"

"""
Después de los tuneos básicos, el mejor tuneo para grupo es el siguiente:
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.001

RunID = f6d638ccb8824d4ab33dc44541df5ee3

Presenta el mejor f1_macro de grupo con muy baja std y un val loss bueno:

cv_mean_loss = 2.418780656999271
cv_std_loss = 0.3285143128505582
cv_mean_val_f1_grupo = 0.28078129054038614


MEJOR CONFIG CAJA:
En cuanto a caja, el mejor es:
HIDDEN_SIZE = 256
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.01
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 50
LR_ADAM = 0.001

run_id = dba04439400942b9a19773e2c9e5740a
cv_mean_loss = 2.600435559337085
cv_std_loss = 0.25785554077955974
cv_mean_val_f1_caja = 0.22685674814518547

Sin embargo presenta loss elevado.

-------------------------Para grupo:-----------------------------------

MEJOR CONFIGURACIÓN GRUPO:
Siguiente tuneo a probar: Ponemos un dropout de 0.3.
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.3
NUM_EPOCHS = 25
LR_ADAM = 0.001

Run id = 95a69bc2f9ce4051ad62e4e8f29bb28a
cv_mean_loss = 2.4344986568334592
cv_std_loss = 0.3678854928418136
cv_mean_val_f1_grupo = 0.28812258187289097

Presenta una leve mejora de rendimiento, tanto en loss como en f1. Vamos a dejar este dropout y vamos a cambiar el lr_adam.
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.3
NUM_EPOCHS = 25
LR_ADAM = 0.0005

Run id = 768e5381dfcb4739b8ddfc6dd7b6cb30
cv_mean_loss = 2.5112022961714326
cv_std_loss = 0.2930928729321515
cv_mean_val_f1_grupo = 0.26271936143859237

Ha empeorado

Podemos ver como ha empeorado, vamos a probar a en vez de bajar el lr_adam, a subirlo.
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.3
NUM_EPOCHS = 25
LR_ADAM = 0.003

run_id = a027fe30a5dc4a81a8001a868f8da797
cv_mean_loss = 2.3720745350912553
cv_std_loss = 0.229312579608923
cv_mean_val_f1_grupo = 0.26024185362758046

Sigue siendo peor que la mejor configuración de grupo. Vamos a probar un lr_adam un poco superior.

HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.3
NUM_EPOCHS = 25
LR_ADAM = 0.0015

run_id = 5a478b20a06149d8a045d24e5fbf0815
cv_mean_loss = 2.4956503671180768
cv_std_loss = 0.3647956356906456
cv_mean_val_f1_grupo = 0.2608732316092172

Mejora muy poco el f1_grupo con respecto al anterior y empeora el loss. Sigue siendo peor que el mejor.
Vamos a dejar el lr a 0.001 y vamos a probar un mayor alpha leaky relu que 0.01.

HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.025
IS_BIDIRECTIONAL = True
DROPOUT = 0.3
NUM_EPOCHS = 25
LR_ADAM = 0.001

run_id = e2ad022eb37941c1b8e640346e23f2e7
cv_mean_loss = 2.400859580188179
cv_std_loss = 0.263050438328603
cv_mean_val_f1_grupo = 0.24543800097153584


Observamos que los valores son peores. Vamos a probar con un num_layers_lstm=3:
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 3
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.3
NUM_EPOCHS = 25
LR_ADAM = 0.001

run_id = a741229304894a7cbb47aedf7a80c672
cv_mean_loss = 2.3018144221591137
cv_std_loss = 0.261402717383965
cv_mean_val_f1_grupo = 0.2528271453041754

Sigue siendo peor que la mejor configuración, por lo que nos quedamos con esta como configuración final.

Final Grupo:
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.3
NUM_EPOCHS = 25
LR_ADAM = 0.001

Run id = 95a69bc2f9ce4051ad62e4e8f29bb28a
cv_mean_loss = 2.4344986568334592
cv_std_loss = 0.3678854928418136
cv_mean_val_f1_grupo = 0.28812258187289097



--------------------Para caja:------------------
Para pelear con el loss vamos a aumentar el valor de dropout, además de poner un leaky_relu menor.
HIDDEN_SIZE = 256
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 50
LR_ADAM = 0.001

Run id = 17fcf99db88245d383b57cb36aa91ade
cv_mean_loss = 2.5867745020901336
cv_std_loss = 0.3383189155883087
cv_mean_val_f1_caja = 0.17671055324984905

Vemos que ha empeorado bastante. Por ello vamos a probar a solo bajar el leaky_relu y num_epochs. Todo igual menos leaky_relu a 0.0 y epochs = 25.
HIDDEN_SIZE = 256
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 25
LR_ADAM = 0.001

Run id = c8f3f6511e6d43da8197ee92bd55f4a4
cv_mean_loss = 2.474314739281655
cv_std_loss = 0.22482645195325288
cv_mean_val_f1_caja = 0.20434699512733276

Es peor que el original. Vamos a probar misma configuración pero sin leaky relu.
HIDDEN_SIZE = 256
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 50
LR_ADAM = 0.001

Run id = dc4b9e19d9e54421aaec3d5144684e16
cv_mean_loss = 2.5379107041574964
cv_std_loss = 0.22924396868088856
cv_mean_val_f1_caja = 0.1876310387552151

Ha empeorado.
Esto significa que no deberíamos bajar el leay relu. Vamos a probar a aumnetarlo a 0.02.
HIDDEN_SIZE = 256
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.02
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 50
LR_ADAM = 0.001

Run id = 3c3bdf8abf86414087de0e6dfd494bbb
cv_mean_loss = 2.5568711385502345
cv_std_loss = 0.24191709741905604
cv_mean_val_f1_caja = 0.2019857076524175

Sigue siendo peor. Nos quedamos con esta configuración final para la caja:

FINAL CAJA:

HIDDEN_SIZE = 256
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.01
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 50
LR_ADAM = 0.001

run_id = dba04439400942b9a19773e2c9e5740a
cv_mean_loss = 2.600435559337085
cv_std_loss = 0.25785554077955974
cv_mean_val_f1_caja = 0.22685674814518547
"""


#Primero probamos grupo luego caja.
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 3
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.3
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
    
