from tune_pipeline_comun import PipelineComunCRNN

pkl_train = "train_mels_chunked_aumentado_diarizado.pkl"
pkl_test = "test_mels_chunked_aumentado_diarizado.pkl"

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

En mlflow = bf832778994444e0af6987a507c625ff
Presenta el mejor f1_macro para caja o grupo y un val_loss equivalente a los demás.
cv_mean_loss = 2.19969505618489
cv_std_loss = 0.04825595020368651
cv_mean_val_f1_grupo = 0.3075294609241225
cv_mean_val_f1_caja = 0.18534202554130771


HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.001

run_id = 6646b638e8e64c5d97b0dfe2c22836e9
cv_mean_loss = 2.048484472925701
cv_std_loss = 0.23625576176375596
cv_mean_val_f1_grupo = 0.3204337440765794
cv_mean_val_f1_caja = 0.23598364847513548

Ha mejorado basante. Vamos a probar a aumentar el dropout a 0.3:

MEJOR CONFIGURACIÓN:
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.3
NUM_EPOCHS = 25
LR_ADAM = 0.001

run_id = d80a0197984e404eaeaf2be31a033c1b
cv_mean_loss = 2.158648055602634
cv_std_loss = 0.17492375726904083
cv_mean_val_f1_grupo = 0.3244227273187449
cv_mean_val_f1_caja = 0.26337580099882024



Ha mejorado sobre todo en caja, sin comprometer mucho el loss, por lo cual nos quedaremos con el dropout a 0.3. Vamos a probar
un lr menor:
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.3
NUM_EPOCHS = 25
LR_ADAM = 0.0005

run_id = 8653bd533ed54b95b4303a4c01fa8637
cv_mean_loss = 2.1670822729193975
cv_std_loss = 0.1419969087449553
cv_mean_val_f1_grupo = 0.26929908989989737 
cv_mean_val_f1_caja = 0.2152639648053071

Vemos como sigue siendo peor que la mejor configuración, vamos a probar un lr un poco superior:
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.3
NUM_EPOCHS = 25
LR_ADAM = 0.0015

run_id = 892690fe57074cae84af5d371f0f25d7
cv_mean_loss = 2.1549369000631664
cv_std_loss = 0.23983201940782206
cv_mean_val_f1_grupo = 0.2970596268192356
cv_mean_val_f1_caja = 0.17347443212088515

Sigue siendo peor que la mejor configuración. Dejamos dicha configuración y cambiamos el alpha leaky relu a 0.025:

HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.025
IS_BIDIRECTIONAL = True
DROPOUT = 0.3
NUM_EPOCHS = 25
LR_ADAM = 0.001

run_id = cabd3ab5e98a4017bfdf769cadd3d80a
cv_mean_loss = 2.175201999195038
cv_std_loss = 0.2939554305914575
cv_mean_val_f1_grupo = 0.32936036274328984
cv_mean_val_f1_caja = 0.1864304488385071


Sigue siendo peor que la mejor configuración, sobre todo en la caja. Vamos a probar a subir el num_layers a 3:
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 3
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.3
NUM_EPOCHS = 25
LR_ADAM = 0.001

run_id = a6ef77e379224d9596bd29c29e9caf0d
cv_mean_loss = 2.0333766704324696
cv_std_loss = 0.32789496456245215
cv_mean_val_f1_grupo = 0.3121347781546918
cv_mean_val_f1_caja = 0.18197455928146353

Como podemos ver el rendimiento es peor. Por este motivo vamos a dejar la configuración mejor como la final:

HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.3
NUM_EPOCHS = 25
LR_ADAM = 0.001

run_id = d80a0197984e404eaeaf2be31a033c1b
cv_mean_loss = 2.158648055602634
cv_std_loss = 0.17492375726904083
cv_mean_val_f1_grupo = 0.3244227273187449
cv_mean_val_f1_caja = 0.26337580099882024

Este será utilizado para caja.

"""

#Hiperparámetros
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.3
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
    
