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
IS_BIDIRECTIONAL = False
DROPOUT = 0.0
NUM_EPOCHS = 20
LR_ADAM = 0.001

RunID = cd7728fa3b6b4634ae246416186caa55
cv_mean_loss = 2.002497133016586
cv_std_val_loss = 0.1654692220291475
cv_mean_val_f1_grupo = 0.3607690763734507
cv_mean_val_f1_caja = 0.21268221555523414

HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.001

RunID = c97e358df0a046eaa0c71c6da1b740bf
cv_mean_loss = 1.9924622770150506
cv_std_val_loss = 0.19797342406844018
cv_mean_val_f1_grupo = 0.3178964214533529
cv_mean_val_f1_caja = 0.18343489452042158

Ha empeordao el tras añadir la capa extra y hacerlo bidireccional. Vamos a probar a dejar la base, es decir, sin bidireccional, 
pero con 2 capas, con dropout y leaky_relu a 0.0.

HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = False
DROPOUT = 0.15
NUM_EPOCHS = 20
LR_ADAM = 0.001

run_id = c9c2fca3260f44c387d89152c6faaa69
cv_mean_loss = 1.9455500733852389
cv_std_val_loss = 0.22442453157147316
cv_mean_val_f1_grupo = 0.31606069489674715
cv_mean_val_f1_caja = 0.19670420706258568



Sigue siendo peor que el original. Vamos entonces a probar esta configuaración con un menor lr_adam.

MEJOR CONFIGURACIÓN:
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 1
ALPHA_LEAKY_RELU = 0.01
IS_BIDIRECTIONAL = False
DROPOUT = 0.0
NUM_EPOCHS = 20
LR_ADAM = 0.0005

run_id = d06346a000694b8595c25f3d5ed79fbd
cv_mean_loss = 1.9249431475003562
cv_std_val_loss = 0.1323528074972972
cv_mean_val_f1_grupo = 0.368520770777634
cv_mean_val_f1_caja = 0.2300929468601495

Esta es ahora la mejor configuración, vamos a probar a aumentar el alpha_leaky_relu:

HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 1
ALPHA_LEAKY_RELU = 0.015
IS_BIDIRECTIONAL = False
DROPOUT = 0.0
NUM_EPOCHS = 20
LR_ADAM = 0.0005

run_id = 67c748ac8221436781dabc3723ef8c83
cv_mean_loss = 1.917386575539907
cv_std_val_loss = 0.1477278978458194
cv_mean_val_f1_grupo = 0.3426353454352481
cv_mean_val_f1_caja = 0.1881259745002006


Es peor que la mejor configuración, vamos a probar a aumentarlo un poco más, si no funciona dejamos dicha configuración:
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 1
ALPHA_LEAKY_RELU = 0.025
IS_BIDIRECTIONAL = False
DROPOUT = 0.0
NUM_EPOCHS = 20
LR_ADAM = 0.0005


run_id = 060bd9a813ea4ab28f2bf377c51118ea
cv_mean_loss = 1.889265093008677
cv_std_val_loss = 0.14185951485946302
cv_mean_val_f1_grupo = 0.32054778229857844
cv_mean_val_f1_caja = 0.25635988227143036


Pues nos vamos a quedar con esta configuración para la caja:

FINAL CAJA:
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 1
ALPHA_LEAKY_RELU = 0.025
IS_BIDIRECTIONAL = False
DROPOUT = 0.0
NUM_EPOCHS = 20
LR_ADAM = 0.0005

run_id = 060bd9a813ea4ab28f2bf377c51118ea
cv_mean_loss = 1.889265093008677
cv_std_val_loss = 0.14185951485946302
cv_mean_val_f1_caja = 0.25635988227143036


FINAL GRUPO:
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 1
ALPHA_LEAKY_RELU = 0.01
IS_BIDIRECTIONAL = False
DROPOUT = 0.0
NUM_EPOCHS = 20
LR_ADAM = 0.0005

run_id = d06346a000694b8595c25f3d5ed79fbd
cv_mean_loss = 1.9249431475003562
cv_std_val_loss = 0.1323528074972972
cv_mean_val_f1_grupo = 0.368520770777634


Este será utilizado para grupo.
"""

HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 1
ALPHA_LEAKY_RELU = 0.01
IS_BIDIRECTIONAL = False
DROPOUT = 0.0
NUM_EPOCHS = 20
LR_ADAM = 0.0005


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
    
