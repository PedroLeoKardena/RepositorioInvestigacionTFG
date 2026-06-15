from tune_pipeline_comun import PipelineComunCRNN

pkl_train = "train_mfcc_chunked_diarizado.pkl"
pkl_test = "test_mfcc_chunked_diarizado.pkl"
#Hiperparámetros
#BATCH_SIZE = 16 #El batch size de los data loaders. https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
#HIDDEN_SIZE = 128 #El numero de neuronas en la capa oculta de la LSTM: https://docs.pytorch.org/docs/2.12/generated/torch.nn.LSTM.html
#NUM_LAYERS_LSTM = 1 #El numero de capas ocultas en la LSTM. En este caso, al ser 1, no habrá apilamiento de capas LSTM: https://docs.pytorch.org/docs/2.12/generated/torch.nn.LSTM.html
#ALPHA_LEAKY_RELU = 0.01 #El valor de alpha para la función de activación Leaky ReLU. https://docs.pytorch.org/docs/2.12/generated/torch.nn.LeakyReLU.html
#IS_BIDIRECTIONAL = False #Establecemos si la LSTM es bidireccional o no: https://docs.pytorch.org/docs/2.12/generated/torch.nn.LSTM.html
#DROPOUT = 0.0 #Añade una capa de dropout entra las capas del LSTM (DROPOUT = REGULARIZACION). https://docs.pytorch.org/docs/2.12/generated/torch.nn.LSTM.html
#LR_ADAM = 0.001 #La tasa de aprendizaje para el optimizador Adam. https://docs.pytorch.org/docs/2.12/generated/torch.optim.Adam.html
#NUM_EPOCHS = 20 #Numero de epocas de entrenamiento.

"""

MEJOR CONFIG:
Después de los tuneos básicos, el mejor tuneo para grupo es el siguiente:

HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.001

RunID = b9f56b79280747b7a2fb918862694877

Presenta uno de los mejores f1_macro para grupo, el mejor f1_macro para caja, además de un muy buen val_loss, 
con un std alto que suele tender valores menores.
cv_mean_loss = 1.9648182272911072
cv_std_loss = 0.3030990735271277
cv_mean_val_f1_grupo = 0.3476013096052434
cv_mean_val_f1_caja = 0.2689190130085076


Siguiente tuneo:
Quitamos el dropout, si aumenta mucho el loss en val y no mejora el rendimiento, lo volvemos a poner
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 25
LR_ADAM = 0.001

run_id = 74e20bd14b4346c3aeba2b9237abcdbe
cv_mean_loss = 1.959678914149602
cv_std_loss = 0.326580201896546
cv_mean_val_f1_grupo = 0.3080643286060253
cv_mean_val_f1_caja = 0.1828525107144428


Ha empeorado basatante al quitar el dropout, ponemos mas.

HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.3
NUM_EPOCHS = 25
LR_ADAM = 0.001

run_id = 8a69eac20d7243819743b69443859a61
cv_mean_loss = 1.9595704833666485
cv_std_loss = 0.33180207778548376
cv_mean_val_f1_grupo = 0.28903776465015313
cv_mean_val_f1_caja = 0.19375396718455318

Vemos que en general, aumentar tanto el dropout no es tan bueno. Vamos a dejarlo en 0.15 y poniendo lr_adam a 0.0005
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.0005

RUN_ID = fd6bb11deeed41708e1a5f938157b925
cv_mean_loss = 1.9526579260826111
cv_std_loss = 0.285775152309748
cv_mean_val_f1_grupo = 0.3189058776150687
cv_mean_val_f1_caja = 0.24374296486175848

Sigue siendo peor que el original, por lo que cogeremos el original y probaremos un lr_adam mayor a 0.001:

HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.003

RUN_ID = 9e8ad7e333554d4584ea4d29555963c2
cv_mean_loss = 1.9754369537035625
cv_std_loss = 0.29848262165069755
cv_mean_val_f1_grupo = 0.31990194243634507
cv_mean_val_f1_caja = 0.25435544864702775

Sigue siendo levemente peor al original. Vamo s aprobar con un lr_adam entre 0.003 y 0.001. Como 0.0015

HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.0015

RUN_ID = c9c1eec553af45ef89b44bf105aafce6
cv_mean_loss = 1.9902945339679718
cv_std_loss = 0.36834338055344074
cv_mean_val_f1_grupo = 0.2703036391055714
cv_mean_val_f1_caja = 0.26957881423379326

Solo ha mejorado levemente la caja pero presenta un poco peor loss. Vamos a dejar el lr original y vamos a probar un alpha superior.

HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.025
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.001



RUN_ID = 2535eb86b0b140468ad0f2fe27ca6fed
cv_mean_loss = 1.9282065828641255
cv_std_loss = 0.30988175461218
cv_mean_val_f1_grupo = 0.296116170283543
cv_mean_val_f1_caja = 0.24956036472270976


Sigue siendo mucho peor que la mejor configuración, por lo que nos quedamos con dicha configuración como la final:


Config Final:
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.001

RunID = b9f56b79280747b7a2fb918862694877
cv_mean_loss = 1.9648182272911072
cv_std_loss = 0.3030990735271277
cv_mean_val_f1_grupo = 0.3476013096052434
cv_mean_val_f1_caja = 0.2689190130085076
"""

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
    
