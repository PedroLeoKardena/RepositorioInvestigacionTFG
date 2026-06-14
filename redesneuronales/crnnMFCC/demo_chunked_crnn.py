from tune_pipeline_comun import PipelineComunCRNN

pkl_train = "train_mfcc_chunked.pkl"
pkl_test = "test_mfcc_chunked.pkl"

"""
Después de los tuneos básicos, el mejor tuneo para caja es el siguiente:
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.001

RunID = e5ee6f9180bf4827aafafffda75d291c

cv_mean_loss = 2.2472401396433512
cv_mean_val_f1_caja = 0.22654179906268973

Para grupo:
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 1
ALPHA_LEAKY_RELU = 0.01
IS_BIDIRECTIONAL = False
DROPOUT = 0.0
NUM_EPOCHS = 20
LR_ADAM = 0.001

RunID = 67bf4bc7e45c47bd8e00af5038bc8b3d
cv_mean_loss = 2.281558305889497
cv_mean_val_f1_grupo = 0.24382322762473888


Entonces, tenemos que probar dos tuneos:

---------------------Para caja:----------------------
Quitamos el dropout, si aumenta mucho el loss en val y no mejora el rendimiento, lo volvemos a poner
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 25
LR_ADAM = 0.001

run_id = b231563610ab4cf4876d5045a38c735f
cv_mean_loss = 2.274390032688777
cv_mean_val_f1_caja = 0.18342886044928386

Presenta un peor f1 de caja y un peor valor de loss.
Vamos a poner entonces mas dropout:
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.3
NUM_EPOCHS = 25
LR_ADAM = 0.001

--------------Para grupo:----------------------
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 30
LR_ADAM = 0.001

run_id = 3a7821a27aa646c0a5c867c88bd92ce3
cv_mean_val_f1_grupo = 0.2910817452236009
cv_mean_loss = 2.2457444698177547

Presenta un mucho mejor valor de f1 de grupo. Seguiremos por esta linea. Vamos a probar a aumentar el dropout.
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.3
NUM_EPOCHS = 30
LR_ADAM = 0.001
"""

#Hiperparámetros. Dejamos el numero de epochs igual que antes y aumentamos dropout para intentar mejorar el val_loss.
#Aumentamos tambien el LR_adam a 0.0005. A la siguiente buscamos aumentar el hidden_size.

#Primero se prueba grupo, luego caja.
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.3
NUM_EPOCHS = 30
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
    
