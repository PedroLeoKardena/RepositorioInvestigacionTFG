from tune_pipeline_comun import PipelineComunCRNN

pkl_train = "train_mfcc_chunked_aumentado_diarizado.pkl"
pkl_test = "test_mfcc_chunked_aumentado_diarizado.pkl"

"""
Después de los tuneos básicos, el mejor tuneo para grupo es el siguiente:
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 1
ALPHA_LEAKY_RELU = 0.01
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 20
LR_ADAM = 0.001

RunID = 655f02b171124849950a867aaa771f1a

Presenta el mejor f1_macro grupo, además de un muy buen val_loss.
cv_mean_loss = 2.0967487896434847
cv_mean_val_f1_grupo = 0.3355031666923745

HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.001

RunID = b4c4a947cccb4bf3b363f19e0af34e20
cv_mean_val_f1_caja = 0.2810336360545458
cv_mean_loss = 2.1069079949363827


-------------------------Para caja:-----------------------------------
Quitamos el dropout, si aumenta mucho el loss en val y no mejora el rendimiento, lo volvemos a poner
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 25
LR_ADAM = 0.001

run_id = d51576c23cc9418d96844ad08d1813e1
cv_mean_val_f1_caja = 0.2371375004825734
cv_mean_val_loss = 2.0741511293441532

Presenta un menor loss pero este valor tiende a fluctuar bastante. Además, presenta un peor f1.

Vamos a probar a poner dropout (0.3)
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.3
NUM_EPOCHS = 25
LR_ADAM = 0.001

-----------------------------Para grupo:------------------------------------------
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 30
LR_ADAM = 0.001

run_id = e7cb9437f8804c8ab0ddffcfbe02c824
cv_mean_val_f1_grupo = 0.2882565146501531
cv_mean_loss = 2.1003719167482284

Presenta peor f1 y loss. Vamos a probar a dejar el num_epochs en 20 y a poner un dropout de 0.15.
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 20
LR_ADAM = 0.001
"""

#Hiperparámetros
#Primero ponemos los de grupo, luego hacemos los de caja:

HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 20
LR_ADAM = 0.001



if __name__ == "__main__":
    pipeline = PipelineComunCRNN(
        nombre_dataset="chunked_aumentado_diarizado",
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
    
