from tune_pipeline_comun import PipelineComunCRNN

pkl_train = "train_mfcc_chunked_aumentado_diarizado.pkl"
pkl_test = "test_mfcc_chunked_aumentado_diarizado.pkl"

"""

MEJOR GRUPO:
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

run_id = 437346e68f5e490ab0a29a9d10a02395
cv_mean_val_f1_caja = 0.22459152500278598
cv_mean_val_loss = 2.0954046663405403
cv_mean_std_loss = 0.3701491630691365

Sigue empeorando: Dejaremos el dropout a 0.15 y meteremos lr_adam a 0.0005.

MEJOR CAJA:
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.0005

run_id = fc29a0027c8c4645b7342879d381e938
cv_mean_val_f1_caja = 0.3055628219497569
cv_mean_val_loss = 2.071359636140248
cv_mean_std_loss = 0.2946508781956708

Es mejor que el original incluso. Entonces, vamos a probar ahora con un lr_adam mayor, simplemente para comprobar:

HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.003

run_id = f25981a75bd84862b00a8da4a2860456
cv_mean_val_f1_caja = 0.204238530967495
cv_mean_val_loss = 2.0781752541140905
cv_mean_std_loss = 0.2586610965694405


Vamos a probar con un alpha_leaky_relu superior:
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.05
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.0005

run_id = fecc569578fe49b7ac46a6afb720bd09
cv_mean_val_f1_caja = 0.28338918941524277
cv_mean_val_loss = 2.0781752541140905
cv_mean_std_loss = 0.2586610965694405


Es levemente peor al mejor. Probaremos un alpha algo menor, si no mejora dejamos dicha config:

HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.025
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.0005

run_id = 36bec0c366bf4fb4a22a8c1ede343cb4
cv_mean_val_f1_caja = 0.26391058867029416
cv_mean_val_loss = 2.0604885061960374
cv_mean_std_loss = 0.2943540911226361

No mejora. Esta es la configuración final de caja:

Final Caja:

HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.0005

run_id = fc29a0027c8c4645b7342879d381e938
cv_mean_val_f1_caja = 0.3055628219497569
cv_mean_val_loss = 2.071359636140248
cv_mean_std_loss = 0.2946508781956708


Elegimos esta para caja

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

run_id = e615219ff3fd4eb5b2eee6f8927f919f
cv_mean_val_f1_grupo = 0.3127086285183544
cv_mean_loss = 2.019415545898771
cv_std_loss = 0.2302773828291568

Ha mejorado bastante. Vamos a probar a aumentar el dropout solamente.
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.3
NUM_EPOCHS = 25
LR_ADAM = 0.001

run_id = bb959b72e8e0449a9fb8a4322efb3d84
cv_mean_val_f1_grupo = 0.30040008115471256
cv_mean_loss = 2.1353802714272154
cv_std_loss = 0.1686790717534015

Ha empeorado levemente y sigue siendo peor que el original. Vamos a probar la configuración original, pero con un lr menor.
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 1
ALPHA_LEAKY_RELU = 0.01
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 20
LR_ADAM = 0.0005

run_id = c3486033de7a412a89853a9e3b8b5f23
cv_mean_val_f1_grupo = 0.288749046603305
cv_mean_loss = 2.1687975772600323
cv_std_loss = 0.17993286748285506

Sigue empeorando. Vamos a probar con un lr_adam un poco superior a 0.001 (0.0015)

HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 1
ALPHA_LEAKY_RELU = 0.01
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 20
LR_ADAM = 0.0015

run_id = 535ee86fb9394bb8b1d0756d939cfc83
cv_mean_val_f1_grupo = 0.28634728553142835
cv_mean_loss = 2.198776277000942
cv_std_loss = 0.17912942980581487


Sigue siendo pero que la config original, por lo que debemos dejar el lr a 0.001. Vamos a probar con un alpha_leaky relu mayor como 0.025.

HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 1
ALPHA_LEAKY_RELU = 0.025
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 20
LR_ADAM = 0.001


run_id = 4d5b98a3ef9346fd8bac8961cf088f89
cv_mean_val_f1_grupo = 0.29840315351059277
cv_mean_loss = 2.139639254165074
cv_std_loss = 0.21344389924411955

Sigue sin mejorar. Nos quedamos con esta como la mejor config para grupo:

HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 1
ALPHA_LEAKY_RELU = 0.01
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 20
LR_ADAM = 0.001

RunID = 655f02b171124849950a867aaa771f1a
cv_mean_loss = 2.0967487896434847
cv_mean_val_f1_grupo = 0.3355031666923745

"""

#Hiperparámetros
#Primero ponemos los de grupo, luego hacemos los de caja:
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.0005



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
    
