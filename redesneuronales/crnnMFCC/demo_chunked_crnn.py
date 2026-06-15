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

run_id = 003451a6de5c4b348b77cbbe3a129ff7
cv_mean_loss = 2.2797300883134204
cv_std_loss = 0.24282156773818653
cv_mean_val_f1_caja = 0.20307458118562277

Sigue siendo peor que el original. Por ello, lo que haremos sera poner el original, dejar el dropout y probar a cambiar el lr_adam:
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.0005

run_id = a8d97cd8dacf4677acdb355fca663337
cv_mean_loss = 2.2546072975794473
cv_std_loss = 0.2328297102884034
cv_mean_val_f1_caja = 0.23637453462099772

Vemos como ha mejorado bastante al disminuir el valor de lr, incluso superando al original. 
Vamos a probar ahora a subir el lr_adam, para ver si con uno mayor funciona mejor.


HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.003

run_id = 417b3c555f374a8ea713a0af03b3ced7
cv_mean_loss = 2.2473302104075747
cv_std_loss = 0.252022385932769
cv_mean_val_f1_caja = 0.21373983043111483

Como podemos entonces comprobar el mejor lr adam es 0.0005. Vamos a probar a configurar un mayor alpha:
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.05
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.0005

run_id = b002d5aec22d4dc9adae6c7b8aa8a7bc
cv_mean_loss = 2.287770094474157
cv_std_loss = 0.24756008148670364
cv_mean_val_f1_caja = 0.2419877183998453

Es mejor. Vamos a probar con un alpha un poco menor, si vemos que empeora intentamos mejorarlo.
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.025
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.0005

run_id = b002d5aec22d4dc9adae6c7b8aa8a7bc
cv_mean_loss = 2.2807329048713045
cv_std_loss = 0.3480101440916815
cv_mean_val_f1_caja = 0.23100968518364864

Vemos que es peor. Vamos a probar con un alpha un poco mayor a 0.05

HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.075
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.0005

run_id = e2320731fbb544f49d4088022c0ff71f
cv_mean_loss = 2.265785546898842
cv_std_loss = 0.2550489717548564
cv_mean_val_f1_caja = 0.1895457954783833

CONFIGURACIÓN FINAL CAJA:
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.05
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.0005

run_id = b002d5aec22d4dc9adae6c7b8aa8a7bc
cv_mean_loss = 2.287770094474157
cv_std_loss = 0.24756008148670364
cv_mean_val_f1_caja = 0.2419877183998453


--------------Para grupo:----------------------

MEJOR Grupo: 
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

RUN_ID = 9e68fda3a7cc4c6bb4dd8af00974bed3
cv_mean_val_f1_grupo = 0.20059013115172494
cv_mean_loss = 2.3132559384962526
cv_std_loss = 0.18616889079940588
Ha empeorado bastante tras poner el dropout a 0.3.

Vamos a probar a bajarlo a 0.15 y a bajar el num_epochs a 25.
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.001

RUN_ID = 93219a09b34840559e18f5f7138ef5e6
cv_mean_loss = 2.333240867245937
cv_std_loss = 0.23683812774669924
cv_mean_val_f1_grupo = 0.20100836484345797

Vemos como ha empeorado bastante, por lo que vamos a dejar el num_epochs a 30, dejar el dropout a 0.0 y vamos a probar
un lr_adam de 0.0005.

HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 30
LR_ADAM = 0.0005

RUN_ID = b28593c510ac423daf67476f4b94d81b
cv_mean_loss = 0.23453351720715693
cv_std_loss = 0.06005338085782518
cv_mean_val_f1_grupo = 0.23453351720715693

Sigue siendo peor. Vamos a probar por último el lr mayor a 0.001:
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 30
LR_ADAM = 0.0015

run_id = 42f413fdb34643118bae5b919e67d5c0
cv_mean_loss = 2.327610669950558
cv_std_loss = 0.23768257663965942
cv_mean_val_f1_grupo = 0.1898677999596313

Es bastante peor que la mejor configuración obtenida. Vamos a probar con valores de alpha leaky relu superior a 0.01.
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.025
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 30
LR_ADAM = 0.001

run_id = baa3efe49ffe43edb4e300fbd5a51b89
cv_mean_loss = 2.3529067314892447
cv_std_loss = 0.27191770667001186
cv_mean_val_f1_grupo = 0.2275475653387712


No mejoró, nos quedamos con esta configuración como la mejor:

Final Grupo:
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
"""

#Hiperparámetros. Dejamos el numero de epochs igual que antes y aumentamos dropout para intentar mejorar el val_loss.
#Aumentamos tambien el LR_adam a 0.0005. A la siguiente buscamos aumentar el hidden_size.

#Primero se prueba grupo, luego caja.

HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.075
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 25
LR_ADAM = 0.0005

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
    
