from tune_pipeline_comun import PipelineComunCRNN

pkl_train = "train_mels_chunked_aumentado.pkl"
pkl_test = "test_mels_chunked_aumentado.pkl"

"""
Hiperparametros tuneo caja: f1 = 0.5687804055911898
hidden_size 256
batch_size 32
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.01
is_bidirectional True
dropout 0.0
num_epochs 50
lr_adam 0.001

Post-Tuneo 1: f1 = 0.5572777502536105. Ha empeorado:
hidden_size 256
batch_size 32
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.01
is_bidirectional True
dropout 0.3
LR_ADAM = 0.0005
num_epochs 75

#Vamos a probar a bajar el numero de epochs a 50, mismo dropout, mismo hiddent_size, distinto LR_ADAM -> Subimos a 0.001

Post-tuneo 2: f1 = 0.5035639520893519. Ha empeorado bastante.
hidden_size 256
batch_size 32
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.01
is_bidirectional True
dropout 0.3
LR_ADAM = 0.001
num_epochs 50

Vamos a probar a bajar el LR_ADAM a 0.0005 y quitar el dropout.

Post-tuneo 3: f1 = 0.5327528138247111. Ha mejorado pero sigue siendo peor que el primer tuneo.
hidden_size 256
batch_size 32
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.01
is_bidirectional True
dropout 0.0
num_epochs 50
lr_adam 0.0005

Vamos a probar con un lr mayor a 0.001 como 0.005 y un leaky_relu a 0. Lo demás se mantiene igual.

Post-tuneo 4: f1 = 0.41499399531997583.
hidden_size 256
batch_size 32
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.0
is_bidirectional True
dropout 0.0
num_epochs 50
lr_adam 0.005

Vamos a probar a bajar lr a 0.001 y dejar el leaky_relu a 0.0. Es decir, la mejor configuración pero con leaky_relu a 0.0.

Post-tuneo 5: f1 = 0.5266239021881699. Sigue siendo peor que el f1  original.
hidden_size 256
batch_size 32
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.0
is_bidirectional True
dropout 0.0
num_epochs 50
lr_adam 0.001

Vamos a probar con un leaky_relu mayor.

Post-tuneo 6: f1 = 0.537408277504679. Sigue siendo peor que el tuneo inicial. 
hidden_size 256
batch_size 32
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.05
is_bidirectional True
dropout 0.0
num_epochs 50
lr_adam 0.001

Vamos a probar a dejar todo igual al tuneo inicial y solo aumentar el batch_size a 64 y el hidden_size a 512.

Post-tuneo 7: f1 = 0.5764008042358557. Es mejor que el mejor tuneo.
hidden_size 512
batch_size 64
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.01
is_bidirectional True
dropout 0.0
num_epochs 50
lr_adam 0.001

En este caso el val_loss ha aumentado mucho, pero el val_loss en si mide la confianza matemática del modelo. 
Penaliza duramente la duda. Si el modelo duda entre una opción u otra con un % parecido entre ambas opciones, el loss es mayor,
pero a lo mejor al dudar al final elige la opción correcta.

Ahora vamos a probar a bajar el hidden_size a 256 y dejar dicho batch_size

Post-tuneo 8: f1 = 0.4949537385251967. Es bastante peor.
hidden_size 256
batch_size 64
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.01
is_bidirectional True
dropout 0.0
num_epochs 50
lr_adam 0.001

Vamos entonces a dejar la configuración del post-tuneo 7 como la configuración final:
hidden_size 512
batch_size 64
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.01
is_bidirectional True
dropout 0.0
num_epochs 50
lr_adam 0.001
"""

#En este caso, se acerca bastante a los hiperparámetros de base, pero mejor. Vamos a probar a aumentar el LR a 0.0005 y aumentar
#el num_epochs. También aumentamos el dropout para pelear en contra del overfitting. Lo siguiente puede ser aumentar el hidden_size

#Hiperparámetros
BATCH_SIZE = 64
HIDDEN_SIZE = 512
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.01
IS_BIDIRECTIONAL = True 
DROPOUT = 0.0
LR_ADAM = 0.001
NUM_EPOCHS = 50 

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
    
