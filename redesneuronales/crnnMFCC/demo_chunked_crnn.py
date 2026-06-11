from tune_pipeline_comun import PipelineComunCRNN

pkl_train = "train_mfcc_chunked.pkl"
pkl_test = "test_mfcc_chunked.pkl"

"""
Hiperparametros tuneo caja: f1 = 0.5619249651086559
hidde_size 256 
batch_size 32
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.01
is_bidirectional True
dropout 0.0
num_epochs 50

Post-Tuneo 1 caja: f1 = 0.5282869719183863
hidde_size 256 
batch_size 32
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.01
is_bidirectional True
lr_adam = 0.0005
dropout 0.3
num_epochs 50

Vamos a probar a subir lr_adam a 0.001 y dejar lo demás igual.
Post-tuneo 2: f1 = 0.5145159402284933.
hidde_size 256 
batch_size 32
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.01
is_bidirectional True
lr_adam = 0.001
dropout 0.3
num_epochs 50


Vamos a probar la configuración del mejor tuneo con LR_adam a 0.0005.

Post-tuneo 3: f1 = 0.5093604015481491. Ha empeorado otra vez.
hidde_size 256 
batch_size 32
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.01
is_bidirectional True
lr_adam = 0.0005
dropout 0.0
num_epochs 50

Vamos a utilizar exactamente la mejor configuación pero con leaky_rely a 0
Vamos a probar a subir el lr_adam a 0.005.

Post-tuneo 4: f1 = 0.516719416896888. Ha empeorado bastante.
hidden_size 256 
batch_size 32
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.0
is_bidirectional True
lr_adam = 0.005
dropout 0.0
num_epochs 50

Entonces vamos a probar a bajar el lr_adam a 0.0005, si no lo dejamos a 0.001 y probamos despues otras configuraciones de leaky_relu.

Post-tuneo 5: f1 caja = 0.5334132218638231. Sigue siendo peor que el tuneo 5. 
hidden_size 256 
batch_size 32
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.0
is_bidirectional True
lr_adam = 0.0005
dropout 0.0
num_epochs 50

El mejor lr_adam es 0.001. Vamos a probar a dejar dicho lr_adam y subir solo el dropout con el leaky_relu a 0.0. También compararemos el val_loss con el del post-tuneo 4.

Post-tuneo 6: f1 caja = 0.5168860677507237:
hidden_size 256 
batch_size 32
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.0
is_bidirectional True
lr_adam = 0.001
dropout 0.3
num_epochs 50

Vemos que aumentar el dropout no mejora el f1. Vamos a probar con leaky_relu a 0 y dropout a 0, lr_adam a 0.001.

Post-tuneo 7: f1 caja = 0.5363037547362733. Sigue siendo peor que el inicial
hidden_size 256 
batch_size 32
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.0
is_bidirectional True
lr_adam = 0.001
dropout 0.0
num_epochs 50

Vamos a probar con 512 de hidden_size y batch_size de 64 con la configuracion inicial.

Post-tuneo 8: f1 caja = 0.557267357416102. Sigue siendo levemente peor que la inicial.
hidden_size 512 
batch_size 64
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.0
is_bidirectional True
lr_adam = 0.001
dropout 0.0
num_epochs 50

Vamos a quedarnos entonces con el tuneo inicial:
hidde_size 256 
batch_size 32
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.01
is_bidirectional True
dropout 0.0
num_epochs 50
lr_adam 0.001
"""

#Hiperparámetros. Dejamos el numero de epochs igual que antes y aumentamos dropout para intentar mejorar el val_loss.
#Aumentamos tambien el LR_adam a 0.0005. A la siguiente buscamos aumentar el hidden_size.

HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 1
ALPHA_LEAKY_RELU = 0.01
IS_BIDIRECTIONAL = False
DROPOUT = 0.0
NUM_EPOCHS = 20
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
    
