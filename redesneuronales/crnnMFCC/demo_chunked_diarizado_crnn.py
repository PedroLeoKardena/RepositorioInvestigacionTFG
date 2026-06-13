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
Hiperparametros tuneo grupo: f1 = 0.835972573078311
hidden_size 128
batch_size 16
num_capas_ocultas_lstm 1
alpha_leaky_relu 0.01
is_bidirectional False
dropout 0.0
num_epochs 20

Post-Tuneo: f1 grupo = 0.7846164874551971

hidden_size 128
batch_size 32
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.01
is_bidirectional True
dropout 0.3
lr_adam 0.0005
num_epochs 50

#Vamos a subir el numero de capas ocultas a 256, vamos a probar LR_adam a 0.001 y lo demas igual

Post-tuneo 2: f1 grupo = 0.7546929824561404. Ha empeorado otra vez.
hidden_size 256
batch_size 32
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.01
is_bidirectional True
dropout 0.3
lr_adam 0.001
num_epochs 50

Vamos a probar la configuración del mejor tuneo pero con epochs 50 y hidden_size 256, y con LR_adam a 0.0005.

Post-tuneo 3: f1 grupo = 0.7866329284750337. Sigue sin ser mejor que el primero. 
hidden_size 256
batch_size 32
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.01
is_bidirectional True
dropout 0.0
lr_adam 0.0005
num_epochs 50

Vamos a probar misma configuración exacta, pero con lr_adam 0.005 y batch_size 32.

Post-tuneo 4: f1 grupo = 0.8801541425818882. Ha mejorado bastante. 
hidden_size 128
batch_size 32
num_capas_ocultas_lstm 1
alpha_leaky_relu 0.01
is_bidirectional False
dropout 0.0
lr_adam 0.001
num_epochs 20

Vamos a probar a bajar el lr_adam de 0.001 a 0.0005.

Post-tuneo 5: f1 grupo = 0.8144848484848485. De este modo podemos llegar a la conclusión que el mejor lr_adam es 0.001.
hidden_size 128
batch_size 32
num_capas_ocultas_lstm 1
alpha_leaky_relu 0.01
is_bidirectional False
dropout 0.0
lr_adam 0.001
num_epochs 20

Ahora vamos a probar a subir el dropout para intentar pelear contra el sobreajuste. Para este caso no solo analizaremos el f1, sino tambien el val_loss.
Como queremos probar dropout distinto a 0.0, necesesitamos aumentar el numero de capas de la LSTM, ya que el dropout solo se aplica a la ultima capa.

Post-tuneo 6: f1 grupo = 0.8020496894409938. Vamos que va a empeorado.
hidden_size 128
batch_size 32
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.01
is_bidirectional False
dropout 0.3
lr_adam 0.001
num_epochs 20

Los resultados de val_loss tambien han empeorado, por lo que aumentar el dropout no ha resultado del todo beneficioso.
De este modo, vamos a deajar el dropout a 0.0, el numero de capas a 1. Vamos a probar cambiando el valor de leaky_relu a 0.0.

Post-tuneo 7: f1 grupo = 0.765765629719118
hidden_size 128
batch_size 32
num_capas_ocultas_lstm 1
alpha_leaky_relu 0.0
is_bidirectional False
dropout 0.0
lr_adam 0.001
num_epochs 20

#Vamos a probar aumentando el hidden_size a 512 y el batch_size a 64, 2 capas de lstm y bidireccional. Si no mejora nos quedamos con post-tuneo 4.

Post-tuneo 8: f1 grupo = 0.8078260869565217
hidden_size 512
batch_size 64
num_capas_ocultas_lstm 2
alpha_leaky_relu 0.0
is_bidirectional True
dropout 0.0
lr_adam 0.001
num_epochs 20

Nos quedamos con la mejor configuración:
hidden_size 128
batch_size 32
num_capas_ocultas_lstm 1
alpha_leaky_relu 0.01
is_bidirectional False
dropout 0.0
lr_adam 0.001
num_epochs 20

"""
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.1
NUM_EPOCHS = 20
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
    
