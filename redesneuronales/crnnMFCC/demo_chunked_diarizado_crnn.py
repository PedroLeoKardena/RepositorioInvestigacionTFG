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
"""

#En este caso, lo que sucede es que tambien hay subida del val_loss. Vamos a dejar el numero de hidden_size, aumentar el batch_size a 32.
#Luego aumentamos el dropout a 0.3. Aumentamos el num_epochs de 20 a 50. LR_adam lo pasamos de 0.001 a 0.0005
BATCH_SIZE = 32
HIDDEN_SIZE = 256
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.01
IS_BIDIRECTIONAL = True 
DROPOUT = 0.3
LR_ADAM = 0.001
NUM_EPOCHS = 50

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
    
