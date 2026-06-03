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
"""

#En este caso, se acerca bastante a los hiperparámetros de base, pero mejor. Vamos a probar a aumentar el LR a 0.0005 y aumentar
#el num_epochs. También aumentamos el dropout para pelear en contra del overfitting. Lo siguiente puede ser aumentar el hidden_size

#Hiperparámetros
BATCH_SIZE = 32
HIDDEN_SIZE = 256 
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.01
IS_BIDIRECTIONAL = True 
DROPOUT = 0.3
LR_ADAM = 0.0005 
NUM_EPOCHS = 75 

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
    
