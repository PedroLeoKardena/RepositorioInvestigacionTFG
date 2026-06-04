from tune_pipeline_comun import PipelineComunCRNN

pkl_train = "train_mfcc_chunked.pkl"
pkl_test = "test_mfcc_chunked.pkl"

"""Hiperparametros tuneo caja: f1 = 0.5619249651086559
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
"""

#Hiperparámetros. Dejamos el numero de epochs igual que antes y aumentamos dropout para intentar mejorar el val_loss.
#Aumentamos tambien el LR_adam a 0.0005. A la siguiente buscamos aumentar el hidden_size.

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
    
