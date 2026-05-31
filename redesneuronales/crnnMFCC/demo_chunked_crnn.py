from tune_pipeline_comun import PipelineComunCRNN

pkl_train = "train_mfcc_chunked.pkl"
pkl_test = "test_mfcc_chunked.pkl"
#Hiperparámetros
BATCH_SIZE = 32
HIDDEN_SIZE = 256
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.01
IS_BIDIRECTIONAL = True 
DROPOUT = 0.0 
LR_ADAM = 0.0001 
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
    
