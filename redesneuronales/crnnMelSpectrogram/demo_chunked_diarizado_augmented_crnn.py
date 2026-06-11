from tune_pipeline_comun import PipelineComunCRNN

pkl_train = "train_mels_chunked_aumentado_diarizado.pkl"
pkl_test = "test_mels_chunked_aumentado_diarizado.pkl"

#Hiperparámetros
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 1
ALPHA_LEAKY_RELU = 0.01
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 20 
LR_ADAM = 0.001

if __name__ == "__main__":
    pipeline = PipelineComunCRNN(
        nombre_dataset="chunked_diarizado_aumentado",
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
    
