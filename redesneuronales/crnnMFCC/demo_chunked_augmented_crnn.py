from tune_pipeline_comun import PipelineComunCRNN

pkl_train = "train_mfcc_chunked_aumentado.pkl"
pkl_test = "test_mfcc_chunked_aumentado.pkl"

"""
Después de los tuneos básicos, el mejor tuneo es el siguiente:
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 1
ALPHA_LEAKY_RELU = 0.01
IS_BIDIRECTIONAL = False
DROPOUT = 0.0
NUM_EPOCHS = 20
LR_ADAM = 0.001

En mlflow = CRNN_MelSpectrogram_chunked_aumentado_hidden128. RunID = 3c3fb3ceabc045c7b948adae3b9b9552

Buscar run = attributes.run_id IN ("3c3fb3ceabc045c7b948adae3b9b9552")

Aun que no presenta el absoluto mejor f1_macro para grupo, si que presenta un mucho menor val_loss que los demás
que presentan un f1_macro equiparable. Si presenta el mejor de caja.
cv_mean_loss = 2.423198068672027
cv_mean_val_f1_grupo = 0.2820297311359166
cv_mean_val_f1_caja = 0.20612217438134217

Siguiente tuneo a probar: Ponemos 2 layers, probamos un leaky rely de 0.0 y un n_epochs de 50
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 30
LR_ADAM = 0.001

run_id = 8b1393788b664d968f5e261564d0c33f
cv_mean_loss = 2.5356093311396712
cv_mean_val_f1_grupo = 0.2259332379093078
cv_mean_val_f1_caja = 0.21940754949496344

De caja es mejor, pero de grupo es bastante peor.


MEJOR CONFIG:
Vamos a probar con el siguiente tuneo. Bajando el num_epochs y poniendo un dropout del 15%
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.15
NUM_EPOCHS = 20
LR_ADAM = 0.001

Run id = 02fc91bcfb894e4fa5e3a8c9539c501c
cv_mean_loss = 2.5077265987083637
cv_std_loss = 0.31695911614604066
cv_mean_val_f1_grupo = 0.2643407233619127
cv_mean_val_f1_caja = 0.20549798475539577

Ha mejorado. Vamos a probar a aumentar el dropout.
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.3
NUM_EPOCHS = 20
LR_ADAM = 0.001

Run id = 1c01217b3b6f43d7aad6e40730cbe904
cv_mean_loss = 2.4418591386872253
cv_std_loss = 0.3023706060190215
cv_mean_val_f1_grupo = 0.2678951338842516
cv_mean_val_f1_caja = 0.1846628149625367

Aumentando minimamente el dropout lo unico que hemos conseguido ha sido tener un mejor loss, una pequeñisima mejora en grupo 
y un peor f1 de caja.


Vamos a probar a dejar dropout de 20% y bajar el lr_adam:
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.2
NUM_EPOCHS = 20
LR_ADAM = 0.0005

Run id = f2a8bfc3c2024eefadb53491b4d3691a
cv_mean_loss = 2.4670401398414903
cv_std_loss = 0.39946097555113996
cv_mean_val_f1_grupo = 0.27993803893850383
cv_mean_val_f1_caja = 0.17735746775232505


Ha mejorado el grupo, pero la caja ha empeorado.
Podemos entonces dividir el tuneo para cajas y grupos a partir de aqui:

-------------------------CAJA------------------------
MEJOR CAJA:
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 30
LR_ADAM = 0.001

run_id = 8b1393788b664d968f5e261564d0c33f
cv_mean_val_f1_caja = 0.21940754949496344

Vamos a probar a subir el lr_adam con esta config.
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 30
LR_ADAM = 0.003

Run id = b1546baa26f64c7386adad009c0d618c
cv_mean_loss = 2.5356093311396712
cv_std_loss = 0.27384279292478264
cv_mean_val_f1_caja = 0.21940754949496344

Peor que de momento la mejor config.
Vamos a dejar dicho lr_adam y vamos simpemente a probar con un alpha mayor.

HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.02
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 30
LR_ADAM = 0.001

Run id = a903308f0a214be38cbc25bdd7f989f8
cv_mean_loss = 2.5481665918963805
cv_std_loss = 0.16721483380436244
cv_mean_val_f1_caja = 0.19254161988414892

Vemos que no mejora. Entonces dejamos esto como la configuración final de caja:

Final Caja:
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 30
LR_ADAM = 0.001

run_id = 8b1393788b664d968f5e261564d0c33f
cv_mean_val_f1_caja = 0.21940754949496344
-------------------------GRUPO------------------------
MEJOR GRUPO:
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.2
NUM_EPOCHS = 20
LR_ADAM = 0.0005

Run id = f2a8bfc3c2024eefadb53491b4d3691a
cv_mean_loss = 2.4670401398414903
cv_std_loss = 0.39946097555113996
cv_mean_val_f1_grupo = 0.27993803893850383

Vamos a probar a subir el lr_adam a 0.0015
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.2
NUM_EPOCHS = 20
LR_ADAM = 0.0015


RUN id = fca61a34939543359f43907b1f5aebbe
cv_mean_loss = 2.510968184921188
cv_std_loss = 0.21968745173992257
cv_mean_val_f1_grupo = 0.25379382733055966

Ha empeorado, por lo que el mejor lr_adam es el de 0.0005. Vamos a probar ahora con un mayor alpha de leaky relu.
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.02
IS_BIDIRECTIONAL = True
DROPOUT = 0.2
NUM_EPOCHS = 20
LR_ADAM = 0.0005

run_id = a2a6df11e7de44a5a681624de1c80738
cv_mean_loss = 2.5737378586955892
cv_std_loss = 0.23503044440397108
cv_mean_val_f1_grupo = 0.2780187080547315

Es levemente peor. De este modo, nos quedamos con configuración final:

FINAL GRUPO:
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.0
IS_BIDIRECTIONAL = True
DROPOUT = 0.2
NUM_EPOCHS = 20
LR_ADAM = 0.0005

Run id = f2a8bfc3c2024eefadb53491b4d3691a
cv_mean_loss = 2.4670401398414903
cv_std_loss = 0.39946097555113996
cv_mean_val_f1_grupo = 0.27993803893850383
"""

#Hiperparámetros
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_LAYERS_LSTM = 2
ALPHA_LEAKY_RELU = 0.02
IS_BIDIRECTIONAL = True
DROPOUT = 0.0
NUM_EPOCHS = 30
LR_ADAM = 0.001



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
    
