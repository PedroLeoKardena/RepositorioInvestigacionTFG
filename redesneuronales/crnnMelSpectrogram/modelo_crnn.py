import torch
from torch import nn

#No cambiamos nada del modelo CRNN desarrollado con respecto a MFCC. 
#Lo que queremos es hacer una comparativa entre MFCC y Mel-Spectrogram.

#Es por esto que vamos a mantener los strides, el kernel, etc... Queremos que se trate de la misma red neuronal.
#Sabemos que como Mel-Spectrogram presenta 128 características, tendremos un input_size para el LSTM final mucho mayor, lo que podría llevar a un problema computacional
#y posible overfitting, sin embargo, queremos comparar rendimiento de ambas redes.
# 
#Luego, podremos crear una nueva red neuronal CRNN basada en la misma idea con un stride distinto que se ajuste al nuevo modelo. 



class CRNN(nn.Module):
   
    def __init__(self, num_features, num_time_steps, hidden_size, num_capas_ocultas_lstm, alpha_leaky_relu, is_bidirectional, dropout):
        super(CRNN, self).__init__()
        assert num_time_steps == 1000, f"El numero de muestras temporales inicial debe ser 1000, pero se obtuvo {num_time_steps}"

        self.num_features = num_features
        self.num_time_steps = num_time_steps
        self.hidden_size = hidden_size
        self.num_capas_ocultas_lstm = num_capas_ocultas_lstm
        self.ALPHA_LEAKY_RELU = alpha_leaky_relu
        self.is_bidirectional = is_bidirectional
        self.dropout = dropout
        canales = [1, 32, 64]

        self.cnn1, (num_pasos_temporales_reducido1) = self.construir_cnn_block(in_channels=canales[0], num_caracteristicas=num_features, num_pasos_temporales=num_time_steps, out_channels = canales[1], alpha_leaky_relu = self.ALPHA_LEAKY_RELU)
        self.cnn2, _ = self.construir_cnn_block(in_channels=canales[1], num_caracteristicas=num_features, num_pasos_temporales=num_pasos_temporales_reducido1, out_channels = canales[2], alpha_leaky_relu = self.ALPHA_LEAKY_RELU)
        self.rnn = nn.LSTM(input_size=num_features*canales[2], hidden_size=self.hidden_size, num_layers=self.num_capas_ocultas_lstm, bidirectional=self.is_bidirectional, batch_first=True, dropout=self.dropout)

        self.salida_grupo = nn.Linear(in_features=self.hidden_size*(2 if self.is_bidirectional else 1), out_features=5)
        self.salida_clase = nn.Linear(in_features=self.hidden_size*(2 if self.is_bidirectional else 1), out_features=5) 


    def construir_cnn_block(self, in_channels, num_caracteristicas, num_pasos_temporales, out_channels, alpha_leaky_relu):
        assert num_pasos_temporales == 1000 or num_pasos_temporales % 250 == 0
        assert num_caracteristicas == 128
        stride = (1,2) #Con este stride, el filtro se mueve 1 paso en la dirección de las características (filas) y 2 pasos en la dirección de los pasos temporales (columnas).

        bloque_cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(alpha_leaky_relu)
        )
       
        num_pasos_temporales_reducido = (num_pasos_temporales - 1)//stride[1] + 1
        return bloque_cnn, (num_pasos_temporales_reducido)



    def forward(self, input):
        x = self.cnn1(input)
        x = self.cnn2(x)

        x = x.permute(0, 3, 1, 2)

        
        batch_size = x.size(0)
        tiempo = x.size(1)

        x = x.reshape(batch_size, tiempo, -1)

        resumen_temporal, _ = self.rnn(x)

        prediccion_grupo = self.salida_grupo(resumen_temporal)
        prediccion_clase = self.salida_clase(resumen_temporal)

        return prediccion_grupo, prediccion_clase
        


        


