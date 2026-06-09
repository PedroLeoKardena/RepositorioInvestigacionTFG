import torch
from torch import nn

#De cara a la teoría, tener varias una red de varias capas convolucionales es lo mismo que tener varias sub-redes convolucionales conectadas en cadena.
#Según las arquitecturas modernas ResNet o VGG, lo que se suele hacer es escribir bloques repetibles, donde cada bloque es una pequeña subred.


class CRNN(nn.Module):
    #En esta clase definimos el numero de características MFCC que tenemos (numero de filas de nuestra matriz MFCC), 
    #el numero de time steps (representado por las muestras temporales, numero de columnas de la matriz MFCC).
    #No establecemos el canal porque lo vamos a establecer en el forward, ya que el canal va a ser 1 (porque es una matriz MFCC).
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

        #Hay que tener en cuenta: el kernel es el tamaño de escaneo del filtro (que será igual al tamaño del filtro).
        #Stride es la cantidad de pasos que da el filtro en cada movimiento de la moving window.
        # 
        # El tamaño de salida se calcula como floor((tamaño_entrada - tamaño_filtro)/stride) + 1
        # padding = numero de ceros añadidos. 

        self.cnn1, (num_pasos_temporales_reducido1) = self.construir_cnn_block(in_channels=canales[0], num_caracteristicas=num_features, num_pasos_temporales=num_time_steps, out_channels = canales[1], alpha_leaky_relu = self.ALPHA_LEAKY_RELU)
        self.cnn2, _ = self.construir_cnn_block(in_channels=canales[1], num_caracteristicas=num_features, num_pasos_temporales=num_pasos_temporales_reducido1, out_channels = canales[2], alpha_leaky_relu = self.ALPHA_LEAKY_RELU)
        self.rnn = nn.LSTM(input_size=num_features*canales[2], hidden_size=self.hidden_size, num_layers=self.num_capas_ocultas_lstm, bidirectional=self.is_bidirectional, batch_first=True, dropout=self.dropout)
        #Apuntes sobre Bidireccional LSTM: https://medium.com/@anishnama20/understanding-bidirectional-lstm-for-sequential-data-processing-b83d6283befc

        self.salida_grupo = nn.Linear(in_features=self.hidden_size*(2 if self.is_bidirectional else 1), out_features=5)
        self.salida_clase = nn.Linear(in_features=self.hidden_size*(2 if self.is_bidirectional else 1), out_features=5) 


    def construir_cnn_block(self, in_channels, num_caracteristicas, num_pasos_temporales, out_channels, alpha_leaky_relu):
        assert num_pasos_temporales % 250 == 0
        assert num_caracteristicas == 30
        stride = (1,2) #Con este stride, el filtro se mueve 1 paso en la dirección de las características (filas) y 2 pasos en la dirección de los pasos temporales (columnas).

        bloque_cnn = nn.Sequential(
            #Aplicamos capa convolucional única, pasando de in_channels cuadriculas en la layer de entrada a out_channels cuadriculas en la layer de salida.
            #VER: https://docs.pytorch.org/docs/2.12/generated/torch.nn.Conv2d.html
            #Si miramos el ejemplo de 100 page ML book sobre CNN, observaremos una matriz de entrada de 4x4 -> canal_entrada = 1, altura = 4 y anchura = 4.
            #Esta imagen tiene un filtro que es una matriz 2x2 que se va desplazando bloque a bloque -> kernel_size = 2x2, stride = 1x1, padding = 0.
            #Tras ello, se genera solo una matriz de 3x3 -> canal_salida = 1, altura = 3 y anchura = 3.
            #En nuestro casos, estaremos pasando de una matriz de entrada de 30x1000 -> canal_entrada = 1, altura = 30 y anchura = 1000.
            #A tener 32 matrices de salida, cada una de 30x500 -> canal_salida = 32, altura = 30 y anchura = 500.

            #Porque generamos 32 matrices -> el numero de canales de salida es el numero de filtros que aplicamos a la matriz de entrada. Cada filtro genera una matriz de salida.
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            
            #Normalizamos los 32 canales (imagenes generadas), donde cada canal almacena información de los valores de activación de cada filtro, tras aplicar pesos y bias.
            nn.BatchNorm2d(out_channels),

            #Aplicamos ReLU que es el estándar. Los filtros suelen devuelven numeros negativos cuando se trata de ruido o información irrelevante.
            #Sin embargo, para no perder del todo información, hacemos uso de la función leaky ReLU, que permite establecer un valor alpha de multiplicación, que permite
            #aproximar los valores negativos al 0, sin establecerlos a 0.

            #Este alpha se puede configurar para probar funciones de activación. Si alpha = 0, se comporta como un ReLU normal.
            nn.LeakyReLU(alpha_leaky_relu)
        )
        #Nosotros no hacemos maxpooling. En el estudio: https://arxiv.org/pdf/1412.6806.pdf. Se establece que el maxpooling no es necesario, y que se puede reducir el numero de pasos temporales (columnas) a la mitad con un kernel de tamaño 2x2 y un stride de 2x2. De este modo, se reduce el numero de pasos temporales directamente en la capa convolucional.
        #'We find that max-pooling can simply be replaced by a convolutional layer with increased stride without loss in accuracy on several image recognition benchmarks'
        #En este estudio prueban la eficiencia de CNNs sin pooling layers y con strides de 2.
        #Conclusión extraida de: https://stackoverflow.com/questions/44666390/max-pool-layer-vs-convolution-with-stride-performance

        num_pasos_temporales_reducido = (num_pasos_temporales - 1)//stride[1] + 1
        return bloque_cnn, (num_pasos_temporales_reducido)



    #En la función forward se define el flujo de ejecución de redes, permitiendonos hacer transformaciones a los datos de por medio.
    def forward(self, input):
        x = self.cnn1(input)
        x = self.cnn2(x)

        #Hay que tener en cuenta que nuestro CNN devuelve una salida con forma (batch_size, canales, num_caracteristicas, num_pasos_temporales_reducidos):
        #Cambiamos para los ejes sean 0=Batch, 1=Canales, 2=Frec, 3=Tiempo a 0=Batch, 1=Tiempo, 2=Canales, 3=Frec, para que el LSTM pueda procesar la secuencia temporal correctamente.
        x = x.permute(0, 3, 1, 2)

        #Aun seguimos teniendo una salida de 4D, ahora mismo con la forma (batch_size, num_pasos_temporales_reducidos, canales, num_caracteristicas). 
        #El LSTM necesita una salida de 3D con la forma (   batch_size, num_pasos_temporales_reducidos, caracteristicas). Deberemos obtener los dos primeros valores:
        batch_size = x.size(0)
        tiempo = x.size(1)

        #Al pasar -1 indicamos que calcule automáticamente el numero de características, que será canales*num_caracteristicas.
        x = x.reshape(batch_size, tiempo, -1)

        resumen_temporal, _ = self.rnn(x)
        contexto_global = resumen_temporal.mean(dim=1) 

        prediccion_grupo = self.salida_grupo(contexto_global)
        prediccion_clase = self.salida_clase(contexto_global)

        return prediccion_grupo, prediccion_clase
        


        


