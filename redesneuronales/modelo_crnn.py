import torch.nn as nn


class CRNN(nn.Module):
    #En esta clase definimos el numero de características MFCC que tenemos (numero de filas de nuestra matriz MFCC), 
    #el numero de time steps (representado por las muestras temporales, numero de columnas de la matriz MFCC),
    #tambien establecemos el batch_size (contando tanto train como test). No establecemos el canal 
    #porque lo vamos a establecer en el forward, ya que el canal va a ser 1 (porque es una matriz MFCC).
    def __init__(self, num_features, num_time_steps, batch_size, num_capas_convolucionales, num_filtros_convolucionales, kernel_size_convolucional, stride_filtro, num_capas_recurrentes, num_neuronas_recurrentes):
        super(CRNN, self).__init__()
        self.num_features = num_features
        self.num_time_steps = num_time_steps
        self.batch_size = batch_size
        self.num_capas_convolucionales = num_capas_convolucionales
        self.num_filtros_convolucionales = num_filtros_convolucionales
        self.kernel_size_convolucional = kernel_size_convolucional
        self.stride_filtro = stride_filtro
        self.num_capas_recurrentes = num_capas_recurrentes
        self.num_neuronas_recurrentes = num_neuronas_recurrentes

        #Hay que tener en cuenta: el kernel es el tamaño de escaneo del filtro (que será igual al tamaño del filtro).
        #Stride es la cantidad de pasos que da el filtro en cada movimiento de la moving window.
        #En nuestro caso, no meteremos padding.
        # 
        # El tamaño de salida se calcula como floor((tamaño_entrada - tamaño_filtro)/stride) + 1
        # padding = numero de ceros añadidos. 

        self.cnn = self._estrutura_convolucional()


    def forward(self, x):
        pass
        


        


