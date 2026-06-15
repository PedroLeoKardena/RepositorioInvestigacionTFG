# Repositorio Utilizado Para Almacenar Código del Trabajo Fin de Grado

## Proceso Ejecución Archivos:

Realmente lo importante en este proyecto es el orden de ejecución de los distintos archivos de generación de datos que se encuentran en la carpeta. Estos son los archivos que nos permitiran generar los conjuntos de datos con los que vamos a trabajar en los demás archivos. 

### Orden Ejecución Scripts Generación Datos:

#### Paso 1 — Generar enviroments en bae a archivos .yml 

**Scripts:** `environment.yml` o `environment_mac.yml` (si se tiene mac) o `environment_avanzado.yml` (si se tiene ordenador con GPU NVIDIA generación superior a las 4000 o 3000) y `environment_diarizacion.yml`

Yo utilizo anaconda, asi que uso este comando:  
conda env create --name envname --file=environments.yml

#### Paso 2 - Dividir Dataset

**Script:** `codigoGeneracionDatos/dividir_dataset.py`

Este script permite simplemente dividir el dataset para generar los folds y los conjuntos train y test.

#### Paso 3 — Generar auidos diarizados 

**Scripts:** `codigoGeneracionDatos/crear_audios_diarizados/diarizacion_reporte.py` y `codigoGeneracionDatos/crear_audios_diarizados/diarizacion_extraccion.py`

Orden acciones:

1. Ejecutar `diarizacion_reporte.py` con entorno `env_diarizacion`
2. Esto genera un archivo .csv que es necesario rellenar para poder ejecutar el siguiente script. Se rellena tal y como se informa en la documentación. Si se dispone de este csv no es necesario hacer estos pasos y saltar directamente al siguiente.
3. Ejecutar en `diarización_extraccion.py` con entorno `env_audio` o `env_diarizacion`. Esto trabaja con el csv previo y genera los audios diarizados.

Todo lo que se explica a continuación se hace con el environment `env_audio` activo.

#### Paso 4 - Generar datasets en base de audios.

**Scripts:** `codigoGeneracionDatos/generar_datasets/generar_dataset_chunkeado.py`, `codigoGeneracionDatos/generar_datasets/generar_dataset_chunkeado_aumentado.py`, `codigoGeneracionDatos/generar_datasets/generar_dataset_chunkeado_diarizado.py` y `codigoGeneracionDatos/generar_datasets/generar_dataset_chunkeado_aumentado_diarizado.py`


Aqui el orden sí que importa y será el siguiente:

1. generar_dataset_chunkeado.
2. generar_dataset_chunkeado_aumentado
3. generar_dataset_chunkeado_diarizado
4. generar_dataset_chunkeado_diarizado_aumentado


Con esto ya tendremos todos los conjuntos de datos con los que trabajar.


#### Paso 5 - Generar MFCC y EspectrogramasMel.

El orden no importa una vez que tengamos generados los datasets. Estos conjuntos de datos son necesarios para los modelos ML clásico y los CRNN. Los transformers trabajan con los datos sin transformar, es decir, con los audios chunkeados, audios diarizados, aumentados,...


#### Paso 6 - Ejecutar Entrenamiento de modelos.

Esto tampoco sigue orden, simplemente se trata de ejecutar los modelos.
