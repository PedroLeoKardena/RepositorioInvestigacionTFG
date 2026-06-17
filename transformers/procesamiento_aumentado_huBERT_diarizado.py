import os
from tune_base_pipeline import BaseTransformerPipeline, HubertMultiTask

"""
Hiperparámtros grupo:
test_f1_grupo 0.752896174863388
test_recall_grupo 0.83
test_precision_grupo 0.6889

learning_rate 5e-05
batch_size 4
gradient_acc_steps 4
num_epochs 8
weight_decay 0.01
warmup_steps 100

Post-Tuneo: f1 = 0.5999156536839494. Ha empeorado.

learning_rate 3e-05
batch_size 4
gradient_acc_steps 4
num_epochs 10
weight_decay 0.01
warmup_steps 100

#Ahora vamos a probar con 8 epochs como antes, LR de 3e-05 y mas warmup_steps.

Post-Tuneo 2: f1 = 0.748097972972973. Mejora con respecto al anterior pero es peor con respecto al anterior:
learning_rate 3e-05
batch_size 4
gradient_acc_steps 4
num_epochs 8
weight_decay 0.01
warmup_steps 250

Vamos a probar con el LR de 5e-05 y el resto lo mismo que el post-tuneo 2.

Post-Tuneo 3: f1 = 0.7122128378378378. Ha empeorado.
learning_rate 5e-05
batch_size 4
gradient_acc_steps 4
num_epochs 8
weight_decay 0.01
warmup_steps 250

Vamos a probar a bajar el LR a 1e-05

Post-Tuneo 4: f1 = 0.607313994090787. Ha empeorado bastante tras probar un LR de 1e-05.
learning_rate 1e-05
batch_size 4
gradient_acc_steps 4
num_epochs 8
weight_decay 0.01
warmup_steps 250

Tenemos que dejar el LR a 3e-05. Vamos a probar a aumentar el numero de warmup_steps a 500.

Post-Tuneo 5: f1 = 0.7031138628813047. Sigue siendo peor que el post-tuneo 2.
learning_rate 3e-05
batch_size 4
gradient_acc_steps 4
num_epochs 8
weight_decay 0.01
warmup_steps 500

Creemos que el mejor resultado es el tuneo original: f1 = 0.752896174863388
learning_rate 5e-05
batch_size 4
gradient_acc_steps 4
num_epochs 8
weight_decay 0.01
warmup_steps 100

Vamos a dejar dichos hiperparámetros.
"""

#Vamos a probar a bajar el valor de learning_rate a 3e-05, aumentamos valor de epochs. Dejamos mismo numero de wamup_steps.
#Siguiente puede ser probar a aumentar weitght_decay a 0.05 o warmup_steps a 250.

LR = 5e-5
BATCH_SIZE = 4
GRAD_STEPS = 8
EPOCHS = 8
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 250

class HubertAugmentedDiarizadoPipeline(BaseTransformerPipeline):
    @property
    def max_audio_length(self): return 160000
    @property
    def nombre_dataset(self): return "Aumentado_Diarizado"
    @property
    def nombre_modelo(self): return "facebook/hubert-base-ls960"
    @property
    def ruta_audios(self): return "audios_aumentados_diarizados"
    @property
    def csv_train(self): return "metadata_train_aumentado_diarizado.csv"
    @property
    def csv_test(self): return "metadata_test_aumentado_diarizado.csv"
    @property
    def nombre_run(self): return f"HuBERT_Diarizado_Chunk_Augmented_{EPOCHS}epochs"
    @property
    def nombre_modelo_guardado(self): return "modelo_multitask_augmented_hubert_diarizado"

    @property
    def learning_rate(self): return LR
    @property
    def batch_size(self): return BATCH_SIZE
    @property
    def grad_steps(self): return GRAD_STEPS
    @property
    def epochs(self): return EPOCHS
    @property
    def weight_decay(self): return WEIGHT_DECAY
    @property
    def warmup_steps(self): return WARMUP_STEPS

    def get_multitask_model(self, num_labels_grupo, num_labels_caja):
        return HubertMultiTask(self.nombre_modelo, num_labels_grupo, num_labels_caja)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pipeline = HubertAugmentedDiarizadoPipeline()
    pipeline.ejecutar()
