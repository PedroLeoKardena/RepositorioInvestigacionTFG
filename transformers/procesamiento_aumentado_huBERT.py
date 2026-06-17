import os
from tune_base_pipeline import BaseTransformerPipeline, HubertMultiTask

#Para este hay que hacer dos hipertuneos:

"""
Hiperparámetros para tuneo de caja: f1 = 0.5169680115710812
learning_rate 3e-05
batch_size 4
gradient_acc_steps 2
num_epochs 5
weight_decay 0.01
warmup_steps 100

Post-Tuneo 1: f1 = 0.5185006995938529 Ha mejorado un poco.
learning_rate 3e-05
batch_size 4
gradient_acc_steps 4
num_epochs 10
weight_decay 0.01
warmup_steps 100

Post-Tuneo 2: f1 = 0.5338814280992676. Ha mejorado aun más:
learning_rate 3e-05
batch_size 4
gradient_acc_steps 4
num_epochs 8
weight_decay 0.01
warmup_steps 250

Vamos a probar a subir el weight_decay a 0.05 y warmup_steps a 500.

Post-Tuneo 3: f1 = 0.5184666323791686. Empeoro.
learning_rate 3e-05
batch_size 4
gradient_acc_steps 4
num_epochs 8
weight_decay 0.05
warmup_steps 500

#Vamos a probar a volver a bajar los warmup_steps a 250 y dejar el weight_decay en 0.05.

Post-Tuneo 4: f1 caja = 0.5046497635678261. Ha empeorado.
learning_rate 3e-05
batch_size 4
gradient_acc_steps 4
num_epochs 8
weight_decay 0.05
warmup_steps 250

Sigue empeorando, por lo que podemos achacar el empeoramiento al valor escogido de weight_decay.
Vamos a probar a bajar el weight_decay a 0.01 y dejar el resto igual que el post-tuneo 3.

Post-Tuneo 5: 0.4942151698283942. Sigue empeorando.
learning_rate 3e-05
batch_size 4
gradient_acc_steps 4
num_epochs 8
weight_decay 0.01
warmup_steps 500


Creemos que hemos llegado al mejor resultado con el post-tuneo 2. Tuneo Final con f1 = 0.5338814280992676
learning_rate 3e-05
batch_size 4
gradient_acc_steps 4
num_epochs 8
weight_decay 0.01
warmup_steps 250
"""

#Vamos a probar a aumentar el numero de epochs de 5 a 10. Probaremos con un mismo peso de 0.01 y aumentaremos el numero de grad_steps
#Dejamos warmup_steps en 100. Lo siguiente: probar con mayor warmup_steps y ver como evoluciona.

#Ahora vamos a probar con 8 epochs, mismo LR y Weight_decay, pero mas warmup_steps.

LR = 5e-5
BATCH_SIZE = 4
GRAD_STEPS = 8
EPOCHS = 8
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 250


class HubertAugmentedPipeline(BaseTransformerPipeline):
    @property
    def max_audio_length(self): return 160000
    @property
    def nombre_dataset(self): return "Aumentado"
    @property
    def nombre_modelo(self): return "facebook/hubert-base-ls960"
    @property
    def ruta_audios(self): return "audios_aumentados"
    @property
    def csv_train(self): return "metadata_train_aumentado.csv"
    @property
    def csv_test(self): return "metadata_test_aumentado.csv"
    @property
    def nombre_run(self): return f"HuBERT_Chunk_Augmented_{EPOCHS}epochs"
    @property
    def nombre_modelo_guardado(self): return "modelo_multitask_augmented_hubert"

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
    pipeline = HubertAugmentedPipeline()
    pipeline.ejecutar()
