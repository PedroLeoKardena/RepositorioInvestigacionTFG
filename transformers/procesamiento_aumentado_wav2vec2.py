import os
from tune_base_pipeline import BaseTransformerPipeline, Wav2Vec2MultiTask
from transformers import Wav2Vec2FeatureExtractor

"""
Mejor Config inicial:
LR = 5e-05
BATCH_SIZE = 4
GRAD_STEPS = 8
EPOCHS = 8
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 250

run_id = 377149536585453fa6f857e8e6d6dfe7
cv_mean_f1_grupo = 0.2814308920146893
cv_mean_f1_caja = 0.21241607808126606

Vamos a probar con todo los mismo menos el LR, que vamos a poner LR = 3e-05

LR = 3e-05
BATCH_SIZE = 4
GRAD_STEPS = 8
EPOCHS = 8
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 250

run_id = 7bb456eb0dc94f9a8b3dee4bff95306a
cv_mean_f1_grupo = 0.2699329832402778
cv_mean_f1_caja = 0.21444286860073047

Empeora bastante al bajar el lr. Vamos a probar con un weight decay menor.


LR = 5e-05
BATCH_SIZE = 4
GRAD_STEPS = 8
EPOCHS = 8
WEIGHT_DECAY = 0.025
WARMUP_STEPS = 250

run_id = 22bed23ef9a645f4a75c0b18694770b6
cv_mean_f1_grupo = 0.20994386370468687
cv_mean_f1_caja = 0.22590371651870958

Vemos que ha empeorado bastante al bajar el weigth decay. Vamos a dejar dicho peso y aumentar los warmup_steps a 350.

LR = 5e-05
BATCH_SIZE = 4
GRAD_STEPS = 8
EPOCHS = 8
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 350

run_id = a0fe14a691a14bde913ee1ffd52eaa20
cv_mean_f1_grupo = 0.21736415461289482
cv_mean_f1_caja = 0.22348182421533994

Sigue siendo peor que el original.

Nos quedamos con este:
LR = 5e-05
BATCH_SIZE = 4
GRAD_STEPS = 8
EPOCHS = 8
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 250

run_id = 377149536585453fa6f857e8e6d6dfe7
cv_mean_f1_grupo = 0.2814308920146893
cv_mean_f1_caja = 0.21241607808126606
"""


LR = 5e-05
BATCH_SIZE = 4
GRAD_STEPS = 8
EPOCHS = 8
WEIGHT_DECAY = 0.025
WARMUP_STEPS = 350

class Wav2Vec2AugmentedPipeline(BaseTransformerPipeline):
    @property
    def max_audio_length(self): return 160000
    @property
    def nombre_dataset(self): return "Aumentado"
    @property
    def nombre_modelo(self): return "facebook/wav2vec2-base-960h"
    @property
    def ruta_audios(self): return "audios_aumentados"
    @property
    def csv_train(self): return "metadata_train_aumentado.csv"
    @property
    def csv_test(self): return "metadata_test_aumentado.csv"
    @property
    def nombre_run(self): return f"Wav2Vec2_Chunk_Augmented_{EPOCHS}epochs"
    @property
    def nombre_modelo_guardado(self): return "modelo_multitask_augmented_wav2vec2"

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
        return Wav2Vec2MultiTask(self.nombre_modelo, num_labels_grupo, num_labels_caja)

    def get_feature_extractor(self):
        return Wav2Vec2FeatureExtractor.from_pretrained(self.nombre_modelo)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pipeline = Wav2Vec2AugmentedPipeline()
    pipeline.ejecutar()
