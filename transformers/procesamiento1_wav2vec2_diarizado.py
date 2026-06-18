import os
from tune_base_pipeline import BaseTransformerPipeline, Wav2Vec2MultiTask
from transformers import Wav2Vec2FeatureExtractor

"""
Mejor config inicial:
LR = 3e-5
BATCH_SIZE = 4
GRAD_STEPS = 2
EPOCHS = 5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100

run_id = 53fc98acfd7d416d9f74dbff432cbb55
cv_mean_f1_grupo = 0.32558649198144013
cv_mean_f1_caja = 0.21170940494848298


Vamos a probar a aumentar los grad_steps a 4 y los epochs a 8.

LR = 3e-5
BATCH_SIZE = 4
GRAD_STEPS = 4
EPOCHS = 8
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100

run_id = be69a3789105482fb7f15e124aba26c5
cv_mean_f1_grupo = 0.3310298984522817
cv_mean_f1_caja = 0.17483199667211732

Empeora bastante en caja. Vamos a probar con mas grad_steps y mas warump steps.

LR = 3e-5
BATCH_SIZE = 4
GRAD_STEPS = 4
EPOCHS = 5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 250

run_id = f7273537d9f144e8ae1216b06ba6283a
cv_mean_f1_grupo = 0.2784848197879978
cv_mean_f1_caja = 0.16116990366695716

Empeoran ambos. Vamos a dejar esta configuración como la final entonces:

CONFIGURACIÓN FINAL:
LR = 3e-5
BATCH_SIZE = 4
GRAD_STEPS = 2
EPOCHS = 5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100

run_id = 53fc98acfd7d416d9f74dbff432cbb55
cv_mean_f1_grupo = 0.32558649198144013
cv_mean_f1_caja = 0.21170940494848298

Lanzamos este para grupo
"""


#Bajamos el valor de LR y dejamos los demas ugual. A la siguiente probar a aumentar numero de epochs y numero warmup_steps.
LR = 3e-5
BATCH_SIZE = 4
GRAD_STEPS = 2
EPOCHS = 5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100


class Wav2Vec2DiarizadoPipeline(BaseTransformerPipeline):
    @property
    def max_audio_length(self): return 160000
    @property
    def nombre_dataset(self): return "Diarizado"

    @property
    def nombre_modelo(self): return "facebook/wav2vec2-base-960h"
    @property
    def ruta_audios(self): return "audios_chunks_diarizados"
    @property
    def csv_train(self): return "metadata_train_chunked_diarizado.csv"
    @property
    def csv_test(self): return "metadata_test_chunked_diarizado.csv"
    @property
    def nombre_run(self): return f"Wav2Vec2_Chunk_Diarizado_{EPOCHS}epochs"
    @property
    def nombre_modelo_guardado(self): return "modelo_multitask_wav2vec2_diarizado"

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
    pipeline = Wav2Vec2DiarizadoPipeline()
    pipeline.ejecutar()
