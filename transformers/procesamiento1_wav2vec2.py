import os
from tune_base_pipeline import BaseTransformerPipeline, Wav2Vec2MultiTask
from transformers import Wav2Vec2FeatureExtractor

"""
Mejor config inicial:
LR = 3e-5
BATCH_SIZE = 4
GRAD_STEPS = 8
EPOCHS = 8
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 250

run_id = d6ed5f2883f24595ab6455c27c0a6299
cv_mean_f1_grupo = 0.25391625609933444
cv_mean_f1_caja = 0.2036732665538099


Vamos a probar a aumentar el weight_decay a 0.05:

LR = 3e-5
BATCH_SIZE = 4
GRAD_STEPS = 8
EPOCHS = 8
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 250

run_id = f227351f8ab64227b19d3a8b19425fa1
cv_mean_f1_grupo = 0.2000220235838865
cv_mean_f1_caja = 0.1896308338192601

Empeora bastante, vamos a probar mas learning_steps (350):

LR = 3e-5
BATCH_SIZE = 4
GRAD_STEPS = 8
EPOCHS = 8
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 350

run_id = cf09455ecf634926a035676eacb8a9ca
cv_mean_f1_grupo = 0.19230135785631014
cv_mean_f1_caja = 0.15588164996924014

Como sigue empeorando vamos a dejar esta configuraciín como la final:

Configuración Final:
LR = 3e-5
BATCH_SIZE = 4
GRAD_STEPS = 8
EPOCHS = 8
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 250

run_id = d6ed5f2883f24595ab6455c27c0a6299
cv_mean_f1_grupo = 0.25391625609933444
cv_mean_f1_caja = 0.2036732665538099
"""

LR = 3e-5
BATCH_SIZE = 4
GRAD_STEPS = 8
EPOCHS = 8
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 350

class Wav2Vec2BaselinePipeline(BaseTransformerPipeline):
    @property
    def max_audio_length(self): return 160000
    @property
    def nombre_dataset(self): return "Baseline"
    @property
    def nombre_modelo(self): return "facebook/wav2vec2-base-960h"
    @property
    def ruta_audios(self): return "audios_chunks"
    @property
    def csv_train(self): return "metadata_train_chunked.csv"
    @property
    def csv_test(self): return "metadata_test_chunked.csv"
    @property
    def nombre_run(self): return f"Wav2Vec2_Chunk_Baseline_{EPOCHS}epochs"
    @property
    def nombre_modelo_guardado(self): return "modelo_multitask_wav2vec2"

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
    pipeline = Wav2Vec2BaselinePipeline()
    pipeline.ejecutar()
