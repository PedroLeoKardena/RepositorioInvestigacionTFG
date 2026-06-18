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

run_id = e3e5f37eb244405097fd0ebb72f2ba8d
cv_mean_f1_grupo = 0.3127828687968345
cv_mean_f1_caja = 0.22496616833005884


Vamos a probar a aumentar el weight_decay a 0.05:

LR = 3e-5
BATCH_SIZE = 4
GRAD_STEPS = 8
EPOCHS = 8
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 250

run_id = 54dc1fde4abe40c199e17a5440ea0095
cv_mean_f1_grupo = 0.3101675092069298
cv_mean_f1_caja = 0.19445868636822733

Empeora, vamos a probar con solo aumentar los warmup_steps:

LR = 3e-5
BATCH_SIZE = 4
GRAD_STEPS = 8
EPOCHS = 8
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 350

run_id = 0ca02f5405f64f64a983d7867d50811d
cv_mean_f1_grupo = 0.2674952561013387
cv_mean_f1_caja = 0.2100281608639038

Sigue empeorando. Dejamos la mejor configuración como la final:

Configuración Final:

LR = 3e-5
BATCH_SIZE = 4
GRAD_STEPS = 8
EPOCHS = 8
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 250

run_id = e3e5f37eb244405097fd0ebb72f2ba8d
cv_mean_f1_grupo = 0.3127828687968345
cv_mean_f1_caja = 0.22496616833005884

Lanzamos este para caja.
"""



LR = 3e-5
BATCH_SIZE = 4
GRAD_STEPS = 8
EPOCHS = 8
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 250

class Wav2Vec2AugmentedDiarizadoPipeline(BaseTransformerPipeline):
    @property
    def max_audio_length(self): return 160000
    @property
    def nombre_dataset(self): return "Aumentado_Diarizado"
    @property
    def nombre_modelo(self): return "facebook/wav2vec2-base-960h"
    @property
    def ruta_audios(self): return "audios_aumentados_diarizados"
    @property
    def csv_train(self): return "metadata_train_aumentado_diarizado.csv"
    @property
    def csv_test(self): return "metadata_test_aumentado_diarizado.csv"
    @property
    def nombre_run(self): return f"Wav2Vec2_Diarizado_Chunk_Augmented_{EPOCHS}epochs"
    @property
    def nombre_modelo_guardado(self): return "modelo_multitask_augmented_wav2vec2_diarizado"

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
    pipeline = Wav2Vec2AugmentedDiarizadoPipeline()
    pipeline.ejecutar()
