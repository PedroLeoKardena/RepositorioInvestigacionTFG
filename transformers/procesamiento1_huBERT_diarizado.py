import os
from tune_base_pipeline import BaseTransformerPipeline, HubertMultiTask

"""
Mejor Inicial:

LR = 5e-5
BATCH_SIZE = 4
GRAD_STEPS = 8
EPOCHS = 8
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 250

run_id = a244caf4fc2a4df29d0a4531cb17b95e
cv_mean_grupo_f1 = 0.3289682993605128
cv_mean_caja_f1 = 0.16840383313317947

Vamos a probar a bajar solo el lr a 3e-5.

LR = 3e-5
BATCH_SIZE = 4
GRAD_STEPS = 8
EPOCHS = 8
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 250


run_id = 048eeb8063684110b27d894a56c363d3
cv_mean_grupo_f1 = 0.27489487164280435
cv_mean_caja_f1 = 0.1483513024712503

Empeora.

Dejamos para grupo esta configuración:
Config Grupo Final:

LR = 5e-5
BATCH_SIZE = 4
GRAD_STEPS = 8
EPOCHS = 8
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 250

run_id = a244caf4fc2a4df29d0a4531cb17b95e
cv_mean_grupo_f1 = 0.3289682993605128
cv_mean_caja_f1 = 0.16840383313317947
"""

LR = 5e-5
BATCH_SIZE = 4
GRAD_STEPS = 8
EPOCHS = 8
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 250



class HubertDiarizadoPipeline(BaseTransformerPipeline):
    @property
    def max_audio_length(self): return 160000
    @property
    def nombre_dataset(self): return "Diarizado"
    @property
    def nombre_modelo(self): return "facebook/hubert-base-ls960"
    @property
    def ruta_audios(self): return "audios_chunks_diarizados"
    @property
    def csv_train(self): return "metadata_train_chunked_diarizado.csv"
    @property
    def csv_test(self): return "metadata_test_chunked_diarizado.csv"
    @property
    def nombre_run(self): return f"HuBERT_Chunk_Diarizado_{EPOCHS}epochs"
    @property
    def nombre_modelo_guardado(self): return "modelo_multitask_hubert_diarizado"

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
    pipeline = HubertDiarizadoPipeline()
    pipeline.ejecutar()
