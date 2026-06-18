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

run_id = 4d094b685b634b55b5310c38293a5946
cv_mean_grupo_f1 = 0.2585880587076065
cv_mean_caja_f1 = 0.24573444512649084

Vamos a probar a bajar solo el lr a 3e-5.

LR = 3e-5
BATCH_SIZE = 4
GRAD_STEPS = 8
EPOCHS = 8
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 250

run_id = 16aee7292edb45b4bb63e060cf39a70c
cv_mean_grupo_f1 = 0.2837171528994029
cv_mean_caja_f1 = 0.22717232362911632

Es mejor en grupo pero peor en caja.


Nos quedamos para caja con este:

LR = 5e-5
BATCH_SIZE = 4
GRAD_STEPS = 8
EPOCHS = 8
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 250

run_id = 4d094b685b634b55b5310c38293a5946
cv_mean_grupo_f1 = 0.2585880587076065
cv_mean_caja_f1 = 0.24573444512649084
"""


LR = 3e-5
BATCH_SIZE = 4
GRAD_STEPS = 8
EPOCHS = 8
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 250

class HubertBaselinePipeline(BaseTransformerPipeline):
    @property
    def max_audio_length(self): return 160000
    @property
    def nombre_dataset(self): return "Baseline"
    @property
    def nombre_modelo(self): return "facebook/hubert-base-ls960"
    @property
    def ruta_audios(self): return "audios_chunks"
    @property
    def csv_train(self): return "metadata_train_chunked.csv"
    @property
    def csv_test(self): return "metadata_test_chunked.csv"
    @property
    def nombre_run(self): return f"HuBERT_Chunk_Baseline_{EPOCHS}epochs"
    @property
    def nombre_modelo_guardado(self): return "modelo_multitask_hubert"

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
    pipeline = HubertBaselinePipeline()
    pipeline.ejecutar()
