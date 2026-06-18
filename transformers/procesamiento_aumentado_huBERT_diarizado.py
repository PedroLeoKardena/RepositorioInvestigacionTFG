import os
from tune_base_pipeline import BaseTransformerPipeline, HubertMultiTask

"""
LR = 5e-5
BATCH_SIZE = 4
GRAD_STEPS = 8
EPOCHS = 8
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 250

run_id = ac04c0df37844e16b3b77c629dd848db
cv_mean_f1_grupo = 0.1582361814157496
cv_mean_f1_caja = 0.08404578515298261 

Vamos a probar a bajar el lr y dejar todo lo demas igual:

LR = 3e-5
BATCH_SIZE = 4
GRAD_STEPS = 8
EPOCHS = 8
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 250

Resultados malos.
"""

#Vamos a probar a bajar el valor de learning_rate a 3e-05, aumentamos valor de epochs. Dejamos mismo numero de wamup_steps.
#Siguiente puede ser probar a aumentar weitght_decay a 0.05 o warmup_steps a 250.

LR = 3e-5
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
