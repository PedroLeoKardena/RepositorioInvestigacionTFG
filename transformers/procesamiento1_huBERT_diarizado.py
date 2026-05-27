import os
from base_pipeline import BaseTransformerPipeline, HubertMultiTask

LR = 3e-5
BATCH_SIZE = 4
GRAD_STEPS = 2
EPOCHS = 5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

class HubertDiarizadoPipeline(BaseTransformerPipeline):
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
    def warmup_ratio(self): return WARMUP_RATIO

    def get_multitask_model(self, num_labels_grupo, num_labels_caja):
        return HubertMultiTask(self.nombre_modelo, num_labels_grupo, num_labels_caja)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pipeline = HubertDiarizadoPipeline()
    pipeline.ejecutar()
