import os
from transformers.tune_base_pipeline import BaseTransformerPipeline, Wav2Vec2MultiTask
from transformers import Wav2Vec2FeatureExtractor

LR = 5e-5
BATCH_SIZE = 4
GRAD_STEPS = 4
EPOCHS = 8
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

class Wav2Vec2BaselinePipeline(BaseTransformerPipeline):
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
    def warmup_ratio(self): return WARMUP_RATIO

    def get_multitask_model(self, num_labels_grupo, num_labels_caja):
        return Wav2Vec2MultiTask(self.nombre_modelo, num_labels_grupo, num_labels_caja)

    def get_feature_extractor(self):
        return Wav2Vec2FeatureExtractor.from_pretrained(self.nombre_modelo)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pipeline = Wav2Vec2BaselinePipeline()
    pipeline.ejecutar()
