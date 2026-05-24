import os
from base_pipeline import BaseTransformerPipeline, Wav2Vec2MultiTask
from transformers import Wav2Vec2FeatureExtractor

LR = 3e-5
BATCH_SIZE = 4
GRAD_STEPS = 2
EPOCHS = 5
WEIGHT_DECAY = 0.01

class Wav2Vec2DiarizadoPipeline(BaseTransformerPipeline):
    @property
    def nombre_modelo(self): return "facebook/wav2vec2-base-960h"
    @property
    def ruta_audios(self): return "audios_chunks_diarizados"
    @property
    def csv_train(self): return "metadata_train_chunked_diarizado.csv"
    @property
    def csv_test(self): return "metadata_test_chunked_diarizado.csv"
    @property
    def nombre_run(self): return "Wav2Vec2_Chunk_Diarizado_5epochs"
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

    def get_multitask_model(self, num_labels_grupo, num_labels_caja):
        return Wav2Vec2MultiTask(self.nombre_modelo, num_labels_grupo, num_labels_caja)

    def get_feature_extractor(self):
        return Wav2Vec2FeatureExtractor.from_pretrained(self.nombre_modelo)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pipeline = Wav2Vec2DiarizadoPipeline()
    pipeline.ejecutar()
