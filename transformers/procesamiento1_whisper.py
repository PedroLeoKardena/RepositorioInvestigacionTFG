import os
from tune_base_pipeline import BaseTransformerPipeline, WhisperMultiTask

LR = 3e-5
BATCH_SIZE = 4
GRAD_STEPS = 2
EPOCHS = 5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 0

class WhisperBaselinePipeline(BaseTransformerPipeline):
    @property
    def max_audio_length(self): return None
    @property
    def nombre_dataset(self): return "Baseline"
    @property
    def nombre_modelo(self): return "openai/whisper-large-v3-turbo"
    @property
    def ruta_audios(self): return "audios_chunks"
    @property
    def csv_train(self): return "metadata_train_chunked.csv"
    @property
    def csv_test(self): return "metadata_test_chunked.csv"
    @property
    def nombre_run(self): return f"Whisper_Chunk_Baseline_{EPOCHS}epochs"
    @property
    def nombre_modelo_guardado(self): return "modelo_multitask_Whisper"

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
        return WhisperMultiTask(self.nombre_modelo, num_labels_grupo, num_labels_caja)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pipeline = WhisperBaselinePipeline()
    pipeline.ejecutar()
