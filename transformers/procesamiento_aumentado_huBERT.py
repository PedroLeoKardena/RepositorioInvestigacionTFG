import os
from tune_base_pipeline import BaseTransformerPipeline, HubertMultiTask

LR = 3e-5
BATCH_SIZE = 4
GRAD_STEPS = 4
EPOCHS = 10
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 250


class HubertAugmentedPipeline(BaseTransformerPipeline):
    @property
    def max_audio_length(self): return 16000
    @property
    def nombre_dataset(self): return "Aumentado"
    @property
    def nombre_modelo(self): return "facebook/hubert-base-ls960"
    @property
    def ruta_audios(self): return "audios_aumentados"
    @property
    def csv_train(self): return "metadata_train_aumentado.csv"
    @property
    def csv_test(self): return "metadata_test_aumentado.csv"
    @property
    def nombre_run(self): return f"HuBERT_Chunk_Augmented_{EPOCHS}epochs"
    @property
    def nombre_modelo_guardado(self): return "modelo_multitask_augmented_hubert"

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
    pipeline = HubertAugmentedPipeline()
    pipeline.ejecutar()
