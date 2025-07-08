import numpy as np
import mlx.core as mx
from mlx_whisper.transcribe import ModelHolder, transcribe as whisper_transcribe


class RealtimeASRService:
    sample_rate = 16000

    def __init__(self, model_path="models/large-v3", dtype: mx.Dtype = mx.float16):
        self.model_path = model_path
        self.dtype = dtype

    def transcribe(self, audio_bytes: bytes) -> str:
        # Преобразуем байты в float32 numpy array
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Ensure the model is loaded
        ModelHolder.get_model(self.model_path, self.dtype)

        # Передаём аудио в whisper
        result = whisper_transcribe(audio_array, path_or_hf_repo=self.model_path)
        return result["text"].strip()
