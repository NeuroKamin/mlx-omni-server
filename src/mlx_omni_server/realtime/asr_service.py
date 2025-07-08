import numpy as np
from mlx_whisper import transcribe as whisper_transcribe


class RealtimeASRService:
    sample_rate = 16000

    def __init__(self, model_path="mlx-community/whisper-large-v3-mlx"):
        self.model_path = model_path

    def transcribe(self, audio_bytes: bytes) -> str:
        # Преобразуем байты в float32 numpy array
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Передаём аудио в whisper
        result = whisper_transcribe(audio_array, path_or_hf_repo=self.model_path)
        return result["text"].strip()
