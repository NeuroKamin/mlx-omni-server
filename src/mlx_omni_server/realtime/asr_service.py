import os
import numpy as np
import tempfile
import soundfile as sf
import gigaam
from sbert_punc_case_ru import SbertPuncCase

os.environ["TOKENIZERS_PARALLELISM"] = "false"

asr_model = gigaam.load_model("v2_rnnt", fp16_encoder=True)
punc_model = SbertPuncCase()


class RealtimeASRService:
    sample_rate = 16000

    def transcribe(self, audio_bytes: bytes) -> str:
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_array, self.sample_rate)
            raw_text = asr_model.transcribe(tmp.name)
            result = punc_model.punctuate(raw_text)

        os.remove(tmp.name)
        return result.strip()
