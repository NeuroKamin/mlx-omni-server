import os
from pathlib import Path
from typing import Union
from .whisper_mlx import WhisperModel
from .whisper_cpp import WhisperCppModel

from .schema import (
    STTRequestForm,
    TranscriptionResponse,
)

class STTService:
    def __init__(self):
        self.model = WhisperModel()

    async def transcribe(
        self,
        request: STTRequestForm,
    ) -> Union[dict, str, TranscriptionResponse]:
        try:
            if 'whisper.cpp' in request.model:
                self.model = WhisperCppModel(
                    whisper_cli_path=os.getenv("WHISPER_CPP_CLI", "./whisper.cpp/build/bin/whisper-cli"),
                    model_path=os.getenv("WHISPER_CPP_MODEL", "./whisper.cpp/models/ggml-large-v3.bin"),
                    vad_model_path=os.getenv("WHISPER_CPP_VAD_MODEL", "./whisper.cpp/models/ggml-silero-v5.1.2.bin"),
                    threads=int(os.getenv("WHISPER_CPP_THREADS", "32")),
                )

            audio_path = await self.model._save_upload_file(request.file)
            result = self.model.generate(audio_path=audio_path, request=request)
            response = self.model._format_response(result, request)
            Path(audio_path).unlink(missing_ok=True)
            return response

        except Exception as e:
            if "audio_path" in locals():
                Path(audio_path).unlink(missing_ok=True)
            raise e
        