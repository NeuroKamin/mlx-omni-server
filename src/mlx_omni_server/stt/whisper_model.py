import asyncio
import os
from pathlib import Path
from typing import Dict, Tuple, Union

from .schema import STTRequestForm, TranscriptionResponse
from .whisper_cpp import WhisperCppModel
from .whisper_mlx import WhisperModel


class STTService:
    _whisper_model: WhisperModel | None = None
    _whisper_cpp_models: Dict[Tuple[str, str, str, int], WhisperCppModel] = {}
    _locks: Dict[Tuple[str, str, str, int], asyncio.Lock] = {}

    def __init__(self):
        pass

    def _get_whisper_model(self) -> WhisperModel:
        if self.__class__._whisper_model is None:
            self.__class__._whisper_model = WhisperModel()
        return self.__class__._whisper_model

    def _get_cpp_key(self) -> Tuple[str, str, str, int]:
        return (
            os.getenv("WHISPER_CPP_CLI", "./whisper.cpp/build/bin/whisper-cli"),
            os.getenv("WHISPER_CPP_MODEL", "./whisper.cpp/models/ggml-large-v3.bin"),
            os.getenv("WHISPER_CPP_VAD_MODEL", "./whisper.cpp/models/ggml-silero-v5.1.2.bin"),
            int(os.getenv("WHISPER_CPP_THREADS", "32")),
        )

    def _get_whisper_cpp_model(self) -> Tuple[WhisperCppModel, asyncio.Lock]:
        key = self._get_cpp_key()
        if key not in self.__class__._whisper_cpp_models:
            self.__class__._whisper_cpp_models[key] = WhisperCppModel(
                whisper_cli_path=key[0],
                model_path=key[1],
                vad_model_path=key[2],
                threads=key[3],
            )
            self.__class__._locks[key] = asyncio.Lock()
        return self.__class__._whisper_cpp_models[key], self.__class__._locks[key]

    async def transcribe(
        self,
        request: STTRequestForm,
    ) -> Union[dict, str, TranscriptionResponse]:
        try:
            if "whisper.cpp" in request.model:
                model, lock = self._get_whisper_cpp_model()
            else:
                model = self._get_whisper_model()
                lock = None

            audio_path = await model._save_upload_file(request.file)

            if lock:
                async with lock:
                    result = await model.generate_async(audio_path=audio_path, request=request)
            else:
                result = model.generate(audio_path=audio_path, request=request)

            response = model._format_response(result, request)
            Path(audio_path).unlink(missing_ok=True)
            return response

        except Exception as e:
            if "audio_path" in locals():
                Path(audio_path).unlink(missing_ok=True)
            raise e
