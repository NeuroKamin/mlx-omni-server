import asyncio
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

from .schema import STTRequestForm, TranscriptionResponse
from .whisper_cpp import WhisperCppModel
from .whisper_mlx import WhisperModel


class STTService:
    _whisper_model: WhisperModel | None = None
    _whisper_cpp_pools: Dict[Tuple[str, str, str, int], List[WhisperCppModel]] = {}
    _semaphores: Dict[Tuple[str, str, str, int], asyncio.Semaphore] = {}

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

    async def _acquire_cpp_model(self) -> WhisperCppModel:
        key = self._get_cpp_key()
        pool = self.__class__._whisper_cpp_pools.setdefault(key, [])
        if key not in self.__class__._semaphores:
            max_workers = int(os.getenv("WHISPER_CPP_MAX_WORKERS", "1"))
            self.__class__._semaphores[key] = asyncio.Semaphore(max_workers)

        await self.__class__._semaphores[key].acquire()
        if pool:
            return pool.pop()

        return WhisperCppModel(
            whisper_cli_path=key[0],
            model_path=key[1],
            vad_model_path=key[2],
            threads=key[3],
        )

    def _release_cpp_model(self, model: WhisperCppModel) -> None:
        key = self._get_cpp_key()
        self.__class__._whisper_cpp_pools.setdefault(key, []).append(model)
        self.__class__._semaphores[key].release()

    async def transcribe(
        self,
        request: STTRequestForm,
    ) -> Union[dict, str, TranscriptionResponse]:
        try:
            if "whisper.cpp" in request.model:
                model = await self._acquire_cpp_model()
                try:
                    audio_path = await model._save_upload_file(request.file)
                    result = await model.generate_async(
                        audio_path=audio_path, request=request
                    )
                finally:
                    if "audio_path" in locals():
                        Path(audio_path).unlink(missing_ok=True)
                    self._release_cpp_model(model)
            else:
                model = self._get_whisper_model()
                audio_path = await model._save_upload_file(request.file)
                result = model.generate(audio_path=audio_path, request=request)
                Path(audio_path).unlink(missing_ok=True)

            response = model._format_response(result, request)
            return response

        except Exception as e:
            if "audio_path" in locals():
                Path(audio_path).unlink(missing_ok=True)
            raise e
