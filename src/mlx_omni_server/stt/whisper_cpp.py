import os
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Union, List
import re
import logging

from .schema import (
    ResponseFormat,
    STTRequestForm,
    TranscriptionResponse,
    TranscriptionWord,
)


class WhisperCppModel:
    def __init__(
        self,
        whisper_cli_path: str = "./whisper.cpp/build/bin/whisper-cli",
        model_path: str = "./whisper.cpp/models/ggml-large-v3.bin",
        vad_model_path: str = "./whisper.cpp/models/ggml-silero-v5.1.2.bin",
        threads: int = 32,
    ):
        # Преобразуем в абсолютные пути
        self.whisper_cli_path = os.path.abspath(whisper_cli_path)
        self.model_path = os.path.abspath(model_path)
        self.vad_model_path = os.path.abspath(vad_model_path)
        self.threads = threads
        
        # Проверяем существование файлов
        if not os.path.exists(self.whisper_cli_path):
            raise Exception(f"Whisper CLI not found at: {self.whisper_cli_path}")
        if not os.path.exists(self.model_path):
            raise Exception(f"Model not found at: {self.model_path}")
        if not os.path.exists(self.vad_model_path):
            logging.warning(f"VAD model not found at: {self.vad_model_path}")

    async def _save_upload_file(self, file) -> str:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            return tmp.name

    def _get_audio_duration(self, file_path: str) -> float:
        """Получить длительность аудиофайла в секундах"""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    file_path,
                ],
                capture_output=True,
                text=True,
            )
            return float(result.stdout.strip())
        except:
            return 0

    def _build_whisper_command(
        self, audio_path: str, request: STTRequestForm, temp_dir: str
    ) -> List[str]:
        """Построить команду для whisper.cpp"""
        duration_seconds = self._get_audio_duration(audio_path)
        duration_minutes = duration_seconds / 60

        # Определяем путь к модели (из request или по умолчанию)
        model_path = self.model_path
        if request.model and request.model.endswith('.bin'):
            # Если передан путь к .bin файлу, используем его
            model_path = request.model

        # Базовое имя для выходных файлов (без расширения)
        output_base = os.path.join(temp_dir, "output")

        cmd = [
            self.whisper_cli_path,
            # Основные параметры
            "--threads", str(self.threads),
            "--model", model_path,
            "--file", audio_path,
            # Параметры декодирования
            "--temperature", str(request.temperature) if request.temperature else "0.2",
            "--word-thold", "0.005",
            "--no-speech-thold", "0.4",
            # Контекст и длина
            "--max-len", "448",
            # Дополнительные улучшения
            "--suppress-nst",
            "--flash-attn",
            # Вывод JSON
            "--output-json",
            "--output-file", output_base,
            # Не печатать цвета в stdout
            "--no-prints",
        ]

        # VAD параметры (если модель существует)
        if os.path.exists(self.vad_model_path):
            cmd.extend([
                "--vad",
                "--vad-model", self.vad_model_path,
                "--vad-threshold", "0.3",
                "--vad-min-speech-duration-ms", "200",
                "--vad-min-silence-duration-ms", "300",
                "--vad-speech-pad-ms", "50",
            ])

        # Язык
        if request.language:
            cmd.extend(["--language", request.language])

        # Prompt
        if request.prompt:
            cmd.extend(["--prompt", request.prompt])

        # Параметры в зависимости от длительности
        if duration_minutes < 5:
            cmd.extend([
                "--best-of", "5",
                "--beam-size", "5",
            ])
        else:
            cmd.extend([
                "--best-of", "3",
                "--beam-size", "3",
                "--max-context", "0",
                "--entropy-thold", "2.5",
            ])

        # Временные метки для слов
        if request.timestamp_granularities and "word" in [
            g.value for g in request.timestamp_granularities
        ]:
            cmd.append("--word-timestamps")
            # Также добавим full JSON для получения слов
            cmd.append("--output-json-full")

        return cmd

    def _parse_stdout_output(self, stdout: str) -> dict:
        """Парсинг текстового вывода whisper.cpp"""
        result = {
            "text": "",
            "language": "ru",  # по умолчанию
            "segments": []
        }
        
        # Парсим строки вида [00:00:00.940 --> 00:00:01.410]   Hello?
        lines = stdout.strip().split('\n')
        full_text = []
        
        for i, line in enumerate(lines):
            match = re.match(r'\[(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\]\s+(.*)', line)
            if match:
                start_time = self._time_to_seconds(match.group(1))
                end_time = self._time_to_seconds(match.group(2))
                text = match.group(3).strip()
                
                full_text.append(text)
                
                segment = {
                    "id": i,
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                    "words": []
                }
                result["segments"].append(segment)
        
        result["text"] = " ".join(full_text)
        return result
    
    def _time_to_seconds(self, time_str: str) -> float:
        """Конвертация времени вида 00:00:00.000 в секунды"""
        parts = time_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds

    def _parse_whisper_output(self, json_path: str) -> dict:
        """Парсинг JSON вывода whisper.cpp"""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        transcription = data.get("transcription", [])

        # Собираем весь текст
        full_text = " ".join(seg.get("text", "").strip() for seg in transcription if seg.get("text"))

        result = {
            "text": full_text,
            "language": data.get("result", {}).get("language", ""),  # из вложенного поля
            "segments": [],
        }

        # Обрабатываем сегменты
        for idx, segment in enumerate(transcription):
            start_time = segment.get("offsets", {}).get("from", 0.0) / 1000
            end_time = segment.get("offsets", {}).get("to", 0.0) / 1000

            seg = {
                "id": idx,
                "start": start_time,
                "end": end_time,
                "text": segment.get("text", ""),
                "words": [],  # whisper.cpp может не давать по-словной разбивки
            }

            result["segments"].append(seg)

        return result


    def generate(self, audio_path: str, request: STTRequestForm) -> dict:
        """Транскрибация через whisper.cpp"""
        temp_dir = tempfile.mkdtemp()
        try:
            # Строим команду
            cmd = self._build_whisper_command(audio_path, request, temp_dir)

            # Запускаем whisper.cpp
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False
            )
            
            # Проверяем, какие файлы были созданы
            files_created = os.listdir(temp_dir)

            # Ищем JSON файл - должен быть output.json
            json_path = os.path.join(temp_dir, "output.json")
            
            if os.path.exists(json_path):
                return self._parse_whisper_output(json_path)
            else:
                # Если JSON не создан, но процесс завершился успешно
                if result.returncode == 0:
                    # Попробуем найти любой JSON файл
                    json_files = [f for f in files_created if f.endswith('.json')]
                    if json_files:
                        json_path = os.path.join(temp_dir, json_files[0])
                        return self._parse_whisper_output(json_path)
                    
                    # Если JSON файлов нет, но есть stdout, парсим его
                    if result.stdout:
                        return self._parse_stdout_output(result.stdout)
                
                raise Exception(f"Whisper.cpp failed with code {result.returncode}: {result.stderr}")

        except Exception as e:
            raise e
        finally:
            # Очищаем временные файлы
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _generate_subtitle_file(self, result: dict, format: str) -> str:
        """Генерация субтитров из результата"""
        content = []
        
        if format == "srt":
            for i, segment in enumerate(result.get("segments", [])):
                content.append(str(i + 1))
                start_time = self._seconds_to_srt_time(segment["start"])
                end_time = self._seconds_to_srt_time(segment["end"])
                content.append(f"{start_time} --> {end_time}")
                content.append(segment["text"].strip())
                content.append("")
        
        elif format == "vtt":
            content.append("WEBVTT")
            content.append("")
            for segment in result.get("segments", []):
                start_time = self._seconds_to_vtt_time(segment["start"])
                end_time = self._seconds_to_vtt_time(segment["end"])
                content.append(f"{start_time} --> {end_time}")
                content.append(segment["text"].strip())
                content.append("")
        
        return "\n".join(content)

    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Конвертация секунд в формат времени SRT"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")

    def _seconds_to_vtt_time(self, seconds: float) -> str:
        """Конвертация секунд в формат времени VTT"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

    def _format_response(
        self, result: dict, request: STTRequestForm
    ) -> Union[dict, str, TranscriptionResponse]:
        """Форматирование ответа"""
        if request.response_format == ResponseFormat.TEXT:
            return result["text"]

        elif request.response_format == ResponseFormat.SRT:
            return self._generate_subtitle_file(result, "srt")

        elif request.response_format == ResponseFormat.VTT:
            return self._generate_subtitle_file(result, "vtt")

        elif request.response_format == ResponseFormat.VERBOSE_JSON:
            return result

        elif request.response_format == ResponseFormat.JSON:
            return {"text": result["text"]}

        else:
            text = result.get("text", "")
            language = result.get("language", "en")

            duration = 0
            if "segments" in result:
                for segment in result["segments"]:
                    if "end" in segment:
                        duration = max(duration, segment["end"])

            words = []
            if request.timestamp_granularities and "word" in [
                g.value for g in request.timestamp_granularities
            ]:
                for segment in result.get("segments", []):
                    for word_data in segment.get("words", []):
                        word = TranscriptionWord(
                            word=word_data["word"],
                            start=word_data["start"],
                            end=word_data["end"],
                        )
                        words.append(word)

            return TranscriptionResponse(
                task="transcribe",
                language=language,
                duration=duration,
                text=text,
                words=words if words else None,
            )