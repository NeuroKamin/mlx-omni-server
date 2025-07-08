from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from datetime import datetime, timedelta
import json
from .asr_service import RealtimeASRService
import numpy as np

router = APIRouter(tags=["realtime"])

PHRASE_TIMEOUT = 3.0  # seconds

asr_service = RealtimeASRService()

@router.websocket("/ws/realtime")
@router.websocket("/v1/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    phrase_bytes = bytearray()
    phrase_time = None

    try:
        while True:
            data = await websocket.receive_bytes()
            now = datetime.utcnow()

            if phrase_time and now - phrase_time > timedelta(seconds=PHRASE_TIMEOUT):
                # Обнуляем фразу при таймауте
                phrase_bytes = bytearray()
                phrase_complete = True
            else:
                phrase_complete = False

            phrase_time = now
            phrase_bytes.extend(data)

            try:
                # Распознаем
                text = asr_service.transcribe(bytes(phrase_bytes))

                await websocket.send_text(json.dumps({
                    "text": text,
                    "phrase_complete": phrase_complete
                }))
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "error": str(e)
                }))

    except WebSocketDisconnect:
        print("Client disconnected")
