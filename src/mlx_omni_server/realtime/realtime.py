from fastapi import APIRouter, Request, HTTPException
from .asr_service import RealtimeASRService

router = APIRouter(tags=["realtime"])

asr_service = RealtimeASRService()


@router.post("/realtime/predict")
@router.post("/v1/realtime/predict")
async def predict_audio(request: Request):
    try:
        audio_bytes = await request.body()
        text = asr_service.transcribe(audio_bytes)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
