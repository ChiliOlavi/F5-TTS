import json
import dotenv
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import tempfile
import os
import uvicorn
from cached_path import cached_path
from f5_tts.infer.utils_infer import load_model, infer_process, load_vocoder
from f5_tts.model import DiT
from functools import lru_cache
from scipy.io import wavfile
import requests
import io
from dotenv import load_dotenv
load_dotenv()
from typing import Optional

HF_TOKEN = os.environ["RK_TTS_TOKEN"]
SWEDISH_MODEL_PATH = os.environ["SWEDISH_MODEL_PATH"]
SWEDISH_VOCAB_PATH = os.environ["SWEDISH_VOCAB_PATH"]


app = FastAPI()

DEFAULT_TTS_MODEL_CFG = [
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]


SWEDISH_MODEL_CONFIG = [
    SWEDISH_MODEL_PATH,
    SWEDISH_VOCAB_PATH,
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4))
]

FINNISH_MODEL_CONFIG = [
    "hf://AsmoKoskinen/F5-TTS_Finnish_Model/model_commonvoice_fi_librivox_fi_vox_populi_fi_20250323/model_last_20250323.safetensors",
    "hf://AsmoKoskinen/F5-TTS_Finnish_Model/model_commonvoice_fi_librivox_fi_vox_populi_fi_20250323/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]

vocoder = load_vocoder()

@lru_cache(maxsize=2)
def get_model(language: str):
    if language == "English":
        ckpt_path = str(cached_path(DEFAULT_TTS_MODEL_CFG[0]))
        model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
        return load_model(DiT, model_cfg, ckpt_path)
    elif language == "Swedish":
        if HF_TOKEN is None:
            raise ValueError("HF_TOKEN is not set in environment variables")
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        ckpt_path = str(cached_path(SWEDISH_MODEL_CONFIG[0], headers=headers))
        model_cfg = json.loads(SWEDISH_MODEL_CONFIG[2])  # Use same config unless you have a Swedish-specific one
        return load_model(DiT, model_cfg, ckpt_path)
    elif language == "Finnish":
        ckpt_path = str(cached_path(FINNISH_MODEL_CONFIG[0]))
        model_cfg = json.loads(FINNISH_MODEL_CONFIG[2])
        return load_model(DiT, model_cfg, ckpt_path, vocab_file=str(cached_path(FINNISH_MODEL_CONFIG[1])))
    else:
        raise ValueError(f"Unsupported language: {language}")


def get_sami_tts(text: str):
    r = requests.post("https://api-giellalt.uit.no/tts/se/sunna", json={"text": text})
    if r.status_code == 200:
        # Convert bytes to NumPy array and sample rate
        wav_bytes = io.BytesIO(r.content)
        sr, wav = wavfile.read(wav_bytes)
        return wav, sr
    else:
        raise RuntimeError(f"Sami TTS API error: {r.status_code} {r.text}")


@app.post("/infer")
async def infer_api(
    language: str = Form(...),
    ref_text: str = Form(...),
    gen_text: str = Form(...),
    audio_file: Optional[UploadFile] = File(None),
    background_tasks: BackgroundTasks = None,
):
    if language == "North Sámi":
        if ref_text is not None or audio_file is not None:
            return JSONResponse({"error": "only gen_text is allowed for North Sámi"}, status_code=400)
        wav, sr = get_sami_tts(gen_text)
        out_wav_path = "sami_tts_output.wav"
        ref_audio_path = None
    else:
        if audio_file is None or ref_text is None or gen_text is None:
            return JSONResponse({"error": "audio_file, ref_text, and gen_text are required for this language"}, status_code=400)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            contents = await audio_file.read()
            tmp.write(contents)
            ref_audio_path = tmp.name

        out_wav_path = ref_audio_path + "_gen.wav"

        model = get_model(language)
        wav, sr, spec = infer_process(
            ref_audio_path,
            ref_text,
            gen_text,
            model,
            vocoder,
            nfe_step=32,
            cfg_strength=2,
            sway_sampling_coef=-1,
            speed=1.0,
        )
    try:
        wavfile.write(out_wav_path, sr, wav)
        if ref_audio_path:
            background_tasks.add_task(os.remove, ref_audio_path)
        background_tasks.add_task(os.remove, out_wav_path)
        return FileResponse(out_wav_path, media_type="audio/wav", filename=out_wav_path, background=background_tasks)
    except Exception as e:
        if ref_audio_path and os.path.exists(ref_audio_path):
            os.remove(ref_audio_path)
        if os.path.exists(out_wav_path):
            os.remove(out_wav_path)
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)