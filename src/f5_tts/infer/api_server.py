import json
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

app = FastAPI()

DEFAULT_TTS_MODEL_CFG = [
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]

SWEDISH_MODEL_PATH = "E:\\F5_TTS\\F5-TTS\\ckpts\\swedish_parliament_data\\model_last.pt"

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
        model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])  # Use same config unless you have a Swedish-specific one
        return load_model(DiT, model_cfg, SWEDISH_MODEL_PATH)
    elif language == "Finnish":
        ckpt_path = str(cached_path(FINNISH_MODEL_CONFIG[0]))
        model_cfg = json.loads(FINNISH_MODEL_CONFIG[2])
        return load_model(DiT, model_cfg, ckpt_path, vocab_file=str(cached_path(FINNISH_MODEL_CONFIG[1])))
    else:
        raise ValueError(f"Unsupported language: {language}")

@app.post("/infer")
async def infer_api(
    language: str = Form(...),
    ref_text: str = Form(...),
    gen_text: str = Form(...),
    audio_file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        contents = await audio_file.read()
        tmp.write(contents)
        ref_audio_path = tmp.name

    out_wav_path = ref_audio_path + "_gen.wav"
    try:
        model = get_model(language)
        # Use infer_process instead of model.infer
        wav, sr, spec = infer_process(
            ref_audio_path,
            ref_text,
            gen_text,
            model,
            vocoder,
            #file_wave=out_wav_path,
            #file_spec=None,
            #seed=666,
            nfe_step=32,
            cfg_strength=2,
            sway_sampling_coef=-1,
            speed=1.0,
            #remove_silence=False,
        )
        # save the output wav:
        wavfile.write(out_wav_path, sr, wav)
        background_tasks.add_task(os.remove, ref_audio_path)
        background_tasks.add_task(os.remove, out_wav_path)
        return FileResponse(out_wav_path, media_type="audio/wav", filename=out_wav_path, background=background_tasks)
    except Exception as e:
        if os.path.exists(ref_audio_path):
            os.remove(ref_audio_path)
        if os.path.exists(out_wav_path):
            os.remove(out_wav_path)
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)