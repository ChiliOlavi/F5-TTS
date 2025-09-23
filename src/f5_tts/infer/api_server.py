from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
import tempfile
import os
import uvicorn
from importlib.resources import files
from f5_tts.api import F5TTS

app = FastAPI()

@app.post("/infer")
async def infer_api(
    model: str = Form(...),
    ref_text: str = Form(...),
    gen_text: str = Form(...),
    audio_file: UploadFile = File(...),
):
    # Save uploaded audio file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        contents = await audio_file.read()
        tmp.write(contents)
        ref_audio_path = tmp.name

    # Create F5TTS instance (you can pass model name if needed)
    f5tts = F5TTS(model=model) if hasattr(F5TTS, "model") else F5TTS()

    # Call inference
    out_wav_path = ref_audio_path + ".gen.wav"
    try:
        wav, sr, spec = f5tts.infer(
            ref_file=ref_audio_path,
            ref_text=ref_text,
            gen_text=gen_text,
            file_wave=out_wav_path,
            file_spec=None,
            seed=None,
        )
        # Return generated audio file
        return FileResponse(out_wav_path, media_type="audio/wav", filename="output.wav")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        # Clean up temp files
        if os.path.exists(ref_audio_path):
            os.remove(ref_audio_path)
        if os.path.exists(out_wav_path):
            os.remove(out_wav_path)


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=7860)