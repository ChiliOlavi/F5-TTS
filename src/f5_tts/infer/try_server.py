import requests

API_URL = "http://localhost:7860/infer"

def call_infer_api(language, ref_text, gen_text, audio_path, **kwargs):
    data = {
            "language": language,
            "ref_text": ref_text,
            "gen_text": gen_text,
        }
    data.update(kwargs)
    if audio_path is not None:
        with open(audio_path, "rb") as audio_file:
            files = {"audio_file": ("input.wav", audio_file, "audio/wav")}
            response = requests.post(API_URL, data=data, files=files)
    else:
        response = requests.post(API_URL, data=data)
    
    if response.status_code == 200:
        print("API call successful")
        with open("output.wav", "wb") as out_f:
            out_f.write(response.content)
        print("Generated audio saved as output.wav")
    else:
        print("API call failed")
        print("Error:", response.json())

if __name__ == "__main__":
    # Example usage
    call_infer_api(
        language="English",  # just the language name
        ref_text="One of the best ways to involve yourself with both writing and magic is to create a magic journal.",
        gen_text="we are all born free and equal in dignity and rights. We are endowed with reason and conscience and should act towards one another in a spirit of brotherhood.",
        audio_path="One of the best ways to involve yourself with both writing and magic is to create a magic journal.wav",  # None for North Sámi
        nfe_step=32,
        speed=1.0,
        cross_fade_duration=0.15,
        seed=0,
        remove_silence=False  # Add this parameter (default: False)
    )