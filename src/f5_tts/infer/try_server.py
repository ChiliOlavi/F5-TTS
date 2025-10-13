import requests

API_URL = "http://localhost:7860/infer"

def call_infer_api(language, ref_text, gen_text, audio_path):
    data = {
            "language": language,
            "ref_text": ref_text,
            "gen_text": gen_text,
        }
    if audio_path is not None:
        with open(audio_path, "rb") as audio_file:
            files = {"audio_file": ("input.wav", audio_file, "audio/wav")}
    
        response = requests.post(API_URL, data=data, files=files)
    else:
        response = requests.post(API_URL, data=data)
        if response.status_code == 200:
            with open("output_sami.wav", "wb") as out_f:
                out_f.write(response.content)
            print("Generated audio saved as output_sami.wav")
        else:
            print("Error:", response.json())

if __name__ == "__main__":
    # Example usage
    call_infer_api(
        language="North Sámi",  # just the language name
        ref_text="Jävla helvete.",
        gen_text="Davvisámegiella gullá sámegielaid oarjesámegielaid davvejovkui ovttas julev- ja bihtánsámegielain.",
        audio_path=None,
    )