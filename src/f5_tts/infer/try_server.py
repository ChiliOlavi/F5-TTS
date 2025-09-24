import requests

API_URL = "http://localhost:7860/infer"

def call_infer_api(language, ref_text, gen_text, audio_path):
    with open(audio_path, "rb") as audio_file:
        files = {"audio_file": ("input.wav", audio_file, "audio/wav")}
        data = {
            "language": language,
            "ref_text": ref_text,
            "gen_text": gen_text,
        }
        response = requests.post(API_URL, data=data, files=files)
        if response.status_code == 200:
            with open("output.wav", "wb") as out_f:
                out_f.write(response.content)
            print("Generated audio saved as output.wav")
        else:
            print("Error:", response.json())

if __name__ == "__main__":
    # Example usage
    call_infer_api(
        language="Swedish",  # just the language name
        ref_text="Jävla helvete.",
        gen_text="Jag tycker om att programmera.",
        audio_path="javla_helvete.wav"
    )