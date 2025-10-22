import requests
import wave
import os

API_URL = "http://127.0.0.1:7860/infer"

SWEDISH_STORY = """Jag märker oftast brist på frid i kroppen innan min hjärna hinner skriva något. Min uppmärksamhet riktar sig mot kroppens reaktioner: min andedräkt spändes upp, mina bröst spreds av en tryckande känsla, mina händer svalnade. Mitt sinne har redan reagerat på dessa känslor innan min kropp är fullt medveten om dem. Kroppslig rastlöshet sätter i gång en kedja: det utlöser behovet av skydd eller anpassning. Mina tankar börjar leta efter lösningar, förbereder sig för en eventuell konflikt, bygger fantasifulla scenarier. Mellan min kropp och mitt sinne uppstår en cykel där var och en matar varandra: ett litet fysiskt tecken växer till en hel upplevelse av oro."""
FINNISH_STORY = """Rauhan puutteen huomaan useimmiten kehossa ennen kuin mieli ehtii sanoittaa mitään. Huomioni kiinnittyy kehon reaktioihin: hengitykseni pingottuu, rintaani leviää puristava tunne, käteni kylmenevät. Mieleni on jo reagoinut näihin tuntemuksiin ennen kuin kehoni on täysin tietoinen niistä. Kehollinen rauhattomuus ikään kuin käynnistää ketjun: se laukaisee suojautumisen tai sopeutumisen tarpeen. Ajatukseni alkavat etsiä ratkaisuja, valmistautua mahdolliseen konfliktiin, rakentaa mielikuviteltuja skenaarioita. Kehoni ja mieleni välillä syntyy sykli, jossa kumpikin ruokkii toistaan: pieni fyysinen merkki kasvaa kokonaiseksi kokemukseksi levottomuudesta."""
ENGLISH_STORY = """Most of the time, I notice a lack of peace in the body before the mind can say anything. My attention is focused on the body’s reactions: my breathing is straining, my chest is spreading a tight feeling, my hands are getting colder. My mind has already responded to these feelings before my body is fully aware of them. Body restlessness, as it were, triggers a chain: it triggers the need for protection or adjustment. My thoughts begin to look for solutions, prepare for a possible conflict, build imaginary scenarios. There is a cycle between my body and my mind in which each feeds one another: a small physical sign grows into an entire experience of anxiety."""
TURKISH_STORY = """Genellikle zihnim bir şey söylemeden önce vücudumda huzursuzluk fark ederim. Dikkatim vücudun tepkilerine odaklanır: nefesim gerilir, göğsümde sıkışan bir his yayılır, ellerim soğur. Zihnim, vücudum tam olarak farkına varmadan önce bu duygulara tepki vermiştir bile. Vücut huzursuzluğu, bir zincirleme reaksiyon başlatır: koruma veya uyum sağlama ihtiyacını tetikler. Düşüncelerim çözümler aramaya başlar, olası bir çatışmaya hazırlanır, hayali senaryolar oluşturur. Vücudum ve zihnim arasında birbirini besleyen bir döngü oluşur: küçük bir fiziksel işaret, tüm bir kaygı deneyimine dönüşür."""
SPANISH_STORY = """La mayoría de las veces, noto una falta de paz en el cuerpo antes de que la mente pueda decir algo. Mi atención se centra en las reacciones del cuerpo: mi respiración se tensa, mi pecho se extiende una sensación de opresión, mis manos se enfrían. Mi mente ya ha respondido a estos sentimientos antes de que mi cuerpo sea plenamente consciente de ellos. La inquietud corporal, por así decirlo, desencadena una cadena: desencadena la necesidad de protección o ajuste. Mis pensamientos comienzan a buscar soluciones, prepararse para un posible conflicto, construir escenarios imaginarios. Hay un ciclo entre mi cuerpo y mi mente en el que cada uno alimenta al otro: una pequeña señal física crece hasta convertirse en toda una experiencia de ansiedad."""
FRENCH_STORY = """La plupart du temps, je remarque un manque de paix dans le corps avant que l'esprit ne puisse dire quoi que ce soit. Mon attention se concentre sur les réactions du corps : ma respiration se tend, ma poitrine diffuse une sensation de pression, mes mains deviennent plus froides. Mon esprit a déjà réagi à ces sentiments avant que mon corps en soit pleinement conscient. L'agitation corporelle, pour ainsi dire, déclenche une chaîne : elle déclenche le besoin de protection ou d'ajustement. Mes pensées commencent à chercher des solutions, à se préparer à un conflit possible, à construire des scénarios imaginaires. Il y a un cycle entre mon corps et mon esprit dans lequel chacun nourrit l'autre : un petit signe physique se transforme en toute une expérience d'anxiété."""
import struct

def float_wav_to_pcm(input_path, output_path):
    with wave.open(input_path, 'rb') as wf_in:
        params = wf_in.getparams()
        if params.sampwidth != 4 or params.comptype != 'NONE':
            raise ValueError("Input file is not 32-bit float PCM WAV")
        frames = wf_in.readframes(params.nframes)
        # Convert float32 to int16
        float_samples = struct.unpack('<' + 'f' * params.nframes * params.nchannels, frames)
        int_samples = [max(-32768, min(32767, int(sample * 32767))) for sample in float_samples]
        int_frames = struct.pack('<' + 'h' * len(int_samples), *int_samples)
        with wave.open(output_path, 'wb') as wf_out:
            wf_out.setnchannels(params.nchannels)
            wf_out.setsampwidth(2)  # 16-bit
            wf_out.setframerate(params.framerate)
            wf_out.writeframes(int_frames)

def call_infer_api(language, ref_text, gen_text, audio_path):
    data = {
            "language": language,
            "ref_text": ref_text,
            "gen_text": gen_text,
            "speed": 1.0
        }
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

def split_story_to_sentences(story):
    import re
    # Simple sentence splitter for Swedish (handles . ! ?)
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', story) if s.strip()]

def combine_wav_files(wav_files, output_path):
    with wave.open(wav_files[0], 'rb') as wf:
        params = wf.getparams()
        frames = [wf.readframes(wf.getnframes())]
    for wav_file in wav_files[1:]:
        with wave.open(wav_file, 'rb') as wf:
            if wf.getparams() != params:
                raise ValueError("WAV files have different parameters!")
            frames.append(wf.readframes(wf.getnframes()))
    with wave.open(output_path, 'wb') as wf:
        wf.setparams(params)
        for f in frames:
            wf.writeframes(f)

def tts_story_to_wav(story, language, ref_text, output_path, audio_path=None):
    sentences = split_story_to_sentences(story)
    temp_files = []
    for idx, sentence in enumerate(sentences):
        temp_wav = f"temp_sentence_{idx}.wav"
        data = {
            "language": language,
            "ref_text": ref_text,
            "gen_text": sentence,
            "speed": 0.5
        }
        if audio_path is not None:
            with open(audio_path, "rb") as audio_file:
                files = {"audio_file": ("input.wav", audio_file, "audio/wav")}
                response = requests.post(API_URL, data=data, files=files)
        else:
            response = requests.post(API_URL, data=data)
        if response.status_code == 200:
            with open(temp_wav, "wb") as out_f:
                out_f.write(response.content)
            temp_files.append(temp_wav)
        else:
            print(f"API call failed for sentence {idx}: {sentence}")
            print("Error:", response.json())
            for f in temp_files:
                os.remove(f)
            return
    print("Individual sentence files are saved as temp_sentence_*.wav")
    # Optionally, you can still try to combine them if you want, but skip PCM conversion
    # combine_wav_files(temp_files, output_path)
    # for f in temp_files:
    #     os.remove(f)
    # print(f"Combined audio saved as {output_path}")

# Example usage:
if __name__ == "__main__":
    tts_story_to_wav(
        story=FRENCH_STORY,
        language="Multilingual",
        ref_text="One of the best ways to involve yourself with both writing and magic is to create a magic journal.",
        output_path="story_output.wav",
        audio_path="One of the best ways to involve yourself with both writing and magic is to create a magic journal.wav"
    )
