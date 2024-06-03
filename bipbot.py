import os
import threading
import subprocess
from dotenv import load_dotenv
import assemblyai as aai
from openai import OpenAI
from eff_word_net import samples_loc
from eff_word_net.streams import SimpleMicStream
from eff_word_net.engine import HotwordDetector
from eff_word_net.audio_processing import Resnet50_Arc_loss

# Load environment variables from .env file
load_dotenv()

# Initialize audio and service clients
SAMPLE_RATE = 16000
client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")
aai.settings.api_key = os.getenv("ASSEMBLY_API_KEY")
aai.settings.polling_interval = 1.0
base_model = Resnet50_Arc_loss()
mic_stream = SimpleMicStream(window_length_secs=1.5, sliding_window_secs=3 / 4)

computer_hw = HotwordDetector(
    hotword="computer",
    model=base_model,
    reference_file=os.path.join(samples_loc, "computer_ref.json"),
    threshold=0.6,
    relaxation_time=1,
)

# State variables
hotword_listening = True
transcribing = False
state_lock = threading.Lock()
transcription = ""


def on_open(session_opened: aai.RealtimeSessionOpened):
    print("Session ID:", session_opened.session_id)


def on_data(transcript: aai.RealtimeTranscript):
    global transcription
    if transcript.text:
        if isinstance(transcript, aai.RealtimeFinalTranscript):
            print(transcript.text)
            transcription = transcript.text
            transcriber.close()
        else:
            print(transcript.text, end="\r")


def on_error(error: aai.RealtimeError):
    global hotword_listening, state_lock
    print("An error occurred:", error)
    with state_lock:
        hotword_listening = True


def on_close():
    global hotword_listening, transcribing, state_lock, transcription
    if transcribing and not hotword_listening:
        print("Querying ChatGPT")
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a friendly assistant, skilled in explaining complex questions with simplicity. Keep your answers short and simple. Don't use slashes or parentheses",
                    },
                    {"role": "user", "content": transcription},
                ],
            )
            answer = completion.choices[0].message.content
            print(answer)
            subprocess.run(
                f"echo {answer} | piper --cuda --model en_US-hfc_male-medium.onnx --output-raw | paplay -d=default -p --raw --rate=16000",
                text=True,
                shell=True,
            )
        except Exception as e:
            print(e)
        with state_lock:
            transcribing = False
            hotword_listening = True


def on_extra_session_information(session_info: aai.RealtimeSessionInformation):
    print("Session information:", session_info)


transcriber = aai.RealtimeTranscriber(
    sample_rate=SAMPLE_RATE,
    on_data=on_data,
    on_error=on_error,
    on_open=on_open,
    on_close=on_close,
    on_extra_session_information=on_extra_session_information,
)


def transcribe():
    global hotword_listening, transcribing, state_lock, transcriber
    with state_lock:
        if transcribing:
            return
        transcribing = True
        hotword_listening = False
    print("Transcribing...")
    transcriber.connect()
    transcriber.stream(aai.extras.MicrophoneStream(sample_rate=SAMPLE_RATE))
    with state_lock:
        transcribing = False
        hotword_listening = True
    transcriber.close()


def detect_hotword():
    global hotword_listening, transcribing, state_lock
    while True:
        with state_lock:
            if not hotword_listening or transcribing:
                continue
        mic_stream._open_stream()
        frame = mic_stream.getFrame()
        result = computer_hw.scoreFrame(frame)
        if result is None or not result["match"]:
            continue
        print("Wakeword uttered", result["confidence"])
        mic_stream.close_stream()
        with state_lock:
            if not transcribing:
                transcribe()


# Start hotword detection in a separate thread
hotword_thread = threading.Thread(target=detect_hotword)
hotword_thread.start()
