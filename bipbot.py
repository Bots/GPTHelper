import threading
from dotenv import load_dotenv
import os
import assemblyai as aai
from assemblyai.types import RealtimeMessageTypes
from elevenlabs import play
from elevenlabs.client import ElevenLabs
from openai import AsyncOpenAI
from eff_word_net import samples_loc
from eff_word_net.streams import SimpleMicStream
from eff_word_net.engine import HotwordDetector
from eff_word_net.audio_processing import Resnet50_Arc_loss

# Load environment variables from .env file
load_dotenv()

# Initialize audio vars
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)

# Initialize services clients
elevenlabsClient = ElevenLabs()
openaiClient = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
aai.settings.api_key = os.getenv('ASSEMBLY_API_KEY')
aai.settings.polling_interval = 1.0

base_model = Resnet50_Arc_loss()

hotword_listening = True
transcribing = False
state_lock = threading.Lock()

mic_stream = SimpleMicStream(
    window_length_secs=1.5,
    sliding_window_secs=3/4,
)

def on_open(session_opened: aai.RealtimeSessionOpened):
    print("Session ID:", session_opened.session_id)

def on_data(transcript: aai.RealtimeTranscript):
    global hotword_listening, transcribing, state_lock
    if not transcript.text:
        return
    print(transcript)
    if isinstance(transcript, aai.RealtimeFinalTranscript):
        transcriber.close()
        print(transcript.text, end="\r\n")
        with state_lock:
            transcribing = False
            hotword_listening = True
    else:
        print(transcript.text, end="\r")

def on_error(error: aai.RealtimeError):
    global hotword_listening, state_lock
    print("An error occured:", error)
    with state_lock:
        hotword_listening = True

def on_close():
    print("Closing Query Session")

computer_hw = HotwordDetector(
    hotword="computer",
    model=base_model,
    reference_file=os.path.join(samples_loc, "computer_ref.json"),
    threshold=0.6,
    relaxation_time=1,
)

transcriber = aai.RealtimeTranscriber(
    sample_rate=SAMPLE_RATE,
    on_data=on_data,
    on_error=on_error,
    on_open=on_open,
    on_close=on_close,
    on_extra_session_information=None,
)

def transcribe():
    global hotword_listening, transcribing, state_lock
    with state_lock:
        if transcribing:
            return
        transcribing = True
        hotword_listening = False
    try:
        # Start the connection
        transcriber.connect()
        print("Transcribing...")
        transcriber.stream(aai.extras.MicrophoneStream(sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE))
        print("Transcription complete")
    finally:
        transcriber.close()
        with state_lock:
            transcribing = False
            hotword_listening = True

def detect_hotword():
    global hotword_listening, transcribing, state_lock
    while True:
        with state_lock:
            if not hotword_listening:
                continue
        mic_stream._open_stream()
        frame = mic_stream.getFrame()
        result = computer_hw.scoreFrame(frame)
        if result is None:
            continue
        if result["match"]:
            print("Wakeword uttered", result["confidence"])
            mic_stream.close_stream()
            with state_lock:
                hotword_listening = False
                if not transcribing:
                    transcribe_thread = threading.Thread(target=transcribe)
                    transcribe_thread.start()

# Start hotword detection in a separate thread
hotword_thread = threading.Thread(target=detect_hotword)
hotword_thread.start()