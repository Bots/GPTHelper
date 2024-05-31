import asyncio
from dotenv import load_dotenv
import os
import assemblyai as aai
from eff_word_net.streams import SimpleMicStream
from eff_word_net.engine import HotwordDetector
from eff_word_net.audio_processing import Resnet50_Arc_loss
from eff_word_net import samples_loc
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from openai import AsyncOpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize services clients
elevenlabsClient = ElevenLabs()
openaiClient = AsyncOpenAI(api_key = os.getenv('OPENAI_API_KEY'))
aai.settings.api_key = os.getenv('ASSEMBLY_API_KEY')

base_model = Resnet50_Arc_loss()

computer_hw = HotwordDetector(
    hotword = "computer",
    model = base_model,
    reference_file=os.path.join(samples_loc, "computer_ref.json"),
    threshold = 0.7,
    relaxation_time = 2,
)

mic_stream = SimpleMicStream(
    window_length_secs = 1.5,
    sliding_window_secs = 0.75,    
)

def on_open(session_opened: aai.RealtimeSessionOpened):
    print("Session ID: ", session_opened.session_id)

def on_data(transcript: aai.RealtimeTranscript):
    if not transcript.text:
        return

    if isinstance(transcript, aai.RealtimeFinalTranscript):
        print(transcript.text, end = "\r\n")
    else:
        print(transcript.text, end = "\r")

def on_error(error: aai.RealtimeError):
    print("An error occurred: ", error)

def on_close():
    print("Closing session")

mic_stream.start_stream()

while True:
    frame = mic_stream.getFrame()
    result = computer_hw.scoreFrame(frame)
    if result == None:
        continue
    if(result["match"]):
        print("Wakeword uttered", result["confidence"])

        transcriber = aai.RealtimeTranscriber(
            sample_rate = 16_000,
            on_data = on_data,
            on_error = on_error,
            on_open = on_open,
            on_close = on_close,
        )

        transcriber.connect()

        transcriber.stream(mic_stream.getFrame())

        transcriber.close()