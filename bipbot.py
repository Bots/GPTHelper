import numpy as np
import pyaudio
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

# Initialize audio stream
FRAMES_PER_BUFFER = 24000
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = 24000
WINDOW_LENGTH_SECS = 1.5
SLIDING_WINDOW_SECS:float = 3/4
p = pyaudio.PyAudio()

# Initialize services clients
elevenlabsClient = ElevenLabs()
openaiClient = AsyncOpenAI(api_key = os.getenv('OPENAI_API_KEY'))
aai.settings.api_key = os.getenv('ASSEMBLY_API_KEY')

base_model = Resnet50_Arc_loss()

out_audio = np.zeros(16000) #blank 1 sec audio
print("Initial S",out_audio.shape)

computer_hw = HotwordDetector(
    hotword = "computer",
    model = base_model,
    reference_file=os.path.join(samples_loc, "computer_ref.json"),
    threshold = 0.7,
    relaxation_time = 2,
)

def transcriber_on_open(session_opened: aai.RealtimeSessionOpened):
    print("Session ID: ", session_opened.session_id)

def transcriber_on_data(transcript: aai.RealtimeTranscript):
    if not transcript.text:
        return

    if isinstance(transcript, aai.RealtimeFinalTranscript):
        print(transcript.text, end = "\r\n")
    else:
        print(transcript.text, end = "\r")

def transcriber_on_error(error: aai.RealtimeError):
    print("An error occurred: ", error)

def transcriber_on_close():
    print("Closing session")

transcriber = aai.RealtimeTranscriber(
    sample_rate = 16000,
    on_data = transcriber_on_data,
    on_error = transcriber_on_error,
    on_open = transcriber_on_open,
    on_close = transcriber_on_close,
)

stream = p.open(
    format = FORMAT,
    channels = CHANNELS,
    rate = SAMPLE_RATE,
    input = True,
    frames_per_buffer = FRAMES_PER_BUFFER
)

while True:
    frame = (
        np.frombuffer(stream.read(CHUNK, exception_on_overflow = False), dtype=np.int16)
    )
    out_audio = np.append(
        out_audio[15999],
        frame
    )
    print(out_audio.shape)
    result = computer_hw.scoreFrame(inp_audio_frame = frame)
    if result == None:
        continue
    if(result["match"]):
        print("Wakeword uttered", result["confidence"])

        transcriber.connect()

        transcriber.stream(stream)

        transcriber.close()