from dotenv import load_dotenv
import os
import numpy as np
import assemblyai as aai
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
CHUNK = 3200

mic_stream = SimpleMicStream(
    window_length_secs=1.5,
    sliding_window_secs=3/4,
)

# Initialize services clients
elevenlabsClient = ElevenLabs()
openaiClient = AsyncOpenAI(api_key = os.getenv('OPENAI_API_KEY'))
aai.settings.api_key = os.getenv('ASSEMBLY_API_KEY')
aai.settings.polling_interval = 1.0

base_model = Resnet50_Arc_loss()

hotword_listening = True

audio = np.zeros(16000) #blank 1 sec audio

def on_open(session_opened: aai.RealtimeSessionOpened):
    print("Session ID:", session_opened.session_id)

def on_data(transcript: aai.RealtimeTranscript):
    global hotword_listening
    if not transcript.text:
        return

    if isinstance(transcript, aai.RealtimeFinalTranscript):
        print(transcript.text, end="\r\n")
        transcriber.close()
    else:
        print(transcript.text, end="\r")

def on_error(error: aai.RealtimeError):
    global hotword_listening
    print("An error occured:", error)
    hotword_listening = True

def on_close():
    global hotword_listening
    print("Closing Query Session")
    hotword_listening = True

computer_hw = HotwordDetector(
    hotword = "computer",
    model = base_model,
    reference_file=os.path.join(samples_loc, "computer_ref.json"),
    threshold = 0.7,
    relaxation_time = 1,
    continuous = False,
)

transcriber = aai.RealtimeTranscriber(
  sample_rate = SAMPLE_RATE,
  on_data = on_data,
  on_error = on_error,
  on_open = on_open, # optional
  on_close = on_close, # optional
  end_utterance_silence_threshold=500,
)

def transcribe():
    # Start the connection
    transcriber.connect()
    print("Transcribing...")
    transcriber.stream(aai.extras.MicrophoneStream(sample_rate = SAMPLE_RATE))
    print("Transcription complete")
    transcriber.close()

# mic_stream.start_stream()

while True & hotword_listening:
    mic_stream.start_stream()
    frame = mic_stream.getFrame()
    result = computer_hw.scoreFrame(frame)
    if result == None:
        continue
    if(result["match"]):
        print("Wakeword uttered", result["confidence"])
        mic_stream.close_stream()
        hotword_listening = False
        while not hotword_listening:
            transcribe()
        
