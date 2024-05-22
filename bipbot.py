import os
import threading
import assemblyai as aai
from eff_word_net.streams import SimpleMicStream
from eff_word_net.engine import HotwordDetector
from eff_word_net.audio_processing import Resnet50_Arc_loss
from eff_word_net import samples_loc
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs import play

# Initialize ElevenLabs and OpenAI clients
elevenlabsClient = ElevenLabs()  # Add your ElevenLabs API key here
openaiClient = OpenAI()  # Add your OpenAI API key here

# Initialize the base model for the hotword detector
base_model = Resnet50_Arc_loss()

# Set up the hotword detector with the specified parameters
computer_hw = HotwordDetector(
    hotword="computer",
    model=base_model,
    reference_file=os.path.join(samples_loc, "computer_ref.json"),
    threshold=0.7,
    relaxation_time=1
)

# Set up the microphone stream with the specified window lengths
mic_stream = SimpleMicStream(
    window_length_secs=1.5,
    sliding_window_secs=0.75
)

# Variable to store the latest transcript
latest_transcript = ""
silence_timer = None
silence_duration = 1  # Duration in seconds to wait for silence before calling OpenAI API

# Flag to track the state of the transcriber
transcriber_active = False

# Function to handle silence detection
def handle_silence():
    global latest_transcript, transcriber_active
    transcriber.close()  # Close the transcriber to stop receiving data
    transcriber_active = False  # Update the flag

    response = openaiClient.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": latest_transcript}
        ]
    )

    # Print and play the response
    response_text = response.choices[0].message.content
    print(response_text)
    audio = elevenlabsClient.generate(
        text=response_text,
        voice="Rachel",
        model="eleven_multilingual_v2"
    )
    play(audio)

    # Restart the microphone stream
    mic_stream.start_stream()

# Define the callback functions for the transcriber
def on_open(session_opened: aai.RealtimeSessionOpened):
    print("Session ID: ", session_opened.session_id)

def on_data(transcript: aai.RealtimeTranscript):
    global latest_transcript, silence_timer
    
    if not transcript.text:
        return

    if isinstance(transcript, aai.RealtimeTranscript):
        latest_transcript = transcript.text
        print(latest_transcript, end="\r\n")
        
        # Reset the silence timer
        if silence_timer:
            silence_timer.cancel()
        
        # Start a new silence timer
        silence_timer = threading.Timer(silence_duration, handle_silence)
        silence_timer.start()

def on_error(error: aai.RealtimeError):
    print(error)

# Initialize the transcriber with the API key and callbacks
transcriber = aai.transcriber.RealtimeTranscriber(
    on_open=on_open,
    on_data=on_data,
    on_error=on_error,
    sample_rate=44_100
)

def start_hotword_detection():
    global transcriber_active

    # Ensure the microphone stream is started
    if not mic_stream.is_streaming():
        mic_stream.start_stream()

    while True:
        frame = mic_stream.getFrame()
        result = computer_hw.scoreFrame(frame)
        if result is None:
            # No voice activity
            continue
        if result["match"]:
            print("Wakeword uttered", result["confidence"])
            transcriber.connect()
            microphone_stream = aai.extras.MicrophoneStream()
            transcriber.stream(microphone_stream)
            transcriber_active = True  # Update the flag

            # Wait for the transcription to complete and handle silence
            while transcriber_active:
                pass
            
            # Break out of the loop to handle the response and restart the detection loop
            break

# Main loop to manage the hotword detection and response handling
while True:
    start_hotword_detection()
    # After handling the response, restart the detection loop
    mic_stream.stop_stream()
    mic_stream.start_stream()
