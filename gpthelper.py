import os
import torch
import threading
from dotenv import load_dotenv
import assemblyai as aai
import numpy as np
import sounddevice as sd
from datetime import datetime
from queue import Queue, Empty
from openai import OpenAI
from piper.voice import PiperVoice
from colorlog import ColoredFormatter
from eff_word_net.streams import SimpleMicStream
from eff_word_net.engine import HotwordDetector
from eff_word_net.audio_processing import Resnet50_Arc_loss
import logging

# Configure colorized logging configuration
formatter = ColoredFormatter(
    "%(log_color)s%(asctime)s [%(levelname)s] %(message)s%(reset)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)

# Create a StreamHandler and set the formatter
handler = logging.StreamHandler()
handler.setFormatter(formatter)

# Get the root logger and add the handler to it
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Check if CUDA is available for PyTorch and log the status
logger.info(f"CUDA enabled: {torch.cuda.is_available()}")

# Load environment variables from the .env file for API access
load_dotenv()

# Set the AssemblyAI API key from environment variables
aai.settings.api_key = os.getenv('ASSEMBLY_API_KEY')
aai.settings.polling_interval = 1.0  # Interval in seconds for polling transcription status

# Initialize the OpenAI client
client = OpenAI()

# Global variables to keep track of the last time we received a transcript and if the program should terminate
last_transcript_received = datetime.now()
base_model = Resnet50_Arc_loss()
transcript_queue = Queue()
terminated = False

# Listen for a specific hotword using an audio stream
def listen_for_hotword(hotword_detector: HotwordDetector, hotword_mic_stream: SimpleMicStream):
    
    # Start the microphone stream for hotword detection
    hotword_mic_stream.start_stream()
    logger.info("Listening for hotword...")

    try:
        while True:
            # Capture a single frame from the audio stream
            frame = hotword_mic_stream.getFrame()
            
            if frame is not None: 
                logger.debug(f"Frame captured with shape: {frame.shape}")
                
            # Send the captured frame to the hotword detector and log the result
            result = hotword_detector.scoreFrame(frame)
            logger.debug(f"Hotword detection result: {result}")
            
            if result is not None and result['match']:
                logger.info(f"Hotword detected: {result}")
                return True  # Hotword detected, proceed with speech recognition
            elif frame is None:
                logger.warning("No frame captured; microphone might not be working.")
    finally:
        # Always ensure the stream is closed when done
        hotword_mic_stream.close_stream()

# Function to handle the transcription process in a separate thread
def handle_transcription(transcriber: aai.RealtimeTranscriber, transcribe_event: threading.Event, stop_transcription_event: threading.Event, transcript_queue: Queue):
    logger.info("Transcription handler thread started.")
    
    # Placeholder for MicrophoneStream instance to be used in the try block below
    transcriber_mic_stream = None
    
    while not stop_transcription_event.is_set():  # Run until stop signal is received
        transcribe_event.wait()  # Wait for the event that signals transcription should start
        transcribe_event.clear()  # Clear the event after it has been set
        logger.info("Starting transcription process...")
        
        try:
            # Initialize the MicrophoneStream for AssemblyAI transcription
            transcriber_mic_stream = aai.extras.MicrophoneStream(sample_rate=16000)
            logger.info("Microphone stream set up for transcription.")
            
            # Start streaming audio from the microphone to the transcriber
            transcriber.stream(transcriber_mic_stream)
            logger.info("Streaming audio to AssemblyAI's transcriber.")
            
            # Loop until a stop signal is received
            while not stop_transcription_event.is_set():
                try:
                    # Retrieve a transcript from the queue with a timeout of 1 second
                    transcript = transcript_queue.get(timeout=1)
                    if transcript:
                        logger.info(f"Final transcript received: {transcript}")
                        break
                except Empty:
                    # If the queue is empty, continue checking until a transcript is available
                    continue
        except Exception as e:
            logger.error(f"An error occurred during the transcription process: {e}")
        finally:
            # Clean up by closing the microphone stream and transcriber connection
            if transcriber_mic_stream:
                transcriber_mic_stream.close()
            transcriber.close()
    
    logger.info("Transcription handler thread stopped.")

# Callback used to handle real-time transcript data received from AssemblyAI
def on_data(transcript: aai.RealtimeTranscript):
    logger.debug("Connection established")
    global last_transcript_received
    global terminated
    
    # Log the processed transcript data and handle termination if needed
    if terminated:
        return
    
    # Handle a scenario where there's no transcript data (i.e., silence)
    if transcript.text == "":
        delay_since_last = (datetime.now() - last_transcript_received).total_seconds()
        logger.debug(f"Silence detected for {delay_since_last}s")
        
        if delay_since_last > 10:
            logger.info("Extended silence detected; stopping transcription.")
            terminate_transcription()
        
        return
    
    # Log and save the final transcript when received
    if isinstance(transcript, aai.RealtimeFinalTranscript):
        logger.info(f"User said: {transcript.text}")
        transcript_queue.put(transcript.text, False)
    else:
        # For partial transcripts, just display them without logger for brevity
        print(transcript.text, end="\r")
    
    # Record the time of the last received transcript
    last_transcript_received = datetime.now()

# Callback used to handle errors during real-time transcription
def on_error(error: aai.RealtimeError):
    logger.error(f"An error occurred during real-time transcription: {error}")

# Callback for when the transcription session is closed
def on_close():
    global terminated
    if not terminated:
        logger.info("Transcription session closed.")
        terminated = True

# Function to gracefully terminate the transcription process
def terminate_transcription():
    global terminated
    if not terminated:
        logger.info("Terminating transcription...")
        transcriber.close()
        terminated = True

# Function to send a transcript to ChatGPT for processing and obtain a response
def process_with_chatgpt(transcript_result):
    start_time = datetime.now()
    logger.info("Sending user transcript to ChatGPT...")
    
    # Use the OpenAI client to interact with ChatGPT using the provided transcript
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": 'You are a highly skilled AI. Keep your answers short and simple'},
            {"role": "user", "content": transcript_result}
        ],
        model='gpt-4o',
    )
    
    response_duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"ChatGPT responded in {response_duration:.2f} seconds.")
    
    # Extract and log the response content
    final_response = response.choices[0].message.content
    logger.info(f"ChatGPT's response: {final_response}")
    return final_response

# Function to convert the ChatGPT response text to speech and play it
def text_to_speech(text):
    start_time = datetime.now()
    logger.info("Converting ChatGPT response to speech...")
    
    # Directory where the TTS model files are stored
    voicedir = "./voices/"
    tts_model_path = os.path.join(voicedir, "en_US-arctic-medium.onnx")
    voice = PiperVoice.load(tts_model_path, use_cuda=True)
    
    # Set up the output audio stream for speech playback
    stream = sd.OutputStream(samplerate=24_000, channels=1, dtype="int16")
    stream.start()
    
    # Synthesize and play the speech audio
    for audio_bytes in voice.synthesize_stream_raw(text):
        int_data = np.frombuffer(audio_bytes, dtype=np.int16)
        stream.write(int_data)
    
    # Close the audio stream once speech playback is complete
    stream.stop()
    stream.close()
    
    stream_duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"Finished speech playback in {stream_duration:.2f} seconds.")

# Initialize the main service components for transcription and hotword detection
transcriber = aai.RealtimeTranscriber(
    on_data=on_data, on_error=on_error, on_close=on_close, sample_rate=16000
)
hot_word_detector = HotwordDetector(
    model=base_model, 
    hotword="jippety", 
    relaxation_time=1,
    verbose=True, 
    threshold=0.6, 
    continuous=False, 
    reference_file="voices/jippety_ref.json"
)
transcriber.connect()

# Initialize threading events to control transcription
transcribe_event = threading.Event()
stop_transcription_event = threading.Event()

# Create and start a thread for handling transcription in the background
transcription_thread = threading.Thread(
    target=handle_transcription,
    args=(transcriber, transcribe_event, stop_transcription_event, transcript_queue)
)
transcription_thread.start()

# Main loop where the application waits for the hotword and processes speech
def main():
    # Create an instance for hotword audio streaming
    hotword_mic_stream = SimpleMicStream(window_length_secs=1.5, sliding_window_secs=0.5)
    
    while True:
        if listen_for_hotword(hot_word_detector, hotword_mic_stream):
            # Trigger the event to start transcription when the hotword is detected
            transcribe_event.set()
            
            try:
                # Wait for the transcription to be available in the queue
                transcript = transcript_queue.get(timeout=60)
                logger.info(f"Received transcript for processing: {transcript}")
            except Empty:
                logger.warning("No transcript received within the timeout period.")
                continue
            
            # Signal the transcription thread to stop transcribing
            stop_transcription_event.set()
            
            # Process the transcript using ChatGPT
            chatgpt_response = process_with_chatgpt(transcript)
            
            # Use TTS to speak out the ChatGPT response
            text_to_speech(chatgpt_response)
            
            # Prepare for the next cycle by resetting the event flags
            stop_transcription_event.clear()

# Run the main loop if this script is executed as the 'main' module.
if __name__ == "__main__":
    try:
        main()
    finally:
        # Signal and wait for the transcription thread to finish before exiting
        stop_transcription_event.set()
        transcription_thread.join()
        logger.info("Application exited gracefully.")