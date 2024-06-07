from dotenv import load_dotenv
import os
import torch
import threading
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

# Load environment variables from the .env file for API access
load_dotenv()


# Define a start time so that we can determine latency throughout
start_time = datetime.now()
print("APP START", start_time)


# Configure the color logger
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

# More loggery and haberdashery
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Check if CUDA is available for PyTorch and log the status
logger.info(f"CUDA enabled: {torch.cuda.is_available()}")

# Set the AssemblyAI API key from environment variables
aai.settings.api_key = os.getenv('ASSEMBLY_API_KEY')
# Interval in seconds for polling transcription status
# aai.settings.polling_interval = 1.0

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Global variables to keep track of the last time we received a transcript and if the program should terminate
base_model = Resnet50_Arc_loss()
transcript_queue = Queue()
terminated = False


def listen_for_hotword(hotword_detector: HotwordDetector, hotword_mic_stream: SimpleMicStream):
    """
    Listen for a specific hotword using an audio stream with a 'sliding window.'
    """

    # Start the microphone stream for hotword detection
    hotwordMetrics1 = datetime.now()
    logger.error("Hotword detection start", hotwordMetrics1)
    logger.critical({hotwordMetrics1 - start_time})

    try:
        while True:
            # Capture a single frame from the audio stream
            frame = hotword_mic_stream.getFrame()

            # Log shape of the array captured from the audio stream (debug)
            if frame is not None:
                logger.debug(f"Frame captured with shape: {frame.shape}")

            # Send the captured frame to the hotword detector for analysis
            result = hotword_detector.scoreFrame(frame)
            logger.debug(f"Hotword detection result: {result}")

            # Check if the hotword was detected and return True if so
            if result is not None and result['match']:
                logger.info(f"Hotword detected: {result}")
                return True  # Hotword detected, proceed with speech recognition
            elif frame is None:
                logger.warning(
                    "No frame captured; Check sound settings")
    finally:
        # Ensure the stream is closed
        hotword_mic_stream.close()
        hotwordMetrics2 = datetime.now()
        logger.error("Hotword detection end", hotwordMetrics2)
        logger.critical({hotwordMetrics2 - start_time})


def handle_transcription(transcriber: aai.RealtimeTranscriber, transcribe_event: threading.Event, stop_transcription_event: threading.Event, transcript_queue: Queue):
    """
    Handle the transcription process in a separate thread.
    """
    # Start transcription
    transcriptionMetrics1 = datetime.now()
    logger.error("Transcription start", transcriptionMetrics1)
    logger.critical({transcriptionMetrics1 - start_time})

    # Placeholder for MicrophoneStream instance to be used in the try block below
    transcriber_mic_stream = None

    while not stop_transcription_event.is_set():  # Run until stop signal is received
        transcribe_event.wait()  # Wait for the event that signals transcription should start
        transcribe_event.clear()  # Clear the event after it has been set

        try:
            # Initialize the MicrophoneStream for AssemblyAI transcription
            transcriber_mic_stream = aai.extras.MicrophoneStream(
                sample_rate=16000)
            logger.info("Microphone open.")

            # Start streaming audio from the microphone to the transcriber
            transcriber.stream(transcriber_mic_stream)
            logger.info("Streaming audio to AssemblyAI's transcriber.")

            # Loop until a stop signal is received
            while not stop_transcription_event.is_set():
                try:
                    # Retrieve a transcript from the queue with a timeout of 1 second
                    transcript = transcript_queue.get(timeout=2)
                    if transcript:
                        logger.info(f"Final transcript received: {transcript}")
                        break
                except Empty:
                    # If the queue is empty, continue checking until a transcript is available
                    continue
        except Exception as e:
            logger.error(
                f"An error occurred during the transcription process: {e}")
        finally:
            # Clean up by closing the microphone stream and transcriber connection
            if transcriber_mic_stream:
                transcriber_mic_stream.close()
            transcriber.close()
            # Stop transcription
            transcriptionMetrics2 = datetime.now()
            logger.error("Transcription end", transcriptionMetrics2)
            logger.critical({transcriptionMetrics2 - start_time})

    logger.info("Transcription thread stopped.")


def on_data(transcript: aai.RealtimeTranscript):
    """
    Callback used to handle real-time transcript data received from AssemblyAI.
    """

    onDataMetric1 = datetime.now()
    logger.error("onData start", onDataMetric1)
    logger.critical(onDataMetric1 - start_time)
    global last_transcript_received
    global terminated

    # Log the processed transcript data and handle termination if needed
    if terminated:
        return

    # Handle silence
    if transcript.text == "":
        print("Silence detected", end="\r")
        if (datetime.now() - start_time) > 5:
            logger.info("5 seconds silence detected; stopping transcription.")
            terminate_transcription()
        return

    # Log and save the final transcript when received
    if isinstance(transcript, aai.RealtimeFinalTranscript):
        logger.info(f"User said: {transcript.text}")
        transcript_queue.put(transcript.text, False)
    else:
        # For partial transcripts, just display them without logger for brevity
        print(transcript.text, end="\r")

    # Start transcription
    onDataMetric2 = datetime.now()
    logger.error("onData end", onDataMetric2)
    logger.critical({onDataMetric2 - start_time})


def on_error(error: aai.RealtimeError):
    """
    Callback used to handle errors during real-time transcription. Connection is closed.
    """

    logger.error(f"An error occurred during real-time transcription: {error}")


def on_close():
    """
    Callback for when the transcription session is closed without errors.
    """

    connectionClosedMetric = datetime.now()
    logger.critical(connectionClosedMetric)
    global terminated
    if not terminated:
        logger.info("Transcription session closed.")
        terminated = True


def terminate_transcription():
    """
    Gracefully terminate the transcription process.
    """

    terminateMetric1 = datetime.now()
    global terminated, transcriber
    if not terminated:
        logger.info("Terminating transcription...")
        transcriber.close()
        terminated = True
        terminateMetric2 = datetime.now()
        logger.critical(terminateMetric2)


def process_with_chatgpt(transcript_result: str):
    """
    Send a transcript to ChatGPT for processing and obtain a response.
    """

    gptMetric1 = datetime.now()
    logger.error("GPT start", gptMetric1)
    logger.critical({gptMetric1 - start_time})

    # Use the OpenAI client to interact with ChatGPT using the provided transcript
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": 'You are a highly skilled AI. Keep your answers short and simple'},
            {"role": "user", "content": transcript_result}
        ],
        model='gpt-4o',
    )
    transcript_result = ""
    gptMetric2 = datetime.now()
    logger.error("Gpt transcription end", gptMetric2)
    logger.critical({gptMetric2 - start_time})

    # Extract and log the response content
    final_response = response.choices[0].message.content
    logger.info(f"ChatGPT's response: {final_response}")
    gptMetric3 = datetime.now()
    logger.error("GPT end", gptMetric3)
    logger.critical({gptMetric3 - start_time})
    return final_response


def text_to_speech(text):
    """
    Convert the ChatGPT response text to speech and play it.
    """

    ttsMetric1 = datetime.now()
    logger.error("TTS start", ttsMetric1)
    logger.critical({ttsMetric1 - start_time})

    # Directory where the TTS model files are stored
    voice = PiperVoice.load("voices/en_US-arctic-medium.onnx", use_cuda=True)

    # Set up the output audio stream for speech playback
    sd.start()

    # Synthesize and play the speech audio
    for audio_bytes in voice.synthesize_stream_raw(text):
        int_data = np.frombuffer(audio_bytes, dtype=np.int16)
        metric2 = datetime.now()
        logger.critical(metric2)
        sd.write(int_data)

    # Close the audio stream once speech playback is complete
    sd.stop()
    sd.close()
    ttsMetric2 = datetime.now()
    logger.error("TTS end", ttsMetric2)
    logger.critical({ttsMetric2 - start_time})


def main():
    """
    Main function to run the application
    """

    # Take start time for stream durations
    mainMetric1 = datetime.now()
    logger.error("Main function start", mainMetric1)
    logger.critical({mainMetric1 - start_time})

    # Initialize hotword detector and microphone stream
    hotword_detector = HotwordDetector(
        hotword="jippety", model=base_model, reference_file="voices/jippety_ref.json", continuous=False)
    hotword_mic_stream = SimpleMicStream(
        window_length_secs=1.5, sliding_window_secs=.5)

    # Initialize AssemblyAI real-time transcriber
    transcriber = aai.RealtimeTranscriber(
        sample_rate=16000, on_data=on_data, on_error=on_error, on_close=on_close)

    # Event objects for coordinating the transcription process
    transcribe_event = threading.Event()
    stop_transcription_event = threading.Event()

    # Start a thread for handling transcription
    transcription_thread = threading.Thread(target=handle_transcription,
                                            args=(transcriber,
                                                  transcribe_event,
                                                  stop_transcription_event,
                                                  transcript_queue))
    transcription_thread.start()

    # Main application loop
    try:
        while True:
            """
            Main application loop
            """

            mainLoopMetric1 = datetime.now()
            logger.error("Main loop start", mainLoopMetric1)
            logger.critical({mainLoopMetric1 - start_time})
            # Listen for the hotword
            if listen_for_hotword(hotword_detector, hotword_mic_stream):
                # Hotword detected, signal to start transcription
                transcribe_event.set()

                # Wait until a final transcript is available
                while not transcript_queue.empty():
                    try:
                        # Get transcript result from the queue
                        transcript_result = transcript_queue.get_nowait()

                        # Process the transcript with ChatGPT
                        chatgpt_response = process_with_chatgpt(
                            transcript_result)

                        # Read response aloud
                        text_to_speech(chatgpt_response)
                    except Empty:
                        # If the queue is empty, break the loop and go back to hotword detection
                        logger.info(
                            "No transcript available Back to hotword detection")
                        break
    except KeyboardInterrupt:
        mainLoopMetric2 = datetime.now()
        logger.critical("Main loop done, app terminating", mainLoopMetric2)
        logger.info("Application terminating...")

    # Signal to stop the transcription process
    stop_transcription_event.set()
    transcription_thread.join()  # Wait for the transcription thread to finish

    # Ensure a complete cleanup
    terminate_transcription()


if __name__ == "__main__":
    main()
