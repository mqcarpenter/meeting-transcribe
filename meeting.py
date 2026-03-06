import sounddevice as sd
import numpy as np
import whisper
import pyautogui
import keyboard
import threading
import time
import os
import queue
import datetime
import tempfile
import soundfile as sf

# ============================================================
# Configuration
# ============================================================
OUTPUT_DIR = r"C:\py\Meetings"
SAMPLE_RATE = 16000          # Whisper expects 16kHz audio
BLOCK_SIZE = 4096            # Buffer size to prevent input overflow
CHANNELS = 2                 # Stereo for speaker separation
CHUNK_DURATION = 5           # Seconds of audio per transcription chunk
WHISPER_MODEL = "base"       # Options: tiny, base, small, medium, large
HOTKEY = "ctrl+shift+t"      # Hotkey to toggle transcription

# ============================================================
# Global Variables
# ============================================================
transcription_active = False
audio_queue_left = queue.Queue()
audio_queue_right = queue.Queue()
model = None
output_file = None
lock = threading.Lock()

# ============================================================
# Load Whisper Model
# ============================================================
def load_whisper_model():
    global model
    print(f"Loading Whisper model '{WHISPER_MODEL}'... This may take a moment.")
    model = whisper.load_model(WHISPER_MODEL)
    print("Whisper model loaded successfully.")

# ============================================================
# Audio Callback - Captures stereo audio and splits channels
# ============================================================
def audio_callback(indata, frames, time_info, status):
    if status:
        # Log status but do not print excessively
        pass
    if transcription_active:
        # Split stereo channels
        # Left channel = Speaker 1 (Microphone)
        # Right channel = Speaker 2 (System/Speaker output)
        left_channel = indata[:, 0].copy()
        right_channel = indata[:, 1].copy()
        audio_queue_left.put(left_channel)
        audio_queue_right.put(right_channel)

# ============================================================
# Transcribe Audio Chunk using Whisper
# ============================================================
def transcribe_audio(audio_data, speaker_label):
    global output_file
    try:
        # Normalize audio data
        audio_data = audio_data.astype(np.float32)
        audio_data = audio_data / np.max(np.abs(audio_data) + 1e-7)

        # Check if audio contains meaningful sound (not silence)
        if np.max(np.abs(audio_data)) < 0.01:
            return  # Skip silent audio

        # Save audio to a temporary WAV file for Whisper
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, f"temp_audio_{speaker_label}.wav")
        sf.write(temp_file, audio_data, SAMPLE_RATE)

        # Transcribe using Whisper
        result = model.transcribe(temp_file, fp16=False)
        text = result.get("text", "").strip()

        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)

        if text:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            formatted_text = f"[{timestamp}] {speaker_label}: {text}"
            print(formatted_text)

            # Write to file
            with lock:
                if output_file:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(formatted_text + "\n")

            # Type into active window using pyautogui
            try:
                pyautogui.typewrite(formatted_text + "\n", interval=0.01)
            except Exception as e:
                print(f"Error typing into active window: {e}")

    except Exception as e:
        print(f"Transcription error for {speaker_label}: {e}")

# ============================================================
# Process Audio Queue for a Speaker
# ============================================================
def process_audio_queue(audio_queue, speaker_label):
    buffer = np.array([], dtype=np.float32)
    samples_per_chunk = SAMPLE_RATE * CHUNK_DURATION

    while True:
        if transcription_active:
            try:
                # Get audio data from queue with timeout
                data = audio_queue.get(timeout=1)
                buffer = np.concatenate((buffer, data))

                # Process when buffer has enough data
                if len(buffer) >= samples_per_chunk:
                    chunk = buffer[:samples_per_chunk]
                    buffer = buffer[samples_per_chunk:]
                    transcribe_audio(chunk, speaker_label)
            except queue.Empty:
                # Process remaining buffer if it has meaningful data
                if len(buffer) > SAMPLE_RATE:  # At least 1 second
                    transcribe_audio(buffer, speaker_label)
                    buffer = np.array([], dtype=np.float32)
        else:
            # Clear buffer when transcription is stopped
            buffer = np.array([], dtype=np.float32)
            # Clear the queue
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    break
            time.sleep(0.5)

# ============================================================
# Toggle Transcription
# ============================================================
def toggle_transcription():
    global transcription_active, output_file
    transcription_active = not transcription_active
    if transcription_active:
        # Create output file with timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = os.path.join(OUTPUT_DIR, f"transcription_{timestamp}.txt")
        print(f"\n>>> Transcription STARTED. Output file: {output_file}")
        print(">>> Transcribing... Press Ctrl+Shift+T to stop.\n")
    else:
        print(f"\n>>> Transcription STOPPED. File saved to: {output_file}\n")

# ============================================================
# Hotkey Listener
# ============================================================
def hotkey_listener():
    print(f"Hotkey listener started. Press {HOTKEY} to toggle transcription.")
    keyboard.add_hotkey(HOTKEY, toggle_transcription)
    keyboard.wait()  # Block this thread and listen for hotkeys

# ============================================================
# Main Entry Point
# ============================================================
def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # List available audio devices
    print("\n=== Available Audio Devices ===")
    print(sd.query_devices())
    print("================================\n")

    # Load Whisper model
    load_whisper_model()

    # Start audio stream
    try:
        stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype="float32",
            callback=audio_callback
        )
        stream.start()
        print("Audio stream started successfully.")
    except Exception as e:
        print(f"Error starting audio stream: {e}")
        print("Please check your audio device configuration.")
        return

    # Start hotkey listener in a separate thread
    hotkey_thread = threading.Thread(target=hotkey_listener, daemon=True)
    hotkey_thread.start()

    # Start speaker processing threads
    speaker1_thread = threading.Thread(
        target=process_audio_queue,
        args=(audio_queue_left, "Speaker 1"),
        daemon=True
    )
    speaker2_thread = threading.Thread(
        target=process_audio_queue,
        args=(audio_queue_right, "Speaker 2"),
        daemon=True
    )
    speaker1_thread.start()
    speaker2_thread.start()

    print("\n=== Meeting Transcription Tool ===")
    print(f"Press {HOTKEY} to START/STOP transcription.")
    print("Press Ctrl+C to EXIT the program.")
    print("===================================\n")

    # Keep the program running
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        stream.stop()
        stream.close()
        print("\nProgram exited. Audio stream closed.")

if __name__ == "__main__":
    main()
