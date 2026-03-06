import sounddevice as sd
import keyboard
import threading
import time

# Global variable to toggle transcription
transcription_active = False

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio stream error: {status}")
    if transcription_active:
        # Process audio data here...
        print("Transcription is active...")  # Placeholder for transcription logic

def toggle_transcription():
    global transcription_active
    transcription_active = not transcription_active
    if transcription_active:
        print("Transcription started.")
    else:
        print("Transcription stopped.")

def hotkey_listener():
    # Listen for the hotkey in a separate thread
    keyboard.add_hotkey('ctrl+shift+t', toggle_transcription)
    print("Hotkey listener started. Press Ctrl+Shift+T to toggle transcription.")
    while True:
        time.sleep(1)  # Keep the thread alive

# Increase buffer size and reduce sample rate
stream = sd.InputStream(
    channels=2,  # Stereo input
    samplerate=22050,  # Reduced sample rate
    blocksize=2048,  # Increased buffer size
    callback=audio_callback
)

# Start the audio stream
stream.start()
print("Audio stream started.")

# Start the hotkey listener in a separate thread
hotkey_thread = threading.Thread(target=hotkey_listener, daemon=True)
hotkey_thread.start()

# Keep the program running
try:
    while True:
        pass
except KeyboardInterrupt:
    stream.stop()
    print("Audio stream stopped.")
