import wave
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

IN_FILE = "stethoscope_test_pot_bottom.wav" # Or your bottom file
HEART_FILE = "stethoscope_heart_clean_bottom.wav"

def load_wav(path):
    try:
        with wave.open(path, "rb") as wf:
            fs = wf.getframerate()
            n_frames = wf.getnframes()
            audio = wf.readframes(n_frames)
        # Convert to float
        data = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
        # Normalize to -1 to 1
        data /= 32768.0
        return data, fs
    except FileNotFoundError:
        print(f"Error: Could not find file {path}")
        return np.array([]), 0

def save_wav(path, data, fs):
    # Normalize before saving to maximize volume without clipping
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val * 0.9  # Normalize to 90% volume
    
    data = np.clip(data, -1.0, 1.0)
    pcm = (data * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(pcm.tobytes())

def notch_filter(data, fs, freq=60.0, quality=30.0):
    """
    Removes a specific frequency (like electrical hum).
    freq: The frequency to remove (60Hz for North America)
    quality: The 'sharpness' of the notch. Higher = narrower cut.
    """
    nyq = 0.5 * fs
    norm_freq = freq / nyq
    b, a = iirnotch(norm_freq, quality)
    return filtfilt(b, a, data)

def butter_bandpass(data, fs, lowcut, highcut, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def enhance_signal(data):
    """
    Optional: Non-linear enhancement to suppress quiet noise 
    and emphasize loud peaks (heartbeats).
    """
    # Square the signal to emphasize peaks (Shannon Energy approach)
    # Note: This changes the 'sound' but makes the beat easier to detect.
    # For pure listening, you might skip this or use a milder power (1.5).
    return np.sign(data) * (np.abs(data) ** 1.5)

def main():
    print(f"Loading {IN_FILE}...")
    data, fs = load_wav(IN_FILE)
    
    if len(data) == 0:
        return

    print(f"Sample rate: {fs} Hz, Length: {len(data)/fs:.2f}s")

    # --- STEP 1: Remove DC Offset ---
    data = data - np.mean(data)

    # --- STEP 2: Notch Filter (Remove 60Hz Mains Hum) ---
    # Critical step for DIY electronics
    data_notched = notch_filter(data, fs, freq=60.0, quality=30.0)
    
    # If you have harmonics (120Hz), you can repeat the notch:
    # data_notched = notch_filter(data_notched, fs, freq=120.0, quality=30.0)

    # --- STEP 3: Optimized Heart Bandpass ---
    # 30Hz: Removes generic body rumble/handling noise
    # 150Hz: Most fundamental heart sounds are below this. 
    # Going up to 200Hz+ introduces hiss without adding clarity.
    heart = butter_bandpass(data_notched, fs, 30, 150, order=3)

    # --- STEP 4 (Optional): Signal Enhancement ---
    # Uncomment the line below to make the beats "punchier" but less natural
    # heart = enhance_signal(heart)

    save_wav(HEART_FILE, heart, fs)
    print(f"Saved processed file: {HEART_FILE}")

if __name__ == "__main__":
    main()


"""
-------------------------------------------------------------------------
WHY THIS FILTERING IS BETTER
-------------------------------------------------------------------------
1. The Notch Filter (fc = 60Hz): 
   Standard bandpass filters often miss specific noise. By targeting the 
   60Hz mains hum (common in North America), we clear up the "mud" 
   without deleting the surrounding heart frequencies.

2. Tighter Bandwidth (30-150Hz):
   - Lower Bound (30Hz): Moving from 20Hz to 30Hz aggressively cuts 
     "handling noise" (friction/rumble from fingers).
   - Upper Bound (150Hz): S1 and S2 heart sounds are low thuds. High 
     frequencies are mostly hiss. Lowering the ceiling from 200Hz to 
     150Hz significantly reduces static.

3. Normalization: 
   The save_wav function now automatically maximizes the volume. 
   MAX4466 gain is tricky; usually, the signal is too quiet or clipping. 
   This ensures the output is audible.

-------------------------------------------------------------------------
PHYSICAL / HARDWARE DEBUGGING
-------------------------------------------------------------------------
Software can only do so much. If the audio is still unclear, check these:

1. Acoustic Impedance: 
   Are you placing the raw microphone against the skin? This rarely works. 
   You need a diaphragm (like a piece of plastic from a milk jug or a 
   real stethoscope head) to capture vibration and transfer it to the 
   air inside the mic chamber.

2. Air Seal: 
   The MAX4466 must be in a sealed chamber against the skin. If air 
   leaks out the side, you lose the low frequencies (the bass of the heart).

3. Pressure: 
   Low frequencies require firm pressure against the chest.
-------------------------------------------------------------------------
"""