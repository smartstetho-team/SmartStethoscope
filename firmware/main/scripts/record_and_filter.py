import serial
import wave
import struct
import time
import numpy as np

from filter import load_wav, notch_filter, butter_bandpass, save_wav


PORT = "COM5"  # adjust if different
BAUD = 921600
SAMPLE_RATE = 8000
RECORD_SECONDS = 10
NUM_SAMPLES = SAMPLE_RATE * RECORD_SECONDS

OUTPUT_WAV = "../recordings/stethoscope_test.wav"
OUTPUT_WAV_CLEAN = "../recordings/stethoscope_heart_clean.wav"


print("Opening serial port...")
ser = serial.Serial()
ser.port = PORT
ser.baudrate = BAUD
ser.timeout = 2
ser.dsrdtr = False
ser.rtscts = False
ser.dtr = False
ser.rts = False
ser.open()

# Give board time + flush any old text
time.sleep(2)
ser.reset_input_buffer()

# (Optional) read the READY line
try:
  line = ser.readline().decode(errors="ignore").strip()
  if line:
      print("ESP32 says:", line)
except Exception:
  pass

print("Sending 'r' to start recording...")
ser.write(b"r")

print("Reading samples from ESP32...")
needed_bytes = NUM_SAMPLES * 4
buf = bytearray()

while len(buf) < needed_bytes:
    chunk = ser.read(needed_bytes - len(buf))
    if not chunk:
        # no more data within timeout
        break
    buf.extend(chunk)

ser.close()

print(f"Collected {len(buf)//4} samples (expected {NUM_SAMPLES}).")

if len(buf) < 2:
    print("No data, nothing to save.")
    raise SystemExit

# Convert bytes -> numpy array
samples = []
for i in range(0, len(buf), 4):
    (raw_val,) = struct.unpack('>H', buf[i:i+2])  # 0..4095-ish
    value = raw_val & 0x0FFF
    samples.append(value)

samples = np.array(samples, dtype=np.float32)

# Map ADC 0-4095 -> -1.0 .. 1.0
samples = (samples / 4095.0) * 2.0 - 1.0

samples_pcm = (samples * 32767).astype(np.int16)

with wave.open(OUTPUT_WAV, "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)      # 16-bit
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(samples_pcm.tobytes())

print(f"Saved RAW {OUTPUT_WAV}")


print(f"Loading {OUTPUT_WAV}...")
data, fs = load_wav(OUTPUT_WAV)

if len(data) == 0:
    print("FAILED TO GET DATA FROM RAW WAVE FILE")

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

save_wav(OUTPUT_WAV_CLEAN, heart, fs)
print(f"Saved processed file: {OUTPUT_WAV_CLEAN}")
