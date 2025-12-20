import serial
import wave
import struct
import time
import numpy as np

PORT = "COM5"  # adjust if different
BAUD = 115200
SAMPLE_RATE = 8000
RECORD_SECONDS = 10
NUM_SAMPLES = SAMPLE_RATE * RECORD_SECONDS

OUTPUT_WAV = "stethoscope_test.wav"

print("Opening serial port...")
ser = serial.Serial(PORT, BAUD, timeout=1)

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
needed_bytes = NUM_SAMPLES * 2
buf = bytearray()

while len(buf) < needed_bytes:
    chunk = ser.read(needed_bytes - len(buf))
    if not chunk:
        # no more data within timeout
        break
    buf.extend(chunk)

ser.close()

print(f"Collected {len(buf)//2} samples (expected {NUM_SAMPLES}).")

if len(buf) < 2:
    print("No data, nothing to save.")
    raise SystemExit

# Convert bytes -> numpy array
samples = []
for i in range(0, len(buf), 2):
    (value,) = struct.unpack('>H', buf[i:i+2])  # 0..4095-ish
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

print(f"Saved {OUTPUT_WAV}")
