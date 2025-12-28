from scipy.signal import iirnotch, butter

fs = 8000 # 8000 Hz

# Notch 60Hz
b_notch, a_notch = iirnotch(60.0 / (0.5 * fs), 30.0)
print(f"Notch b: {list(b_notch)}")
print(f"Notch a: {list(a_notch)}")

# Bandpass 30-150Hz
b_band, a_band = butter(3, [30/(0.5*fs), 150/(0.5*fs)], btype='band')
print(f"Bandpass b: {list(b_band)}")
print(f"Bandpass a: {list(a_band)}")