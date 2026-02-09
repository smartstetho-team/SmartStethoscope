import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Load the files
noisy_path = 'stethoscope_test_pot_top.wav'
clean_path = 'stethoscope_heart_clean.wav'

y_noisy, sr_noisy = librosa.load(noisy_path)
y_clean, sr_clean = librosa.load(clean_path)

# Create the plot
plt.figure(figsize=(12, 8))

# Plot Noisy Signal
plt.subplot(2, 1, 1)
librosa.display.waveshow(y_noisy, sr=sr_noisy, color='r', alpha=0.6)
plt.title('Unfiltered Signal (Test/Noise Input)')
plt.ylabel('Amplitude')
plt.xlabel('Time (s)')

# Plot Clean Signal
plt.subplot(2, 1, 2)
librosa.display.waveshow(y_clean, sr=sr_clean, color='b', alpha=0.8)
plt.title('Filtered Signal (Heartbeat Output)')
plt.ylabel('Amplitude')
plt.xlabel('Time (s)')

plt.tight_layout()
plt.show()