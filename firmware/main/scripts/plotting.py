import matplotlib.pyplot as plt
from scipy.signal import welch

def plot_comparison(raw_data, filtered_data, fs):
    plt.figure(figsize=(12, 8))

    # 1. Time Domain Comparison
    plt.subplot(2, 1, 1)
    t = [i/fs for i in range(len(raw_data))] # x-axis (time)

    plt.plot(t, raw_data, label='Unfiltered', alpha=0.6)
    plt.plot(t, filtered_data, label='After 60Hz Notch', color='red', linewidth=1)
    plt.title("Time Domain: Signal Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.xlim(1, 6) # 1-6 seconds
    plt.legend()
    plt.grid(True)

    # 2. Frequency Domain (PSD)
    plt.subplot(2, 1, 2)
    f_raw, p_raw = welch(raw_data, fs, nperseg=2048)
    f_filt, p_filt = welch(filtered_data, fs, nperseg=2048)
    
    plt.semilogy(f_raw, p_raw, label='Unfiltered PSD', alpha=0.6)
    plt.semilogy(f_filt, p_filt, label='Notched PSD', color='red')
    plt.axvline(60, color='green', linestyle='--', label='60Hz Mains')
    
    plt.title("Frequency Domain: Power Spectral Density")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.xlim(0, 500) # Most heart sounds are < 500Hz
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()