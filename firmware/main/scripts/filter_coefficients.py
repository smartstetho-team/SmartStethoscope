'''
Docstring for scripts.filter_coefficients

Description: Use this script to generate coefficients for our
DSP filters.

Articles I looked at:
- https://www.earlevel.com/main/2021/09/02/biquad-calculator-v3/
- https://www.dspguide.com/ch19/1.htm
- https://www.wescottdesign.com/articles/Sampling/sampling.pdf
'''

from scipy.signal import iirnotch, butter

fs = 8000 # 8000 Hz, current nyquist/sampling rate

# Notch 60Hz
remove_freq = 60.0
q_factor = 30.0
b_notch, a_notch = iirnotch(remove_freq, q_factor, fs)

print("\nNotch Filter at 60Hz")
print(f"float coeffs[5] = {{{b_notch[0]:.6f}f, {b_notch[1]:.6f}f, {b_notch[2]:.6f}f, {a_notch[1]:.6f}f, {a_notch[2]:.6f}f}};\n")

# Band (30-150 Hz)
low = 30
high = 150
sos = butter(2, [low, high], btype='band', fs=fs, output='sos') # 'order=2' in SciPy's butter function creates a 4th-order bandpass (2 stages)

print("Bandpass 30-150Hz (Order 4) - 2 Stages")
for i, s in enumerate(sos):
    # s is [b0, b1, b2, a0, a1, a2] -> we skip a0 since its just 1 (index 3)
    print(f"float coeffs_s{i+1}[5] = {{{s[0]:.6f}f, {s[1]:.6f}f, {s[2]:.6f}f, {s[4]:.6f}f, {s[5]:.6f}f}};")