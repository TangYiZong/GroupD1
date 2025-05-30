import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit

# Load breathing data during rest
df = pd.read_csv("breathing_data.csv")
timestamps = df['Time (s)'].values
fused_filtered = df['Fused (filtered)'].values

# Smoothing the signal with Savitzky-Golay filter
def smooth(signal, window=21, poly=2):
    return savgol_filter(signal, window_length=window, polyorder=poly)

# Estimate breath count from smoothed signal
def estimate_breaths(signal, time, threshold=4500, min_spacing=1, win=5, min_prom=200):
    smoothed = smooth(signal, window=win)
    trim = win // 2
    sig_crop = smoothed[trim:-trim]
    time_crop = time[trim:-trim]

    duration = time[-1] - time[0]
    sampling_rate = len(time) / duration if duration > 0 else 1
    min_distance = int(sampling_rate * min_spacing)

    peaks, _ = find_peaks(sig_crop, height=threshold, distance=min_distance, prominence=min_prom)
    count = len(peaks)
    return count

# Sigmoid Function
def sigmoid(x, k, x0):
    z = np.clip(-k * (x - x0), -500, 500)
    return 1 / (1 + np.exp(z))

# Fitting with basic error handling
def try_fit(x, y, p0):
    params, _ = curve_fit(sigmoid, x, y, p0=p0, maxfev=5000)
    return params

# Set up trimmed signal
win = 5
fused_trim = fused_filtered[win//2: -win//2]
time_trim = timestamps[win//2: -win//2]
baseline = estimate_breaths(fused_filtered, timestamps)
print(f"Baseline breath count: {baseline}")

#Additive Noise
def test_jnd_noise(signal, time, baseline, trials=10):
    levels = np.linspace(10, 1000, 10)
    detected = []
    for level in levels:
        changed = 0
        for _ in range(trials):
            noisy = signal + level * np.random.normal(0, 1, len(signal))
            count = estimate_breaths(noisy, time)
            if abs(count - baseline) >= 1:
                changed += 1
        detected.append(changed / trials)
    return levels, detected

#time-stretching
def test_jnd_stretch(signal, time, baseline, trials=10):
    factors = np.linspace(0.4, 1.6, 15)
    detected = []
    for f in factors:
        changed = 0
        for _ in range(trials):
            stretched = np.interp(
                np.linspace(0, len(signal) - 1, int(len(signal) * f)),
                np.arange(len(signal)),
                signal
            )
            if len(stretched) < len(signal):
                stretched = np.pad(stretched, (0, len(signal) - len(stretched)))
            else:
                stretched = stretched[:len(signal)]
            count = estimate_breaths(stretched, time)
            if abs(count - baseline) >= 1:
                changed += 1
        detected.append(changed / trials)
    return factors, detected

# Run JND tests
noise_x, noise_y= test_jnd_noise(fused_trim, time_trim, baseline)
stretch_x, stretch_y= test_jnd_stretch(fused_trim, time_trim, baseline)

# Fit curves
params_noise, _ = curve_fit(sigmoid, noise_x, noise_y, p0=[0.01, 500], maxfev=5000)
params_stretch = try_fit(stretch_x, stretch_y, p0=[10, 1.0])
jnd_noise = params_noise[1]
jnd_stretch = params_stretch[1]

#Pltting
plt.figure()
plt.plot(noise_x, noise_y, 'bo', label=(f'Additive Noise (JND= {jnd_noise:.2f} ADC)'))
plt.plot(np.linspace(10, 1000, 200), sigmoid(np.linspace(10, 1000, 200), *params_noise), 'b--')
plt.plot(stretch_x, stretch_y, 'go', label=(f'Time Stretching (JND= {jnd_stretch:.2f})'))
plt.plot(np.linspace(0.4, 1.6, 200), sigmoid(np.linspace(0.4, 1.6, 200), *params_stretch), 'g--')
plt.axhline(0.5, color='r', linestyle='--', label='50% Threshold')
plt.xlabel("Stimulus Difference")
plt.ylabel("Proportion Judged as Different")
plt.title("Just Noticeable Difference (JND)")
plt.legend()
plt.grid(True)
plt.show()

print(f"JND (Noise): {jnd_noise:.1f} ADC")
print(f"JND (Time Stretch): {jnd_stretch:.2f}")
