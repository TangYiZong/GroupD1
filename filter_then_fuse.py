import serial
import time
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt, savgol_filter
import matplotlib.pyplot as plt
import pandas as pd

def smooth(signal, window=21, polyorder=2):
    return savgol_filter(signal, window_length=window, polyorder=polyorder)

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='low', analog=False)

def apply_butterworth_filter(data, fs, cutoff=1.5, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)

def estimate_breaths(fused_values, timestamps,
                     threshold=4500,
                     min_interval_s=1,
                     smooth_win=5,
                     min_prominence=200):
    """
    Count one breath per prominent peak above threshold.
    Applies a moving‐average smooth, then finds peaks that:
      • exceed threshold
      • are at least min_prominence high above their surroundings
      • are separated by at least min_interval_s seconds
    Returns: (count, bpm, smoothed signal, peak indices)
    """
    signal = np.array(fused_values, dtype=float)
    times  = np.array(timestamps,  dtype=float)
    duration = times[-1] - times[0] if len(times) > 1 else 0.1
    sample_rate  = len(times) / duration if duration > 0 else 1
    min_distance = int(sample_rate * min_interval_s)

    # Smooth signal and trim borders
    smooth_sig_full = smooth(signal, window=smooth_win)
    trim = smooth_win // 2
    smooth_sig = smooth_sig_full[trim:-trim]
    times_trimmed = times[trim:-trim]

    # Peak finding
    peaks, props = find_peaks(
        smooth_sig,
        height=threshold,
        distance=min_distance,
        prominence=min_prominence
    )

    count = len(peaks)
    bpm = count * (60.0 / duration) if duration > 0 else 0.0
    return count, bpm, smooth_sig, peaks, times_trimmed

def main():
    input("Press ENTER to begin. Sync breathing to metronome, then CTRL+C to stop...\n")
    ser = serial.Serial('COM13', 115200, timeout=1)
    time.sleep(2)
    print("🔴 Recording started...\n")

    data = []
    try:
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            try:
                ch1, ch2 = map(int, line.split(','))
                data.append([elapsed, ch1, ch2])
                print(f"[{elapsed:5.2f}s] CH1: {ch1}   CH2: {ch2}")
            except ValueError:
                continue
    except KeyboardInterrupt:
        print("\n🛑 Recording stopped.\n")
    finally:
        ser.close()

    # Extract raw time‐series
    timestamps = np.array([row[0] for row in data])
    ch1_raw = np.array([row[1] for row in data], dtype=float)
    ch2_raw = np.array([row[2] for row in data], dtype=float)

    # Estimate sample rate
    duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.1
    fs = len(timestamps) / duration if duration > 0 else 1

    # Filter each channel
    ch1_filt = apply_butterworth_filter(ch1_raw, fs, cutoff=1.5)
    ch2_filt = apply_butterworth_filter(ch2_raw, fs, cutoff=1.5)

    # Fuse filtered channels
    fused_filt_full = ch1_filt + ch2_filt

    # Smooth + estimate breaths
    smooth_win = 5
    count, bpm, smooth_sig, peaks, timestamps_trimmed = estimate_breaths(
        fused_filt_full, timestamps,
        smooth_win=smooth_win
    )

    # Trim fused_filt to match
    trim = smooth_win // 2
    fused_filt = fused_filt_full[trim:-trim]

    print(f"Detected breathing cycles: {count}")
    print(f"Estimated breath rate: {bpm:.2f} BPM")

    # Save to CSV
    fused_raw = ch1_raw + ch2_raw  # Optional, for comparison
    df = pd.DataFrame({
        'Time (s)': timestamps,
        'CH1': ch1_raw,
        'CH2': ch2_raw,
        'Fused (raw)': fused_raw,
        'Fused (filtered)': fused_filt_full
    })
    df.to_csv('breathing_data.csv', index=False)
    print("✅ Data with fused signal saved to 'breathing_data.csv'")

    # —— PLOT RAW vs FILTERED & FUSED ——  
    plt.figure(figsize=(12, 6))

    # CH1
    plt.subplot(3, 1, 1)
    plt.plot(timestamps, ch1_raw, linestyle='--', color='orange', alpha=0.5, label='CH1 Raw')
    plt.plot(timestamps, ch1_filt, color='purple', label='CH1 Filtered')
    plt.legend(loc='upper right')
    plt.ylabel('ADC CH1 (Temperature)')
    plt.ylim([-500, 4500])

    # CH2
    plt.subplot(3, 1, 2)
    plt.plot(timestamps, ch2_raw, linestyle='--', color='orange', alpha=0.5, label='CH2 Raw')
    plt.plot(timestamps, ch2_filt, color='purple', label='CH2 Filtered')
    plt.legend(loc='upper right')
    plt.ylabel('ADC CH2 (Pressure)')
    plt.ylim([-500, 4500])

    # Fused & Smoothed
    plt.subplot(3, 1, 3)
    plt.plot(timestamps_trimmed, fused_filt, color='orange', label='Fused Filtered', linewidth=2)
    plt.plot(timestamps_trimmed, smooth_sig, color='purple', linestyle='--', label='Fused + MA Smooth', linewidth=1.5)
    plt.plot(timestamps_trimmed[peaks], smooth_sig[peaks], 'ro', label='Detected Peaks')
    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Fused ADC')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
