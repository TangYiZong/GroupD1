import serial
import time
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt

def smooth(signal, window=5):
    return np.convolve(signal, np.ones(window)/window, mode='same')

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
    Returns: (count, bpm)
    """
    if len(fused_values) < 2:
        return 0, 0.0

    signal = np.array(fused_values, dtype=float)
    times  = np.array(timestamps,  dtype=float)

    # Total recording duration
    duration = times[-1] - times[0]
    if duration <= 0:
        return 0, 0.0

    # Estimate sample rate and min distance in samples
    sample_rate  = len(times) / duration
    min_distance = int(sample_rate * min_interval_s)

    # 1) Smooth the signal
    kernel = np.ones(smooth_win) / smooth_win
    smooth_sig = np.convolve(signal, kernel, mode='same')

    # 2) Find only truly prominent peaks
    peaks, props = find_peaks(
        smooth_sig,
        height=threshold,
        distance=min_distance,
        prominence=min_prominence
    )

    count = len(peaks)
    bpm   = count * (60.0 / duration)
    return count, bpm

def main():
    input("Press ENTER to begin. Start breathing with metronome (5 cycles), press CTRL+C to stop...\n")

    ser = serial.Serial('COM13', 115200, timeout=1)
    time.sleep(2)

    print("🔴 Recording started. Sync your breathing with the metronome...\n")

    data = []

    try:
        start_time = time.time()
        while True:
            now = time.time()
            elapsed = now - start_time

            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue

            try:
                ch1, ch2 = map(int, line.split(','))
                fused = ch1 + ch2
                data.append([elapsed, ch1, ch2, fused])
                print(f"[{elapsed:5.2f}s]  CH1: {ch1}   CH2: {ch2}   Fused: {fused}")
            except ValueError:
                print(f"→ Malformed line: '{line}'")
    except KeyboardInterrupt:
        print("\n Recording stopped by user.\n")
    finally:
        ser.close()
    
    # Extract time‐series
    timestamps   = [row[0] for row in data]
    ch1_values   = [row[1] for row in data]
    ch2_values   = [row[2] for row in data]
    fused_values = [row[3] for row in data]

    # Estimate sample rate
    duration = timestamps[-1] - timestamps[0]
    sample_rate = len(timestamps) / duration if duration > 0 else 1

    # Apply Butterworth filters
    ch1_filtered   = apply_butterworth_filter(ch1_values, sample_rate, cutoff=1.5)
    ch2_filtered   = apply_butterworth_filter(ch2_values, sample_rate, cutoff=1.5)
    fused_filtered = apply_butterworth_filter(fused_values, sample_rate, cutoff=1.5)

    # Estimate breathing
    count, bpm = estimate_breaths(fused_filtered, timestamps)
    print(f" Detected breathing cycles: {count}")
    print(f" Estimated breath rate: {bpm:.2f} BPM")

    # ——⭑ PLOT FILTERED & RAW SIGNALS ⭑——
    plt.figure(figsize=(12, 6))

    # Raw signals (dashed and transparent)
    plt.plot(timestamps, ch1_values, '--', alpha=0.4, label='CH1 Raw (Temp)')
    plt.plot(timestamps, ch2_values, '--', alpha=0.4, label='CH2 Raw (Pressure)')
    plt.plot(timestamps, fused_values, '--', alpha=0.4, label='Fused Raw')

    # Filtered signals
    plt.plot(timestamps, ch1_filtered, label='CH1 Filtered (Temp)')
    plt.plot(timestamps, ch2_filtered, label='CH2 Filtered (Pressure)')
    plt.plot(timestamps, fused_filtered, label='Fused Filtered', linewidth=2)

    plt.xlabel('Time (s)')
    plt.ylabel('ADC Value')
    plt.title('ADC Signals: Raw vs Filtered (Butterworth)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
