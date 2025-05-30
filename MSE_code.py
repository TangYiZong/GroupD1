import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error # For MSE calculation 
from tkinter import Tk, filedialog

# === CONFIG ===
start_time = 0
end_time = 65
data_path = r"C:\Users\gwqr0\OneDrive - Monash University\Year3\Y3S2\TRC3500\M3\breathing_data_resting.csv"

# === LOAD FULL DATA ===
full_df = pd.read_csv(data_path)

# === Identify and skip first group of continuous 'Yes' markers ===
yes_mask = full_df['Marker'] == 'Yes'
yes_indices = full_df[yes_mask].index

for i in range(1, len(yes_indices)):
    if yes_indices[i] != yes_indices[i - 1] + 1:
        end_of_first_block_idx = yes_indices[i - 1]
        break
else:
    end_of_first_block_idx = yes_indices[-1]

cutoff_time = full_df.loc[end_of_first_block_idx, 'Time (s)']
print(f"Skipping setup markers, trimming data after {cutoff_time:.3f}s")

# Trim to data after setup
df = full_df[full_df['Time (s)'] > cutoff_time].copy()
timestamps = df['Time (s)'].values
ch1 = df['CH1']
ch2 = df['CH2']
fused = df['Fused (raw)'].values
marker_times_raw = df[df['Marker'] == 'Yes']['Time (s)']

#added this for git 
# === SENSOR INHALE/EXHALE DETECTION ===
smoothed = savgol_filter(fused, window_length=51, polyorder=2)
peaks, _ = find_peaks(smoothed, prominence=500, distance=100)

inhale_starts = []
exhale_starts = []

for peak_idx in peaks:
    peak_val = smoothed[peak_idx]
    # Inhale
    i = peak_idx
    while i > 0 and timestamps[peak_idx] - timestamps[i] < 3:
        if smoothed[i] < smoothed[i - 1] and smoothed[i] < peak_val * 0.7:
            inhale_starts.append(timestamps[i])
            break
        i -= 1
    # Exhale
    i = peak_idx
    while i < len(smoothed) - 2 and timestamps[i] - timestamps[peak_idx] < 3:
        time_since_peak = timestamps[i] - timestamps[peak_idx]
        if smoothed[i] > smoothed[i + 1] > smoothed[i + 2] and smoothed[i] < peak_val * 0.7 and time_since_peak > 1.5:
            exhale_starts.append(timestamps[i])
            break
        i += 1

# Add final exhale if needed
if len(exhale_starts) < len(inhale_starts):
    final_exhale_time = timestamps[-1]
    exhale_starts.append(final_exhale_time)
    print(f"Added final sensor exhale at end of data: {final_exhale_time:.3f}s")

# Save sensor output
n = min(len(inhale_starts), len(exhale_starts))
sensor_df = pd.DataFrame({
    "Sensor Inhale (s)": inhale_starts[:n],
    "Sensor Exhale (s)": exhale_starts[:n]
})
sensor_df.to_csv("inhale_exhale_paired.csv", index=False)
print("\nSensor inhale/exhale times:")
print(sensor_df.head())

# === MANUAL MARKER DEBOUNCE ===
yes_df = df[df['Marker'] == 'Yes'][['Time (s)']].copy()
yes_df['Time Diff'] = yes_df['Time (s)'].diff()
event_threshold = 0.3
yes_df['New Event'] = (yes_df['Time Diff'] > event_threshold) | (yes_df['Time Diff'].isna())
marker_cleaned = yes_df[yes_df['New Event']]['Time (s)'].values

manual_inhale = marker_cleaned[::2]
manual_exhale = marker_cleaned[1::2]

# Add missing exhale after final inhale if needed
if len(manual_inhale) > len(manual_exhale):
    last_unpaired_inhale = manual_inhale[-1]
    if len(manual_exhale) >= 3:
        durations = manual_exhale[-3:] - manual_inhale[-4:-1]
        avg_duration = np.mean(durations)
    else:
        avg_duration = 3.0
    estimated_exhale = last_unpaired_inhale + avg_duration
    manual_exhale = np.append(manual_exhale, estimated_exhale)
    print(f"Added estimated manual exhale at {estimated_exhale:.3f}s")

# Save manual output
m = min(len(manual_inhale), len(manual_exhale))
manual_df = pd.DataFrame({
    "Ground Truth Inhale (s)": manual_inhale[:m],
    "Ground Truth Exhale (s)": manual_exhale[:m]
})
manual_df.to_csv("manual_marker_inhale_exhale.csv", index=False)
print("\nGround Truth inhale/exhale times:")
print(manual_df.head())

# # === PLOT ===
mask = (timestamps >= start_time) & (timestamps <= end_time)
t_zoom = timestamps[mask]
ch1_zoom = ch1[mask]
ch2_zoom = ch2[mask]
fused_zoom = fused[mask]
x_ticks = np.arange(start_time, end_time, 0.1)

plt.figure(figsize=(14, 10))

# # CH1
# plt.subplot(3, 1, 1)
# plt.plot(t_zoom, ch1_zoom, label='CH1 (Temperature)', color='blue')
# for t in marker_times_raw:
#     if start_time <= t <= end_time:
#         plt.axvline(x=t, color='red', linestyle='--', alpha=0.5)
#         plt.text(t, max(ch1_zoom), 'Marker', color='red', rotation=90, fontsize=8, ha='center', va='bottom')
# plt.xticks(x_ticks)
# plt.title("CH1 with Markers")
# plt.ylabel("ADC Value")
# plt.grid(True)
# plt.legend()

# # CH2
# plt.subplot(3, 1, 2)
# plt.plot(t_zoom, ch2_zoom, label='CH2 (Pressure)', color='green')
# for t in marker_times_raw:
#     if start_time <= t <= end_time:
#         plt.axvline(x=t, color='red', linestyle='--', alpha=0.5)
#         plt.text(t, max(ch2_zoom), 'Marker', color='red', rotation=90, fontsize=8, ha='center', va='bottom')
# plt.xticks(x_ticks)
# plt.title("CH2 with Markers")
# plt.ylabel("ADC Value")
# plt.grid(True)
# plt.legend()

# # Fused
# plt.subplot(3, 1, 3)
# plt.plot(t_zoom, fused_zoom, label='Fused Signal', color='purple')
# for t in marker_times_raw:
#     if start_time <= t <= end_time:
#         plt.axvline(x=t, color='red', linestyle='--', alpha=0.5)
#         plt.text(t, max(fused_zoom), 'Marker', color='red', rotation=90, fontsize=8, ha='center', va='bottom')
# for t in inhale_starts:
#     if start_time <= t <= end_time:
#         plt.axvline(x=t, color='blue', linestyle='--', alpha=0.5)
#         plt.text(t, max(fused_zoom)*0.95, 'Inhale', color='blue', rotation=90, fontsize=8, ha='center')
# for t in exhale_starts:
#     if start_time <= t <= end_time:
#         plt.axvline(x=t, color='green', linestyle='--', alpha=0.5)
#         plt.text(t, max(fused_zoom)*0.9, 'Exhale', color='green', rotation=90, fontsize=8, ha='center')
# plt.xticks(x_ticks)
# plt.title("Fused Signal with Sensor & Manual Events")
# plt.xlabel("Time (s)")
# plt.ylabel("ADC Value")
# plt.grid(True)
# plt.legend()

# plt.tight_layout()
# plt.show()

# === SIMPLIFIED PLOT: Fused Signal + Sensor and Manual Markers ===
plt.figure(figsize=(15, 5))
plt.plot(timestamps, fused, label='Fused Signal', color='purple')

# Plot sensor-detected inhale and exhale events
for t in inhale_starts:
    plt.axvline(x=t, color='blue', linestyle='--', alpha=0.6)
    plt.text(t, max(fused) * 0.95, 'Inhale', color='blue', rotation=90, fontsize=8, ha='center')

for t in exhale_starts:
    plt.axvline(x=t, color='green', linestyle='--', alpha=0.6)
    plt.text(t, max(fused) * 0.9, 'Exhale', color='green', rotation=90, fontsize=8, ha='center')

# Plot manual marker events
for t in marker_cleaned:
    plt.axvline(x=t, color='red', linestyle='--', alpha=0.4)
    plt.text(t, max(fused) * 0.85, 'Marker', color='red', rotation=90, fontsize=8, ha='center')

plt.title("Fused Signal with Sensor Breathing Events and Manual Markers")
plt.xlabel("Time (s)")
plt.ylabel("ADC Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# === COMBINED MSE Calculation ===
# Ensure equal length for both
min_len = min(len(sensor_df), len(manual_df))
sensor_inhales = sensor_df["Sensor Inhale (s)"][:min_len].values
sensor_exhales = sensor_df["Sensor Exhale (s)"][:min_len].values
ground_truth_inhales = manual_df["Ground Truth Inhale (s)"][:min_len].values
ground_truth_exhales = manual_df["Ground Truth Exhale (s)"][:min_len].values

# Combine inhale and exhale into single arrays
sensor_all = np.concatenate([sensor_inhales, sensor_exhales])
ground_truth_all = np.concatenate([ground_truth_inhales, ground_truth_exhales])

squared_errors = []
for i in range(len(sensor_all)):
    sensor_val = sensor_all[i]
    ground_truth_val = ground_truth_all[i]
    error = sensor_val - ground_truth_val
    squared_error = error ** 2
    squared_errors.append(squared_error)
    print(f"Pair {i+1}: Sensor = {sensor_val:.6f}, Ground Truth = {ground_truth_val:.6f}, Squared Error = {squared_error:.6f}")

    
# Calculate MSE manually
sum_squared_error = 0
for val in squared_errors:
    sum_squared_error += val

manual_mse = sum_squared_error / len(squared_errors)

print(f"MSE = {manual_mse:.6f} s²")

# === SAVE TO CSV FOR EXCEL ===
error_df = pd.DataFrame({
    "Sensor Time (s)": sensor_all,
    "Ground Truth Time (s)": ground_truth_all,
    "Squared Error (s²)": squared_errors
})

# Prompt user for location
root = Tk()
root.withdraw()
save_path = filedialog.asksaveasfilename(
    defaultextension=".csv",
    filetypes=[("CSV files", "*.csv")],
    title="Save Squared Errors CSV"
)

if save_path:
    error_df.to_csv(save_path, index=False)
    print(f"\nSquared error data saved to:\n{save_path}")
else:
    print("\nSave cancelled.")
