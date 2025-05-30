import serial
import time
import matplotlib.pyplot as plt
from collections import deque

def main():
    # — Serial init —
    try:
        ser = serial.Serial('COM8', 115200, timeout=1)
        time.sleep(2)
    except serial.SerialException as e:
        print(f"Error opening COM port: {e}")
        return

    print("Listening on COM8 for temp,pressure values…\n")

    # — Data buffers & plot setup —
    max_pts = 100
    temps     = deque(maxlen=max_pts)
    pressures = deque(maxlen=max_pts)
    times     = deque(maxlen=max_pts)
    start = time.time()

    plt.ion()
    fig, ax = plt.subplots()
    line1, = ax.plot([], [], label="Temp (°C)")
    line2, = ax.plot([], [], label="Pressure (Pa)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    ax.set_title("Live Temp & Pressure")
    ax.legend()

    try:
        while True:
            raw = ser.readline().decode('utf-8', errors='ignore').strip()
            if not raw:
                continue

            # —— DEBUG INFO ——
            print(f"DEBUG raw line repr: {raw!r}")
            parts = [p.strip() for p in raw.split(',')]
            print(f"DEBUG split parts: {parts}")

            if len(parts) != 2:
                print("→ unexpected format, skipping\n")
                continue

            # —— PARSE ——
            try:
                temp    = float(parts[0])
                press   = float(parts[1])
            except ValueError:
                print("→ conversion failed, skipping\n")
                continue

            print(f"Parsed → Temp: {temp:.2f} °C, Pressure: {press:.2f} Pa\n")

            # —— STORE & PLOT ——
            t = time.time() - start
            times.append(t)
            temps.append(temp)
            pressures.append(press)

            line1.set_data(times, temps)
            line2.set_data(times, pressures)

            ax.set_xlim(times[0], times[-1])
            ymin = min(min(temps), min(pressures)) - 5
            ymax = max(max(temps), max(pressures)) + 5
            ax.set_ylim(ymin, ymax)

            fig.canvas.flush_events()
            plt.pause(0.1)

    except KeyboardInterrupt:
        print("\nInterrupted, exiting…")
    finally:
        ser.close()
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    main()
