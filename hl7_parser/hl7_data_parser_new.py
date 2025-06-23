#!/usr/bin/env python3
"""
1st spo2 frametime is a time after receivng 1st chunk of data
that's why we need to assume the data was coleceted before receving it, so
timestamp will = timestamp-10ms for the last sample from 1st chunk, and 1st_timestamp-20ms for previous...
next samples will have timestamp = 1st timestamp, timestamp = 1st timestamp +10ms ...

"""
import csv, re, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt


# OBX||CD|30021^SPO2^BHC|1|1^^1^^100^0&250||||||F <-  ^100^ -> 100Hz -> 10ms per sample
DT_MS = 10.0  # 100 Hz

in_path  = Path(r"C:\Users\admin\Desktop\Dataset\2025-06-22_18-45-24.447018\hl7\hl7_raw.txt")
out_path = Path(r"C:\Users\admin\Desktop\Dataset\2025-06-22_18-45-24.447018\hl7\spo2_timebase.csv")

# regexy
re_ts = re.compile(r'^Frame timestamp:\s*(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)')
re_na = re.compile(r'^OBX\|\|NA\|30021\^SPO2\^BHC\|1\|(.+?)\|')

def to_ms(txt):
    dt = datetime.strptime(txt, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

# przebieg: zbior probek + dane listy
values, first_ts_ms, first_len = [], None, None
pending_ts, grab_it = None, False

with in_path.open(encoding="utf-8") as fh:
    for raw in fh:
        line = raw.rstrip("\n")

        if m := re_ts.match(line):
            pending_ts = to_ms(m.group(1))
            grab_it = True
            continue

        if m := re_na.match(line):
            nums = [float(v) for v in m.group(1).split("^") if v]
            values.extend(nums)

            if grab_it and first_ts_ms is None:
                first_ts_ms = pending_ts
                first_len = len(nums)
            grab_it = False

if first_ts_ms is None or first_len is None:
    sys.exit("Nie znaleziono pierwszego Frame timestamp dla SPO2-NA.")
if not values:
    sys.exit("Brak probek SPO2-NA.")

t0_ms = first_ts_ms - first_len * DT_MS          # czas 1 probki
samples = []
for i, val in enumerate(values):
    t_ms = t0_ms + i * DT_MS
    iso = datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc)\
                  .strftime("%Y-%m-%d %H:%M:%S.%f")
    samples.append((t_ms, iso, val))

# zapis CSV
with out_path.open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh, delimiter=";")
    w.writerow(["t_ms", "iso_utc", "spo2"])
    w.writerows(samples)

print(f"Probek SPO2: {len(samples)}  |  zapisano -> {out_path}")


PLOT_N = 100                     # max samples shown on the plot

#NumPy array [t_ms, spo2]  (all samples)
arr = np.array([(t, v) for t, _, v in samples], dtype=float)

npy_path = out_path.with_suffix(".npy")
np.save(npy_path, arr)

#  plot only the first PLOT_N samples
plot_arr = arr[:PLOT_N]           # slice safely handles N < PLOT_N
t_rel_s  = (plot_arr[:, 0] - plot_arr[0, 0]) / 1000.0

png_path = out_path.with_suffix(".png")
plt.figure(figsize=(10, 4))
plt.plot(t_rel_s, plot_arr[:, 1])
plt.title(f"SPO2 waveform - first {len(plot_arr)} samples")
plt.xlabel("Time [s] (relative)")
plt.ylabel("SPO2")
plt.tight_layout()
plt.savefig(png_path, dpi=120)
plt.close()

print(f"NumPy  {npy_path}")
print(f"Plot   {png_path}  (first {len(plot_arr)} samples)")


