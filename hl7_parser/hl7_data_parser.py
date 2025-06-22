import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import hl7


source_dir = r"C:/Users/admin/Desktop/phd-workspace/input_data/HL7_raw_data/u130vswir"
dest_dir = r"C:/Users/admin/Desktop/phd-workspace/input_data/output/u130vswir"
os.makedirs(dest_dir, exist_ok=True)


def flatten_list(nested_list):
    result = []
    for row in nested_list:
        result.extend(row)
    return result


def parse_file(file_name):
    message = ""
    msh_detected = False
    messages = []

    with open(file_name, "r", encoding="utf-8") as file:
        while True:
            msg_line = file.readline()
            if not msg_line:
                break

            if msg_line.startswith("Frame timestamp:") or msg_line.startswith("HL7 Data:"):
                continue

            if msg_line[:3] == 'MSH':
                if msh_detected:
                    messages.append(message)
                    message = ""
                msh_detected = True

            message += (msg_line + "\r")

    if message:
        messages.append(message)

    return messages


def get_signals(msg, sig_type='30021'):
    timestamp_value = 0
    times = []
    data = []

    for j in range(len(msg)):
        if str(msg[j][0]) == 'OBX':
            if str(msg[j][2]) == 'TS' and str(msg[j][3])[:5] == sig_type:
                timestamp_value = int(str(msg[j][5]))

            if str(msg[j][2]) == 'NA' and str(msg[j][3])[:5] == sig_type:
                msg_data = str(msg[j][5])
                values = msg_data.split('^')
                values_int = [eval(k) for k in values]
                data.append(values_int)

                if timestamp_value > 0:
                    for _ in range(len(values_int)):
                        times.append(timestamp_value)
                    timestamp_value = 0

    return times, data


def process_hl7_file(file_path, sig_type='30021'):
    messages = parse_file(file_path)
    all_timestamps = []
    all_data = []

    for i in range(len(messages)):
        h = hl7.parse(messages[i])
        ts, sig_data = get_signals(h, sig_type=sig_type)
        all_timestamps.append(ts)
        all_data.append(sig_data)

    ts_flat = flatten_list(all_timestamps)
    data_flat = flatten_list(flatten_list(all_data))

    ts_numpy = np.array(ts_flat, dtype=float)
    data_numpy = np.array(data_flat, dtype=float)

    return ts_numpy, data_numpy


def main():
    for file_name in os.listdir(source_dir):
        if file_name.lower().endswith(".txt"):
            full_path = os.path.join(source_dir, file_name)
            ts_numpy, data_numpy = process_hl7_file(full_path, sig_type='30021')

            print(f"File: {file_name}")
            print("Timestamps shape:", ts_numpy.shape)
            print("Pulse shape:", data_numpy.shape)

            base_name = os.path.splitext(file_name)[0]
            csv_filename = f"{base_name}_pulse.csv"
            csv_path = os.path.join(dest_dir, csv_filename)
            npy_filename = f"{base_name}_pulse.npy"
            npy_path = os.path.join(dest_dir, npy_filename)
            png_filename = f"{base_name}_pulse.png"
            png_path = os.path.join(dest_dir, png_filename)

            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                writer.writerow(["timestamp", "pulse_value"])
                for t, val in zip(ts_numpy, data_numpy):
                    writer.writerow([t, val])

            combined = np.column_stack((ts_numpy, data_numpy))
            np.save(npy_path, combined)

            if data_numpy.size > 0:
                plt.figure()
                plt.plot(data_numpy)
                plt.title(f"Pulse signal for file: {file_name}")
                plt.xlabel("Sample index")
                plt.ylabel("Pulse value")
                plt.savefig(png_path)
                plt.close()

            print(f"Saved CSV, NPY, and PNG for {file_name} in {dest_dir}\n")


if __name__ == "__main__":
    main()
