import serial
import csv
import struct
import time
import numpy as np
from collections import defaultdict, Counter

PORT = '/dev/cu.usbmodem21301'
BAUD = 115200
SIGNAL_LEN = 64

N_PER_CLASS = {0: 1812, 1: 56, 2: 145, 3: 16, 4: 161}

def resample_linear(signal_187):
    return np.interp(
        np.linspace(0, len(signal_187) - 1, SIGNAL_LEN),
        np.arange(len(signal_187)),
        signal_187
    ).astype(np.float32).tolist()

# Raccogli campioni
samples = defaultdict(list)
with open('mitbih_test.csv') as f:
    for row in csv.reader(f):
        label = int(float(row[-1]))
        if len(samples[label]) < N_PER_CLASS[label]:
            signal_187 = [float(x) for x in row[:187]]
            samples[label].append(resample_linear(signal_187))
        if all(len(samples[c]) >= N_PER_CLASS[c] for c in range(5)):
            break

print(f"Campioni raccolti: { {k: len(v) for k, v in samples.items()} }")

with serial.Serial(PORT, BAUD, timeout=30) as ser:
    print("Aspetto READY...")
    while True:
        line = ser.readline().decode(errors='ignore').strip()
        if line == "READY":
            print("Pronta!\n")
            break

    correct = 0
    total = 0
    confusion = defaultdict(Counter)

    for label in sorted(samples.keys()):
        for i, signal in enumerate(samples[label]):
            data = struct.pack(f'<{SIGNAL_LEN}f', *signal)
            ser.write(data)

            response = ''
            deadline = time.time() + 30
            while time.time() < deadline:
                line = ser.readline().decode(errors='ignore').strip()
                if 'CLASS:' in line or 'FAIL' in line:
                    response = line
                    break

            if not response:
                print(f"  TIMEOUT classe {label} campione {i}")
                continue

            pred = int(response.split('CLASS:')[1].split(' ')[0])
            confusion[label][pred] += 1
            if pred == label:
                correct += 1
            total += 1

            if (total % 20) == 0:
                print(f"  Progresso: {total} campioni, accuracy parziale: {100*correct/total:.1f}%")

        time.sleep(0.1)

    print(f"\nAccuracy: {correct}/{total} = {100*correct/total:.1f}%\n")
    print("Confusion matrix (riga=reale, colonna=predetto):")
    print(f"{'':8}", end="")
    for c in range(5):
        print(f"pred{c:1d} ", end="")
    print()
    for real in range(5):
        print(f"real{real:1d}:  ", end="")
        for pred in range(5):
            print(f"{confusion[real][pred]:5d} ", end="")
        print()