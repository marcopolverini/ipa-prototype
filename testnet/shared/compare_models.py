#!/usr/bin/env python3
"""
Confronto di due modelli TinyMLP in formato testo
=================================================
Confronta layer per layer (W e B) due dump generati
dal sender (pkt_sender.py) e dallo switch (TinyMLP.dump).
"""

import sys
import numpy as np

import re

def parse_dump(path: str):
    """Parsa un file di dump e restituisce una lista di layer [(W, B)]."""
    layers = []
    W = []
    B = []
    rows, cols = None, None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Layer"):
                if W and B:
                    layers.append((np.array(W, dtype=float), np.array(B, dtype=float)))
                W, B = [], []
                # regex per catturare le dimensioni
                m = re.search(r"W shape=(\d+)x(\d+)", line)
                if m:
                    rows, cols = int(m.group(1)), int(m.group(2))
            elif line.startswith("W["):
                vals = [float(x) for x in line.split(":")[1].split()]
                W.append(vals)
            elif line.startswith("B:"):
                vals = [float(x) for x in line.split(":")[1].split()]
                B = vals
        # ultimo layer
        if W and B:
            layers.append((np.array(W, dtype=float), np.array(B, dtype=float)))
    return layers


def compare_layers(layers1, layers2, tol=1e-3):
    if len(layers1) != len(layers2):
        print(f"[ERROR] Numero di layer diversi: {len(layers1)} vs {len(layers2)}")
        return False
    ok = True
    for i, ((W1, B1), (W2, B2)) in enumerate(zip(layers1, layers2)):
        if W1.shape != W2.shape:
            print(f"[ERROR] Layer {i}: shape diversa {W1.shape} vs {W2.shape}")
            ok = False
            continue
        if B1.shape != B2.shape:
            print(f"[ERROR] Layer {i}: bias length diversa {len(B1)} vs {len(B2)}")
            ok = False
            continue
        diffW = np.max(np.abs(W1 - W2))
        diffB = np.max(np.abs(B1 - B2))
        if diffW > tol or diffB > tol:
            print(f"[WARN] Layer {i}: differenze maggiori della tolleranza {tol}")
            print(f"       max|ΔW|={diffW:.6f}, max|ΔB|={diffB:.6f}")
            ok = False
        else:
            print(f"[OK] Layer {i}: identico entro tolleranza {tol}")
    return ok

def main():
    if len(sys.argv) != 3:
        print(f"Uso: {sys.argv[0]} dump_sender.txt dump_switch.txt")
        sys.exit(1)

    dump1, dump2 = sys.argv[1], sys.argv[2]
    layers1 = parse_dump(dump1)
    layers2 = parse_dump(dump2)

    print(f"[INFO] Parsed {len(layers1)} layer da {dump1}")
    print(f"[INFO] Parsed {len(layers2)} layer da {dump2}")
    res = compare_layers(layers1, layers2)

    if res:
        print("[RESULT] I due modelli coincidono (entro la tolleranza).")
    else:
        print("[RESULT] Sono presenti discrepanze.")

if __name__ == "__main__":
    main()
