#!/usr/bin/env python3
"""
Guided Intelligent-Packet Sender (quantizzazione, pesi+bias, alpha e INPUT_HEADER)
=================================================================================

Formato pacchetto:
[8B model_id ASCII] +
[2B alpha uint16] +
[ARCH_HEADER I(2B) O(2B) L(1B) N(2B)*L] +
[INPUT_HEADER] +
[MODEL_BLOB pesi+bias quantizzati]
"""

import argparse
import os
import sys
import torch
import numpy as np
from scapy.all import Ether, Raw, sendp

ETHERTYPE_INTELLIGENT = 0x88B5

# =========================
# Configurazione default INPUT_HEADER
# =========================
# Modificala a mano per test/debug
DEFAULT_INPUT_SPEC = [(0x07, 4), (0x00, 4), (0x02, 1), (0x01, 6)]

# =========================
# Helpers
# =========================
def pad32(b: bytes) -> bytes:
    pad = (4 - (len(b) % 4)) % 4
    return b + (b"\x00" * pad)

def ensure_value(v, prompt, default=None):
    if v:
        return v
    if default is not None:
        s = input(f"{prompt} [{default}]: ") or default
    else:
        s = input(f"{prompt}: ")
    return s

def build_input_header(spec):
    out = bytes([len(spec)])
    for t, c in spec:
        out += bytes([t & 0xFF, c & 0xFF])
    return pad32(out)

# =========================
# Quantizzazione modello
# =========================
def quantize_model(path: str, bits: int = 8):
    model = torch.load(path, map_location="cpu")
    if hasattr(model, "state_dict"):
        state = model.state_dict()
    else:
        state = model

    levels = 2 ** bits
    scale = (levels // 2) - 1

    # Trova alpha
    all_params = np.concatenate([p.cpu().numpy().astype(np.float32).flatten() for p in state.values()])
    alpha = np.max(np.abs(all_params))
    if alpha == 0:
        alpha = 1.0

    # Architettura
    layers = []
    for key in state.keys():
        if "weight" in key:
            layers.append(state[key].shape[0])

    input_size = state["fc1.weight"].shape[1]
    output_size = state["out.weight"].shape[0]
    num_layers = len(layers)

    print("[DEBUG pkt_sender] Architettura:")
    print(f"  Input={input_size}, Output={output_size}, Layers={num_layers}, Sizes={layers}")
    total_expected = 0

    # Header architettura
    header = bytearray()
    header += int(input_size).to_bytes(2, "big")
    header += int(output_size).to_bytes(2, "big")
    header += int(num_layers).to_bytes(1, "big")
    for n in layers:
        header += int(n).to_bytes(2, "big")

    # Serializzazione pesi + bias
    all_w = []
    for lname in ["fc1", "fc2", "out"]:
        for suffix in ["weight", "bias"]:
            key = f"{lname}.{suffix}"
            if key not in state:
                continue
            arr = state[key].cpu().numpy().astype(np.float32).flatten()
            arr = arr / alpha
            q = np.clip(np.round(arr * scale), -scale, scale).astype(np.int32)

            if bits == 8:
                all_w.append(q.astype(np.int8).tobytes())
            elif bits == 16:
                all_w.append(q.astype(np.int16).tobytes())
            else:
                raise ValueError("Quantizzazione supportata solo a 8 o 16 bit")

            print(f"[DEBUG pkt_sender] {key}: {arr.size} valori")
            total_expected += arr.size

    blob = b"".join(all_w)
    payload = pad32(bytes(header) + blob)

    print(f"[DEBUG pkt_sender] Blob length={len(blob)} bytes (before padding)")
    print(f"[DEBUG pkt_sender] Totale parametri (pesi+bias) attesi={total_expected}")

    print(f"[INFO] Quantizzato modello {path}:")
    print(f"  Alpha={alpha:.6f}, Payload={len(payload)} bytes (bits={bits})")

    return alpha, payload

# =========================
# Argomenti
# =========================
def get_args():
    ap = argparse.ArgumentParser(description="Guided intelligent-packet sender with quantization")
    ap.add_argument("--iface", help="egress interface (e.g., eth0)")
    ap.add_argument("--dst-mac", dest="dst_mac", help="destination MAC")
    ap.add_argument("--model-path", dest="model_path", help=".pt file path")
    ap.add_argument("--model-id", dest="model_id", help="8-char ASCII id")
    ap.add_argument("--weight-bits", type=int, default=8)
    ap.add_argument("--no-wizard", action="store_true")
    return ap.parse_args()

# =========================
# Main
# =========================
def main():
    args = get_args()

    if not args.no_wizard:
        args.iface = ensure_value(args.iface, "Interface to send on", "eth0")
        args.dst_mac = ensure_value(args.dst_mac, "Destination MAC", "02:00:00:00:00:01")
        args.model_path = ensure_value(args.model_path, "Path to .pt model", "model.pt")
        args.model_id = ensure_value(args.model_id, "Model ID (8 ASCII)", "ROUTE001")
    else:
        missing = [k for k in (args.iface, args.dst_mac, args.model_path, args.model_id) if not k]
        if missing:
            print("Missing required args and --no-wizard set.")
            sys.exit(2)

    if not os.path.isfile(args.model_path):
        print(f"Model path not found: {args.model_path}")
        sys.exit(1)

    mid = (args.model_id or "").encode("ascii", errors="ignore")[:8]
    mid = (mid + b"\x00" * 8)[:8]

    alpha, model_blob = quantize_model(args.model_path, args.weight_bits)
    alpha_enc = int(alpha * 1000).to_bytes(2, "big", signed=False)

    input_hdr = build_input_header(DEFAULT_INPUT_SPEC)

    # Ordine corretto: mid + alpha + arch_header+blob + input_hdr
    payload = mid + alpha_enc + model_blob + input_hdr

    frame = Ether(dst=args.dst_mac, type=ETHERTYPE_INTELLIGENT) / Raw(payload)

    print(f"\nSending intelligent packet on {args.iface} to {args.dst_mac}")
    print(f"  model_id: {mid!r}")
    print(f"  alpha: {alpha:.6f} (encoded={int(alpha*1000)})")
    print(f"  input_hdr: {len(input_hdr)} bytes")
    print(f"  total payload: {len(payload)} bytes, bits={args.weight_bits}")

    sendp(frame, iface=args.iface, verbose=False)
    print("Done.")

if __name__ == "__main__":
    main()
