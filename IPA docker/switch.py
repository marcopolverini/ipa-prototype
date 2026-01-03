#!/usr/bin/env python3
"""
Intelligent Packet Switch (Scapy) – Versione completa con parsing nuovo modello + debug
======================================================================================

Funzionalità:
- Parsing del modello nel formato: [8B model_id][2B alpha][ARCH_HEADER][MODEL_BLOB]
  * ARCH_HEADER: [I:2B][O:2B][L:1B][N1:2B][N2:2B]...[NL:2B]
  * MODEL_BLOB: concat di pesi + bias layer-by-layer
- Decremento TTL manuale
- Gestione DROP come ultima classe
- Stato interfacce UP/DOWN con CLI
- Feature IFACE_STATUS
- Logging dettagliato (ricezione pacchetto, parsing, cache hit/miss, inferenza, forwarding/drop)
"""

import argparse
import logging
import threading
import time
import queue
import struct
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from scapy.all import sniff, sendp, Ether, Raw
from scapy.layers.inet import IP

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# =========================
# Costanti
# =========================
ETHERTYPE_INTELLIGENT = 0x88B5
MAX_TTL = 30.0

FEAT_INGRESS_IFACE_ID = 0x00
FEAT_SWITCH_ID        = 0x01
FEAT_IP_TTL           = 0x02
FEAT_IN_QUEUE_DEPTH   = 0x03
FEAT_OUT_QUEUE_DEPTH  = 0x04
FEAT_PKT_LEN          = 0x05
FEAT_TIMESTAMP_S      = 0x06
FEAT_IFACE_STATUS     = 0x07

# =========================
# Data classes
# =========================
@dataclass
class PacketCtx:
    scapy_pkt: object
    ingress_logical_port: int
    ingress_iface: str
    ts_rx: float
    metadata: dict

@dataclass
class ModelEntry:
    model_id: str
    I: int
    O: int
    hidden: List[int]
    runtime: "TinyMLP"
    last_used: float
    size_bytes: int

# =========================
# TinyMLP runtime
# =========================
class TinyMLP:
    def __init__(self, I: int, O: int, hidden: List[int], weights: List[float]):
        self.I = I
        self.O = O
        self.hidden = hidden[:]
        self.layers = []
        dims = [I] + hidden + [O]
        # Logging layer sizes
        logging.info(f"[DEBUG TinyMLP] Building model with architecture I={I}, hidden={hidden}, O={O}")
        total_params = 0
        for l in range(len(dims) - 1):
            in_dim = dims[l]
            out_dim = dims[l + 1]
            w_count = out_dim * in_dim
            b_count = out_dim
            total_params += w_count + b_count

        logging.info(f"[DEBUG TinyMLP] Total expected params (weights+bias) = {total_params}, weights list length = {len(weights)}")

        if len(weights) < total_params:
            raise ValueError(f"Too few parameters in blob: expected {total_params}, got {len(weights)}")
        elif len(weights) > total_params:
            logging.warning(f"Blob has extra {len(weights) - total_params} values; truncating")

        # Truncate if needed
        weights = weights[:total_params]

        idx = 0
        for l in range(len(dims) - 1):
            in_dim = dims[l]
            out_dim = dims[l + 1]
            w_count = out_dim * in_dim
            b_count = out_dim

            W = weights[idx: idx + w_count]
            idx += w_count
            B = weights[idx: idx + b_count]
            idx += b_count

            self.layers.append((W, B, out_dim, in_dim))
            logging.info(f"[DEBUG TinyMLP] Layer {l}: W shape {out_dim}x{in_dim}, B length {len(B)}")

    def _matvec(self, W: List[float], B: List[float], rows: int, cols: int, x: List[float]) -> List[float]:
        out = []
        for r in range(rows):
            acc = B[r]
            base = r * cols
            for c in range(cols):
                acc += W[base + c] * x[c]
            out.append(acc)
        return out

    @staticmethod
    def _relu(v: List[float]) -> List[float]:
        return [x if x > 0.0 else 0.0 for x in v]

    def infer(self, x: List[float]) -> List[float]:
        logging.info(f"[DEBUG TinyMLP] Running infer; input vector len = {len(x)} (expected {self.I})")
        h = x
        for i, (W, B, rows, cols) in enumerate(self.layers[:-1]):
            h = self._matvec(W, B, rows, cols, h)
            h = self._relu(h)
            logging.debug(f"[DEBUG TinyMLP] After layer {i}, intermediate output len = {len(h)}")
        # last layer
        W, B, rows, cols = self.layers[-1]
        y = self._matvec(W, B, rows, cols, h)
        logging.info(f"[DEBUG TinyMLP] Final output len = {len(y)}")
        return y

# =========================
# HeaderParser
# =========================
class HeaderParser:
    @staticmethod
    def parse_model_and_header(raw: bytes) -> Tuple[str, float, int, int, List[int], List[float]]:
        """
        Parsing del pacchetto:
        - model_id (8B)
        - alpha (2B uint big endian) → float = alpha_enc / 1000
        - ARCH_HEADER: I(2B), O(2B), L(1B), then L×(2B) hidden sizes
        - MODEL_BLOB: pesi+bias quantizzati bytewise (int8 per default)
        Restituisce:
          (model_id, alpha, I, O, hidden_sizes, quantized_weights_bytes)
        """
        if len(raw) < 8 + 2 + 2 + 2 + 1:
            raise ValueError("payload too small to contain mandatory model header")

        model_id = raw[0:8].decode(errors="ignore")
        alpha_enc = struct.unpack(">H", raw[8:10])[0]
        alpha = alpha_enc / 1000.0

        off = 10
        I = struct.unpack(">H", raw[off:off+2])[0]
        off += 2
        O = struct.unpack(">H", raw[off:off+2])[0]
        off += 2
        L = raw[off]
        off += 1

        hidden = []
        for _ in range(L):
            hidden_i = struct.unpack(">H", raw[off:off+2])[0]
            off += 2
            hidden.append(hidden_i)

        logging.info(f"[DEBUG switch] Parsed header: model_id='{model_id}', alpha={alpha:.6f}, I={I}, O={O}, hidden={hidden}")

        # Remaining is blob
        blob = raw[off:]
        logging.info(f"[DEBUG switch] Blob length after header = {len(blob)} bytes")
        return model_id, alpha, I, O, hidden, blob

    @staticmethod
    def decode_weights(blob: bytes, bits: int, alpha: float, I: int, O: int, hidden: List[int]) -> List[float]:
        """
        Decodifica dei pesi quantizzati:
        - bits = 8 o 16
        - mappa valore quantizzato → float: (q / scale) * alpha
        """
        if np is None:
            raise ImportError("numpy manca, necessario per decode_weights")

        scale = (2 ** bits) // 2 - 1

        if bits == 8:
            import array
            arr = array.array("b")
            arr.frombytes(blob)
            q = np.array(arr, dtype=np.float32)
        elif bits == 16:
            import array
            arr = array.array("h")
            arr.frombytes(blob)
            q = np.array(arr, dtype=np.float32)
        else:
            raise ValueError("decode_weights: bits non supportati")

        # Normalizzazione
        w_f = (q / scale) * alpha
        logging.info(f"[DEBUG switch] Decoded weights array length = {len(w_f)}")
        return w_f.tolist()

# =========================
# ModelCache
# =========================
class ModelCache:
    def __init__(self, max_entries: int = 8, max_bytes: int = 128_000_000):
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._bytes = 0
        self._cache: Dict[str, ModelEntry] = {}
        self._lock = threading.Lock()

    def get(self, mid: str) -> Optional[ModelEntry]:
        with self._lock:
            e = self._cache.get(mid)
            if e:
                e.last_used = time.time()
            return e

    def put(self, entry: ModelEntry):
        with self._lock:
            if entry.model_id in self._cache:
                self._bytes -= self._cache[entry.model_id].size_bytes
            self._cache[entry.model_id] = entry
            self._bytes += entry.size_bytes
            # Evizione LRU
            while len(self._cache) > self.max_entries or self._bytes > self.max_bytes:
                victim_id, victim = min(self._cache.items(), key=lambda kv: kv[1].last_used)
                self._bytes -= victim.size_bytes
                del self._cache[victim_id]
                logging.info(f"[DEBUG switch] Evicted model '{victim_id}' from cache")

# =========================
# FeatureRegistry
# =========================
class FeatureRegistry:
    def __init__(self, switch_ref: "Switch"):
        self.sw = switch_ref

    def extract(self, code: int, multiplicity: int, pkt, md: dict) -> List[float]:
        if code == FEAT_INGRESS_IFACE_ID:
            return [float(md["ingress_logical_port"])]
        if code == FEAT_SWITCH_ID:
            return [float(self.sw.switch_id)]
        if code == FEAT_IP_TTL:
            if IP in pkt:
                # assume pkt[IP].ttl exists
                return [float(pkt[IP].ttl) / MAX_TTL]
            else:
                return [0.0]
        if code == FEAT_IN_QUEUE_DEPTH:
            return [float(self.sw.get_ingress_queue(md["ingress_iface"]).qsize())]
        if code == FEAT_OUT_QUEUE_DEPTH:
            return [float(self.sw.out_queue.qsize())]
        if code == FEAT_PKT_LEN:
            # Entire packet length in bytes
            return [float(len(bytes(pkt)))]
        if code == FEAT_TIMESTAMP_S:
            return [float(md.get("ts_rx", time.time()))]
        if code == FEAT_IFACE_STATUS:
            # For dummy/features, produce multiplicity values for each iface and pad zeros
            values: List[float] = []
            ifaces = sorted(self.sw.iface_map.values())
            for i in range(multiplicity):
                if i < len(ifaces):
                    iface = ifaces[i]
                    values.append(1.0 if self.sw.iface_status.get(iface, False) else 0.0)
                else:
                    values.append(0.0)
            return values
        # default zero
        return [0.0] * multiplicity

# =========================
# Ingress, Output & Core Stages
# =========================
class IngressStage(threading.Thread):
    def __init__(self, iface: str, logical_port: int, out_queue: queue.Queue, switch_ref: "Switch"):
        super().__init__(daemon=True)
        self.iface = iface
        self.logical_port = logical_port
        self.out_queue = out_queue
        self.sw = switch_ref

    def _on_pkt(self, pkt):
        if Ether not in pkt or pkt[Ether].type != ETHERTYPE_INTELLIGENT:
            return
        logging.info(f"[Ingress] Packet received on iface '{self.iface}' (logical {self.logical_port})")
        ctx = PacketCtx(pkt, self.logical_port, self.iface, time.time(),
                        {"ingress_logical_port": self.logical_port, "ingress_iface": self.iface})
        try:
            self.out_queue.put_nowait(ctx)
        except queue.Full:
            logging.warning(f"[Ingress] Queue full ({self.iface}), dropping packet")

    def run(self):
        sniff(iface=self.iface, prn=self._on_pkt, store=0, promisc=True)

class OutputStage(threading.Thread):
    def __init__(self, in_queue: queue.Queue, switch_ref: "Switch"):
        super().__init__(daemon=True)
        self.in_q = in_queue
        self.sw = switch_ref

    def run(self):
        while True:
            ctx = self.in_q.get()
            out_iface = ctx.metadata.get("out_iface", None)
            if out_iface is None:
                logging.info("[Output] No out_iface set, dropping packet")
                continue
            if not self.sw.iface_status.get(out_iface, True):
                logging.info(f"[Output] Interface {out_iface} DOWN, dropping packet")
                continue
            logging.info(f"[Output] Forwarding packet via '{out_iface}'")
            try:
                sendp(ctx.scapy_pkt, iface=out_iface, verbose=False)
            except Exception as e:
                logging.warning(f"[Output] sendp failed on iface '{out_iface}': {e}")

class CoreStage(threading.Thread):
    def __init__(self, ingress_queues: Dict[str, queue.Queue], out_queue: queue.Queue, switch_ref: "Switch", allow_pt: bool = False, weight_bits: int = 8):
        super().__init__(daemon=True)
        self.ingress_queues = ingress_queues
        self.out_queue = out_queue
        self.sw = switch_ref
        self.cache = ModelCache()
        self.features = FeatureRegistry(switch_ref)
        self.iface_cycle = list(ingress_queues.keys())
        self.idx = 0
        self.allow_pt = allow_pt
        self.weight_bits = weight_bits

    def _pull_next(self) -> Optional[PacketCtx]:
        for _ in range(len(self.iface_cycle)):
            iface = self.iface_cycle[self.idx]
            self.idx = (self.idx + 1) % len(self.iface_cycle)
            try:
                return self.ingress_queues[iface].get_nowait()
            except queue.Empty:
                continue
        time.sleep(0.0005)
        return None

    def run(self):
        logging.info("[Core] Starting core loop")
        while True:
            ctx = self._pull_next()
            if ctx is None:
                continue
            raw = bytes(ctx.scapy_pkt[Raw]) if Raw in ctx.scapy_pkt else b""
            try:
                model_id, alpha, I, O, hidden, blob = HeaderParser.parse_model_and_header(raw)
                logging.info(f"[Core] Parsed model '{model_id}', alpha={alpha:.6f}, I={I}, O={O}, hidden={hidden}")
                entry = self.cache.get(model_id)
                if entry:
                    logging.info(f"[Core] Cache HIT for model '{model_id}'")
                else:
                    logging.info(f"[Core] Cache MISS for model '{model_id}', reconstructing")
                    # Decode weights
                    weights = HeaderParser.decode_weights(blob, self.weight_bits, alpha, I, O, hidden)
                    # Check expected # of parameters
                    expected = sum((dims_out * dims_in + dims_out) for dims_in, dims_out in zip([I] + hidden, hidden + [O]))
                    logging.info(f"[Core] Expected params = {expected}, got decoded length = {len(weights)}")
                    if len(weights) < expected:
                        logging.error(f"[Core] PARAMETER COUNT MISMATCH: too few parameters, dropping packet")
                        continue
                    # Build model
                    runtime = TinyMLP(I, O, hidden, weights)
                    entry = ModelEntry(model_id, I, O, hidden, runtime, time.time(), len(weights))
                    self.cache.put(entry)

                # TTL decrement
                if IP in ctx.scapy_pkt:
                    old_ttl = ctx.scapy_pkt[IP].ttl
                    new_ttl = max(0, old_ttl - 1)
                    ctx.scapy_pkt[IP].ttl = new_ttl
                    logging.info(f"[Core] TTL: {old_ttl} -> {new_ttl}")

                # Build input vector
                # ** Qui va costruita la lista di features in base a input_header **
                # Per ora mettiamo placeholder zeros
                x = [0.0] * I
                logging.debug(f"[Core] Input vector (placeholder) length = {len(x)}")

                y = entry.runtime.infer(x)

                # Interpret output
                best = max(range(len(y)), key=lambda i: y[i])
                logging.info(f"[Core] Raw infer output: {y}")
                logging.info(f"[Core] Chosen best_idx = {best}")

                # Se l’indice corrisponde all’ultima classe, interpreta come DROP
                if best == len(y) - 1:
                    logging.info(f"[Core] Decision: DROP (best == last)")
                    continue

                out_iface = self.sw.logical_to_iface(best)
                if out_iface is None:
                    logging.info(f"[Core] Decision: no interface mapped for logical {best}, DROP")
                    continue

                if not self.sw.iface_status.get(out_iface, True):
                    logging.info(f"[Core] Decision: interface '{out_iface}' is DOWN, DROP")
                    continue

                ctx.metadata["out_iface"] = out_iface
                logging.info(f"[Core] Forwarding via iface '{out_iface}' (logical {best})")
                self.out_queue.put_nowait(ctx)

            except Exception as e:
                logging.exception(f"[Core] Error processing packet: {e}")

# =========================
# Switch class + CLI
# =========================
class Switch:
    def __init__(self, iface_map: Dict[int, str], switch_id: int = 0, log_level: str = "INFO", allow_pt: bool = False, weight_bits: int = 8):
        self.iface_map = iface_map
        self.rev_map = {v: k for k, v in iface_map.items()}
        self.switch_id = switch_id
        self.allow_pt = allow_pt
        self.weight_bits = weight_bits
        self._configure_logging(log_level)
        self.ingress_queues: Dict[str, queue.Queue] = {iface: queue.Queue(maxsize=4096) for iface in iface_map.values()}
        self.out_queue: queue.Queue = queue.Queue(maxsize=4096)
        self.iface_status: Dict[str, bool] = {iface: True for iface in iface_map.values()}

    def _configure_logging(self, level: str):
        logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                            format="%(asctime)s %(levelname)s %(threadName)s: %(message)s")

    def get_ingress_queue(self, iface: str) -> queue.Queue:
        return self.ingress_queues[iface]

    def logical_to_iface(self, logical: int) -> Optional[str]:
        return self.iface_map.get(logical, None)

    def start(self):
        # Start ingress threads
        for logical, iface in self.iface_map.items():
            t = IngressStage(iface, logical, self.ingress_queues[iface], self)
            t.setName(f"ingress-{iface}")
            t.start()
        # Start core thread
        core = CoreStage(self.ingress_queues, self.out_queue, self, allow_pt=self.allow_pt, weight_bits=self.weight_bits)
        core.setName("core")
        core.start()
        # Start output thread
        out = OutputStage(self.out_queue, self)
        out.setName("egress")
        out.start()
        # CLI loop
        threading.Thread(target=self.cli_loop, daemon=True, name="cli").start()
        logging.info(f"[Switch] Started switch-id={self.switch_id} with ifaces {self.iface_map}")

    def cli_loop(self):
        while True:
            try:
                cmd = input().strip().split()
                if not cmd:
                    continue
                if cmd[0] == "fail" and len(cmd) == 2:
                    iface = cmd[1]
                    if iface in self.iface_status:
                        self.iface_status[iface] = False
                        # clear ingress queue
                        q = self.ingress_queues[iface]
                        with q.mutex:
                            q.queue.clear()
                        logging.info(f"[CLI] Interface '{iface}' marked DOWN (ingress queue cleared)")
                elif cmd[0] == "recover" and len(cmd) == 2:
                    iface = cmd[1]
                    if iface in self.iface_status:
                        self.iface_status[iface] = True
                        logging.info(f"[CLI] Interface '{iface}' marked UP")
                elif cmd[0] == "show":
                    for iface, st in self.iface_status.items():
                        logging.info(f"[CLI] {iface}: {'UP' if st else 'DOWN'}")
                else:
                    logging.info("[CLI] Commands: fail <iface>, recover <iface>, show")
            except EOFError:
                break
            except Exception as e:
                logging.error(f"[CLI] Error in command loop: {e}")

# =========================
# Main
# =========================
def parse_iface_map(pairs: List[str]) -> Dict[int, str]:
    mapping = {}
    for p in pairs:
        if ':' not in p:
            raise ValueError(f"Invalid iface-map entry '{p}', expected 'logical:iface'")
        s, iface = p.split(":", 1)
        mapping[int(s)] = iface
    return mapping

def main():
    parser = argparse.ArgumentParser(description="Switch with debug, new model header")
    parser.add_argument("--iface-map", nargs="+", required=True,
                        help="logical:iface pairs, e.g. 0:eth0 1:eth1")
    parser.add_argument("--switch-id", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--allow-pt", action="store_true")
    parser.add_argument("--weight-bits", type=int, default=8)
    args = parser.parse_args()

    iface_map = parse_iface_map(args.iface_map)
    sw = Switch(iface_map, switch_id=args.switch_id, log_level=args.log_level,
                allow_pt=args.allow_pt, weight_bits=args.weight_bits)
    sw.start()

    while True:
        time.sleep(60)

if __name__ == "__main__":
    main()
