"""
Minimal Intelligent-Packet Receiver
==================================

Sniffs for intelligent packets (EtherType 0x88B5) on a given interface and
prints a one-line notification for each packet received. Optionally decodes the
first 8 bytes of payload as model_id (ASCII), matching the sender and switch POC.

Usage
  python pkt_receiver.py --iface eth0
  python pkt_receiver.py --iface eth0 --quiet       # only count
  python pkt_receiver.py --iface eth0 --count 10    # exit after N packets
"""

import argparse
import time
from scapy.all import sniff, Ether, Raw

ETHERTYPE_INTELLIGENT = 0x88B5


def parse_args():
    ap = argparse.ArgumentParser(description="Minimal intelligent-packet sniffer")
    ap.add_argument('--iface', required=True, help='interface to sniff on (e.g., eth0)')
    ap.add_argument('--count', type=int, default=0, help='stop after N packets (0 = infinite)')
    ap.add_argument('--quiet', action='store_true', help='only print a periodic counter')
    ap.add_argument('--show-raw', action='store_true', help='print payload length and first bytes')
    return ap.parse_args()


def on_pkt(pkt, state):
    if Ether not in pkt or pkt[Ether].type != ETHERTYPE_INTELLIGENT:
        return
    ts = time.strftime('%H:%M:%S')
    src = pkt[Ether].src
    dst = pkt[Ether].dst
    model_id = None
    if Raw in pkt and len(bytes(pkt[Raw])) >= 8:
        model_id = bytes(pkt[Raw])[:8].decode(errors='ignore')
    state['count'] += 1
    if not state['quiet']:
        msg = f"[{ts}] intelligent packet #{state['count']} src={src} dst={dst}"
        if model_id is not None:
            msg += f" model_id='{model_id}'"
        print(msg)
        if state['show_raw'] and Raw in pkt:
            raw = bytes(pkt[Raw])
            print(f"    raw_len={len(raw)} preview={raw[:24]!r}")


def main():
    args = parse_args()
    state = {'count': 0, 'quiet': args.quiet, 'show_raw': args.show_raw}
    print(f"Sniffing on {args.iface} for EtherType 0x{ETHERTYPE_INTELLIGENT:04X} ... Press Ctrl+C to stop.")
    try:
        sniff(iface=args.iface, store=0, prn=lambda p: on_pkt(p, state))
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\nTotal intelligent packets: {state['count']}")

if __name__ == '__main__':
    main()
