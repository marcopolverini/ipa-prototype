FROM kathara/base:latest

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Strumenti di sistema e librerie utili per Scapy/sniffing
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    iproute2 iputils-ping ethtool net-tools \
    iptables ebtables tcpdump tshark \
    libpcap0.8 ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# Virtualenv per evitare PEP 668
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Aggiorna pip/setuptools/wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 1) Installa PyTorch CPU dal suo index (solo torch qui)
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch

# 2) Installa Scapy e NumPy da PyPI standard
RUN pip install --no-cache-dir \
    scapy==2.5.0 \
    numpy>=1.26

# Codice (assicurati che questi file siano nel build context)
WORKDIR /opt/intpkt
COPY switch.py ./switch.py
COPY pkt_sender.py ./pkt_sender.py
COPY pkt_receiver.py ./pkt_receiver.py

# Comando di default
CMD ["/bin/bash"]
