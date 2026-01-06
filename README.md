# Intelligent Packet Proof-of-Concept (IPA Prototype)

This repository contains the proof-of-concept (PoC) implementation of the **Intelligent Packet Architecture (IPA)** — a network concept in which packets embed compact machine learning (ML) models to enable inference-based forwarding decisions directly in the data plane.

---

## Requirements

- [Kathara](https://github.com/KatharaFramework/Kathara) network emulator  
- Docker (for building the custom node image)

---

## Build the IPA Docker Image

From the `IPA docker` directory:

```bash
docker build -t my/intpkt:latest -f Dockerfile.Dockerfile .
```
This command creates the Docker image used by all nodes in the Kathara emulated network.

## Running a Test
Two topologies are provided in this repository:

testnet/ – a simple 6-node topology for demonstration

geant/ – a 24-node topology derived from the GEANT network

To run a test, navigate to the desired topology and start the network:

```bash
cd testnet
kathara lstart
```

Once all containers are running, each IPA-enabled node will display a command to launch the software switch.
Execute the provided command in each node terminal (example for node A):

```bash
python3 switch.py --switch-id 0 --log-level INFO --iface-map 0:eth0 1:eth1 2:eth2 3:eth3 --mac 11:22:33:44:55:0A
```

When all switches are running:

Start the receiver on the destination host:

```bash
python3 pkt_receiver.py --iface eth0
```

Send an IPA packet from the source host:

```bash
python3 pkt_sender.py
```

You can press Enter at all prompts to use the default model and parameters.

## Simulating a Link Failure
To disable an interface on any IPA-enabled node, use:

```bash
fail <interface_name>
```

This allows you to test the model’s adaptive behavior under link failures.

## Model Description
The ML model used in this PoC demonstrates IPA-aided Fast Restoration, a use case designed to ensure packet delivery under multiple simultaneous link failures.
A lightweight fully connected neural network is trained offline to emulate the shortest-path routing policy in a hop-by-hop forwarding environment.

## Model I/O
Input features:
- State of the local interfaces
- Ingress interface of the packet
- Normalized TTL value
- Identifier of the current node

Output: 
- Next-hop interface ID or a DROP action

The model is trained in a supervised fashion to reproduce the behavior of a reference LOCAL algorithm, which recomputes shortest paths locally upon encountering link failures.
Each dataset entry corresponds to a node observation during simulated packet traversal, labeled with the next-hop decision selected by the LOCAL policy.

Different combinations of destination node and maximum number of failures (d, K) define distinct missions and corresponding models.
All model weights are quantized to 8 bits and serialized within the IPA header.
