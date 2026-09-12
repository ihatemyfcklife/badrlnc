# go-rlnc

[![Go Reference](https://pkg.go.dev/badge/github.com/vectis-net/rlnc.svg)](https://pkg.go.dev/github.com/vectis-net/rlnc)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Go Version](https://img.shields.io/badge/Go-1.23+-00ADD8?logo=go)](go.mod)
[![Zero Alloc](https://img.shields.io/badge/Allocations-0%20allocs%2Fop-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/Coverage-89.4%25-success.svg)]()

**go-rlnc** is an ultra-high-performance, production-ready Go library implementing **Sliding-Window Random Linear Network Coding (RLNC)** over the Galois Field $\text{GF}(2)$. It features an on-the-fly **incremental Gauss-Jordan solver** with recursive cascade back-substitution, SIMD-accelerated XOR vector reductions, and a jitter buffer resequencer.

Designed for line-rate, low-latency networking (UDP tunnels, multi-path streaming, VPNs, VoIP, real-time gaming, and video broadcast), `go-rlnc` eliminates block boundaries, removes block serialization latency, and recovers lost packets in microseconds.

---

## Table of Contents

- [Why Sliding-Window RLNC?](#why-sliding-window-rlnc)
- [Architecture Overview](#architecture-overview)
- [Key Features](#key-features)
- [Performance & Benchmarks](#performance--benchmarks)
  - [Micro-Benchmarks](#micro-benchmarks)
  - [Erasure Recovery Matrix](#erasure-recovery-matrix)
- [Installation](#installation)
- [Quickstart Guide](#quickstart-guide)
  - [1. Sender: Sliding Window Encoder](#1-sender-sliding-window-encoder)
  - [2. Receiver: Incremental Decoder](#2-receiver-incremental-decoder)
  - [3. Eliminating Jitter: In-Order Resequencer](#3-eliminating-jitter-in-order-resequencer)
  - [4. Wire Serialization (Zero-Alloc)](#4-wire-serialization-zero-alloc)
- [Detailed Component Reference](#detailed-component-reference)
  - [SlidingWindowEncoder](#slidingwindowencoder)
  - [IncrementalDecoder](#incrementaldecoder)
  - [InOrderResequencer](#inorderresequencer)
  - [Wire Protocol & Headers](#wire-protocol--headers)
  - [Buffer Pools & Memory Management](#buffer-pools--memory-management)
  - [Bitset256 & SIMD XOR](#bitset256--simd-xor)
- [Production Recipes & Patterns](#production-recipes--patterns)
  - [Pattern A: Safe Concurrent Pipeline (Default)](#pattern-a-safe-concurrent-pipeline-default)
  - [Pattern B: Line-Rate Zero-Copy (Thread-per-Stream)](#pattern-b-line-rate-zero-copy-thread-per-stream)
  - [Pattern C: Protecting Tail Bursts with `FlushParity`](#pattern-c-protecting-tail-bursts-with-flushparity)
- [Production Deployment Checklist](#production-deployment-checklist)
- [Error Handling](#error-handling)
- [Testing, Fuzzing & Validation](#testing-fuzzing--validation)
- [License](#license)

---

## Why Sliding-Window RLNC?

Traditional Forward Error Correction (FEC) algorithms like **Reed-Solomon** or **RaptorQ** divide packet streams into discrete blocks $[K, N]$:

```
Block FEC (Reed-Solomon / RaptorQ):
[ Packet 1 ][ Packet 2 ] ... [ Packet K ][ Parity 1 ][ Parity 2 ]
|<-------------------- Block Latency -------------------------->|
(Receiver cannot decode missing packets until the entire block arrives)
```

In contrast, **Sliding-Window RLNC** maintains a moving convolution window $W$. Every incoming source packet is immediately emitted systematically ($0\text{ ms}$ delay). Redundant parity shards are generated across the sliding window, and the receiver solves missing packets on-the-fly using incremental **Gauss-Jordan elimination**:

```
Sliding-Window RLNC (go-rlnc):
Incoming: [ P1 ] -> Sent immediately (0 delay)
          [ P2 ] -> Sent immediately (0 delay)
          [ Parity(P1, P2) ] -> Emitted periodically
          [ P3 ] -> Sent immediately (0 delay)
Receiver: Lost P2? Reconstructed instantly upon receiving Parity(P1, P2) in < 1 µs!
```

$$\text{Latency}_{\text{RLNC}} \ll \text{Latency}_{\text{Block FEC}}$$

---

## Architecture Overview

```
 +-----------------------------------------------------------------------------+
 |                              SENDER PIPELINE                                |
 |                                                                             |
 |   Source Packets                                                            |
 |        │                                                                    |
 |        ▼                                                                    |
 |  [ SlidingWindowEncoder ] ───► Systematic Shards (0 delay) ────────┐        |
 |        │                                                           │        |
 |        └─────────────────────► Parity Shards (SIMD XOR) ───────────┤        |
 +--------------------------------------------------------------------┼--------+
                                                                      │ (Wire / UDP)
                                                                      ▼ (Loss & Jitter)
 +-----------------------------------------------------------------------------+
 |                             RECEIVER PIPELINE                               |
 |                                                                             |
 |                              Incoming Wire Shards                           |
 |                                       │                                     |
 |                                       ▼                                     |
 |                           [ IncrementalDecoder ]                            |
 |                                       │                                     |
 |                   Incremental Gauss-Jordan Elimination                      |
 |                   Recursive Cascade Back-Substitution                       |
 |                                       │                                     |
 |                                Decoded Packets                              |
 |                                       │                                     |
 |                                       ▼                                     |
 |                         [ InOrderResequencer ]                              |
 |                                       │                                     |
 |                                       ▼                                     |
 |                       Monotonic In-Order Stream Delivery                    |
 |                       (TUN Device / TCP Stack / App Handler)                |
 +-----------------------------------------------------------------------------+
```

---

## Key Features

- **Decoupled & Standalone:** Operates directly on standard byte slices (`[]byte`) of configurable symbol size. Zero external dependencies.
- **Strictly 0 Heap Allocations (`0 allocs/op`):** Zero allocations on the systematic fast-path, parity generation, wire parsing, and zero-copy decoding.
- **Hardware-Accelerated SIMD XOR:** Achieves **31 to 35+ GB/s** vector XOR throughput using AVX-512, AVX2, and ARM NEON intrinsics via Go's `crypto/subtle`.
- **Incremental Gauss-Jordan Solver:** Solves reduced row echelon form (RREF) continuously upon shard arrival without waiting for block completion.
- **Recursive Cascade Back-Substitution:** When a single packet is solved, all historical dependent equations across the window are resolved immediately in $O(1)$ per pivot.
- **Safe Memory Isolation by Default:** Default configurations clone buffers to ensure complete memory safety across goroutines. Zero-copy mode can be enabled with explicit concurrency contracts.
- **Integrated Anti-Jitter Resequencer:** Monotonic packet ordering buffer that eliminates TCP Duplicate ACKs and manages session restarts cleanly.
- **Wire Integrity with CRC32-Castagnoli:** Hardware-accelerated CRC32 option to detect bit flips and pollution attacks.
- **High Test Coverage (89.4%):** Validated under extreme sequence spaces, fuzz tests, corruption tests, and loss channel simulations.

---

## Performance & Benchmarks

### Micro-Benchmarks

Measured on an AMD Ryzen 5 3600 (6-Core, 3.6 GHz, Windows/AMD64, Go 1.26):

| Operation | Throughput / Rate | Latency | Memory Allocs |
|---|---|---|---|
| **SIMD XOR (1400 B)** | **31,927 MB/s (31.9 GB/s)** | 43.8 ns | **0 B/op, 0 allocs/op** |
| **SIMD XOR Multi-4 (1400 B)** | **31,072 MB/s (31.0 GB/s)** | 180.2 ns | **0 B/op, 0 allocs/op** |
| **Encoder `Push()` (Systematic)** | **13.8 Mpps** | 72.5 ns/op | **0 B/op, 0 allocs/op** |
| **Encoder `GenerateParity()`** | **18.3 Mpps** | 54.6 ns/op | **0 B/op, 0 allocs/op** |
| **End-to-End Systematic Path** | **~700 Kpps** | 1,443 ns/op | **0 B/op, 0 allocs/op** |
| **Incremental RREF Solver** | **3,066 Kpps (3.06 Mpps)** | 326.1 ns/op | **0 B/op, 0 allocs/op** |
| **Bitset256 Discrete Algebra** | **38.9 Mops** | 25.6 ns/op | **0 B/op, 0 allocs/op** |

### Erasure Recovery Matrix

Simulated over an erasure channel ($W=32$, $N=120$ packets) under both uniform and burst loss profiles:

| Channel Loss Rate | Parity Redundancy Ratio | Recovery Rate | Innovation Efficiency |
|---|---|---|---|
| **10% Loss** | 25% Parity (1 parity per 4 data) | **100.0%** | Optimal (100%) |
| **20% Loss** | 50% Parity (1 parity per 2 data) | **100.0%** | Optimal (100%) |
| **30% Loss** | 66% Parity (2 parity per 3 data) | **100.0%** | Optimal (100%) |
| **40% Loss** | 100% Parity (1 parity per 1 data) | **100.0%** | Optimal (100%) |
| **20% Burst Loss (4-packet burst)** | 50% Parity | **100.0%** | Optimal (100%) |

---

## Installation

```bash
go get github.com/vectis-net/rlnc
```

> **Requirements:** Go 1.23 or higher. No CGO required. No external dependencies.

---

## Quickstart Guide

### 1. Sender: Sliding Window Encoder

```go
package main

import (
	"fmt"
	"time"

	"github.com/vectis-net/rlnc"
)

func main() {
	// Create an encoder with default Safe Mode (cloned slices returned)
	enc := rlnc.NewSlidingEncoder(rlnc.EncoderConfig{
		WindowSize:        32,                  // Number of source packets in active sliding window
		SymbolSize:        1400,                // Maximum payload size per packet
		InactivityTimeout: 150 * time.Millisecond,
		ZeroCopy:          false,               // Caller owns emitted shard memory
		Checksum:          true,                // Add hardware CRC32-Castagnoli checksum
	})

	payload := []byte("Hello, High-Performance RLNC!")

	// 1. Emit systematic shard (sent immediately, 0 delay)
	sysShard, err := enc.Push(payload)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Emitted Systematic: Seq=%d, Length=%d\n", sysShard.Seq(), len(sysShard.Data))

	// 2. Proactively or periodically emit parity shards
	parityShard, err := enc.GenerateParity()
	if err != nil {
		panic(err)
	}
	fmt.Printf("Emitted Parity: BaseSeq=%d, Mask=%s\n", parityShard.BaseSeq, parityShard.Mask)

	// 3. When finishing a stream burst, protect the tail packets
	tailShard, err := enc.FlushParity()
	if err == nil {
		fmt.Printf("Emitted Tail Parity: BaseSeq=%d\n", tailShard.BaseSeq)
	}
}
```

### 2. Receiver: Incremental Decoder

```go
package main

import (
	"fmt"

	"github.com/vectis-net/rlnc"
)

func main() {
	// Default Safe Mode: PushShard returns independent heap-allocated cloned slices
	dec := rlnc.NewIncrementalDecoder(rlnc.DecoderConfig{
		Capacity:   2048,
		SymbolSize: 1400,
		ZeroCopy:   false, // Safe mode: recovered packets safe for concurrent goroutines
	})

	// Ingest shards as they arrive over UDP
	recoveredPackets, err := dec.PushShard(shard)
	if err != nil {
		panic(err)
	}

	for _, pkt := range recoveredPackets {
		fmt.Printf("Decoded Packet: %s\n", string(pkt))
	}
}
```

### 3. Eliminating Jitter: In-Order Resequencer

Delayed Gaussian recoveries or multi-path routing can introduce packet jitter. The `InOrderResequencer` ensures packets are delivered strictly monotonically:

```go
package main

import (
	"fmt"
	"time"

	"github.com/vectis-net/rlnc"
)

func main() {
	reseq := rlnc.NewInOrderResequencer(
		15*time.Millisecond, // Max wait before skipping gap
		1024,                // Max pending buffer capacity
		func(seq uint64, packet []byte) {
			// Delivered strictly in monotonic order: 0, 1, 2, 3...
			// Perfect for TUN devices or TCP stacks
			fmt.Printf("Emitted #%d: %d bytes\n", seq, len(packet))
		},
	)
	defer reseq.Close()

	// Push recovered packets (even if out-of-order)
	reseq.Push(seq, packet)
}
```

### 4. Wire Serialization (Zero-Alloc)

`go-rlnc` includes zero-boxing buffer pools for network transmission:

```go
// --- Sender ---
wireBuf := rlnc.GetShardBuffer() // Borrow 2048B buffer from pool
defer rlnc.PutShardBuffer(wireBuf)

n, err := shard.EncodeTo(wireBuf)
if err != nil {
	panic(err)
}
udpConn.Write(wireBuf[:n])

// --- Receiver ---
recvBuf := make([]byte, 2048)
nRead, _ := udpConn.Read(recvBuf)

shard, err := rlnc.DecodeShard(recvBuf[:nRead])
if err != nil {
	// Handles ErrCorruptHeader, ErrInvalidVersion, ErrChecksumMismatch
	return
}
recovered, err := dec.PushShard(shard)
```

---

## Detailed Component Reference

### SlidingWindowEncoder

The `SlidingWindowEncoder` maintains a moving convolution window of $W$ source packets. It is fully thread-safe (`sync.RWMutex`).

#### Configuration (`EncoderConfig`)

| Field | Type | Default | Description |
|---|---|---|---|
| `WindowSize` | `int` | `32` | Number of packets in the active convolution window ($1 \le W \le 256$). |
| `SymbolSize` | `int` | `1400` | Maximum raw payload size in bytes per packet. |
| `InactivityTimeout` | `time.Duration` | `150ms` | Evicts packets older than this duration from parity generation to avoid stale combinations. Set to `< 0` to disable. |
| `DisableInactivityTimeout` | `bool` | `false` | When true, packets stay in window until evicted by window overflow. |
| `Seed` | `uint64` | `0` (Crypto PRNG) | Seed for the SplitMix64 PRNG generating parity coefficients. If `0`, initialized cryptographically. |
| `ZeroCopy` | `bool` | `false` | When `true`, emitted shard data references internal encoder buffers (0 allocs). When `false` (default), copies are returned. |
| `Checksum` | `bool` | `false` | Enables CRC32-Castagnoli checksum generation for wire verification. |

#### Key Methods

- **`Push(payload []byte) (SystematicShard, error)`**: Ingests payload and returns an uncoded systematic shard (sequence number assigned).
- **`GenerateParity() (ParityShard, error)`**: Generates an innovative linear combination over GF(2) across active packets. Respects `InactivityTimeout`.
- **`FlushParity() (ParityShard, error)`**: Generates a parity combination across active packets **ignoring** `InactivityTimeout`. Essential for protecting the tail of a burst.
- **`WindowState() (baseSeq, latestSeq uint64, count int)`**: Returns the current sliding window state.
- **`Reset()`**: Resets sequence numbers and clears all internal buffers.

---

### IncrementalDecoder

The `IncrementalDecoder` performs on-the-fly Gauss-Jordan elimination. When a packet is decoded, it immediately triggers recursive cascade back-substitution.

#### Configuration (`DecoderConfig`)

| Field | Type | Default | Description |
|---|---|---|---|
| `Capacity` | `int` | `2048` | Circular slot capacity for pivots and solved records. |
| `SymbolSize` | `int` | `1400` | Maximum payload size in bytes per packet. |
| `ZeroCopy` | `bool` | `false` | Enables zero-allocation delivery via `OnDecoded`. Requires `OnDecoded != nil` (panics otherwise). |
| `OnDecoded` | `func(seq, []byte)` | `nil` | Callback executed immediately when a packet is recovered. Always executed outside locks. |

#### Safe Mode vs Zero-Copy Mode

```
+------------------------------------------------------------------------------------------+
|                                    DECODER MODES                                         |
+------------------------------------+-----------------------------------------------------+
| Safe Mode (ZeroCopy: false)        | Zero-Copy Mode (ZeroCopy: true)                     |
+------------------------------------+-----------------------------------------------------+
| - Default setting.                 | - Ultra-high performance (0 allocs/op).             |
| - PushShard returns [][]byte copy. | - PushShard returns (nil, nil).                     |
| - Caller owns slice memory.        | - Delivered exclusively via OnDecoded callback.     |
| - Safe across any goroutines.      | - Slice valid ONLY for the duration of callback.    |
| - No concurrency restrictions.     | - Thread-per-stream architecture required.          |
+------------------------------------+-----------------------------------------------------+
```

#### Key Methods

- **`PushShard(shard Shard) ([][]byte, error)`**: Ingests a shard, performs RREF reduction, and returns any recovered packets (in Safe Mode).
- **`PushRawShard(raw []byte) ([][]byte, error)`**: Parses wire bytes and ingests in a single zero-allocation call.
- **`Stats() (received, innovative, redundant, decoded uint64)`**: Returns real-time telemetry metrics.
- **`PivotsEvicted() uint64`**: Returns the count of unresolved pivots overwritten due to ring capacity overflow.

---

### InOrderResequencer

The `InOrderResequencer` reorders packets, absorbs network jitter, and skips unrecoverable gaps after a configurable timeout.

#### Configuration (`ResequencerConfig`)

| Field | Type | Default | Description |
|---|---|---|---|
| `MaxWait` | `time.Duration` | `15ms` | Maximum duration to hold future packets waiting for a missing sequence before skipping the gap. |
| `MaxPending` | `int` | `1024` | Maximum pending out-of-order packet capacity before a forced skip occurs. |
| `ZeroCopy` | `bool` | `false` | When `true`, emitted slices reuse internal / pooled buffers. |
| `AllowLateDelivery`| `bool` | `false` | When `true`, packets arriving after a gap timeout are still emitted. When `false` (default), they are dropped to preserve strict monotonicity. |
| `BackwardJumpThreshold` | `int64` | `10000` | Sequence drop magnitude that triggers an automatic session reset. |
| `MaxConsecutiveStale` | `uint64` | `64` | Consecutive retrograde packets before forcing a session reset (detects remote restarts). |
| `InitialSeq` | `*uint64` | `nil` | Explicit starting sequence number (optional). |
| `OnEmit` | `func(seq, []byte)` | Required | Callback invoked with ordered packets outside mutex locks. |

---

### Wire Protocol & Headers

`go-rlnc` provides a compact binary wire format with automatic Bounds Check Elimination (BCE) hints:

#### 1. Compact Header (24 bytes, for windows $W \le 64$):
```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       BaseSeq (uint64)                        |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       Mask[0] (uint64)                        |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|         DataLen (uint16)      | Flags (uint8) | Version(0x01) |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                 CRC32 Checksum / Reserved (uint32)            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         Data Payload ...                      |
```

#### 2. Extended Header (48 bytes, for windows up to $W = 256$):
Sets `FlagExtendedMask (0x02)`. Appends `Mask[1]`, `Mask[2]`, and `Mask[3]` (24 bytes) at offsets `24..47` before payload bytes.

#### Flags:
- `0x01` (`FlagParity`): Indicates a parity combination shard.
- `0x02` (`FlagExtendedMask`): Indicates a 256-bit mask header (48 bytes).
- `0x04` (`FlagChecksum`): Indicates bytes `[20..23]` contain a CRC32-Castagnoli checksum.

---

### Buffer Pools & Memory Management

To eliminate garbage collector pressure in high-throughput network tunnels, `go-rlnc` exports zero-boxing buffer pools:

```go
// Borrow buffer for wire serialization (2048 bytes)
buf := rlnc.GetShardBuffer()
defer rlnc.PutShardBuffer(buf)

// Borrow buffer for packet payloads (2048 bytes)
pktBuf := rlnc.GetPacketBuffer()
defer rlnc.PutPacketBuffer(pktBuf)
```

> **Note on Jumbo Frames:** The built-in pools are sized for `2048` bytes (standard MTU $\le 1500$ + headers). If your network uses jumbo frames ($\ge 9000$ bytes), `InOrderResequencer` automatically allocates standard heap slices without truncation.

---

### Bitset256 & SIMD XOR

The underlying mathematical primitives are fully exported for custom discrete linear algebra:

```go
// 256-bit vector stack algebra over GF(2)
var b1, b2 rlnc.Bitset256
b1.SetBit(0)
b1.SetBit(42)
b2.SetBit(42)
b1.XOR(b2)
fmt.Println(b1.TestBit(42)) // false

// SIMD XOR vector combination (31+ GB/s)
dst := make([]byte, 1400)
src := make([]byte, 1400)
rlnc.XORBytes(dst, src, 1400)
```

---

## Production Recipes & Patterns

### Pattern A: Safe Concurrent Pipeline (Default)

In this pattern, memory is completely isolated. Decoded packets can be dispatched across arbitrary worker pools without synchronization:

```go
decoder := rlnc.NewIncrementalDecoder(rlnc.DecoderConfig{
    Capacity:   2048,
    SymbolSize: 1400,
    ZeroCopy:   false, // Safe mode
})

// Read loop (Goroutine 1)
go func() {
    for {
        n, _ := udpConn.Read(wireBuf)
        recovered, err := decoder.PushRawShard(wireBuf[:n])
        if err != nil {
            continue
        }
        for _, pkt := range recovered {
            // Safely dispatched to worker pool
            workerPool.Submit(pkt)
        }
    }
}()
```

### Pattern B: Line-Rate Zero-Copy (Thread-per-Stream)

For maximum multi-gigabit wire speed, dedicate one decoder per network stream and process packets directly within the callback:

```go
decoder := rlnc.NewIncrementalDecoder(rlnc.DecoderConfig{
    Capacity:   2048,
    SymbolSize: 1400,
    ZeroCopy:   true, // 0 heap allocations
    OnDecoded: func(seq uint64, packet []byte) {
        // Zero-copy: packet references internal solver ring buffer
        // Process synchronously or copy if retaining
        tunInterface.Write(packet)
    },
})

// Reader dedicated to this stream
for {
    n, _ := udpConn.Read(wireBuf)
    _, _ = decoder.PushRawShard(wireBuf[:n])
}
```

### Pattern C: Protecting Tail Bursts with `FlushParity`

Sliding-window codes generate parity across recent packets. When a burst transmission stops, the last few packets would be vulnerable to loss if no more packets arrive. Use `FlushParity()` to seal the burst:

```go
func sendBurst(enc *rlnc.SlidingWindowEncoder, conn net.Conn, packets [][]byte) {
    for _, pkt := range packets {
        shard, _ := enc.Push(pkt)
        sendShard(conn, shard)
    }

    // Protect the tail of the burst before pausing
    for i := 0; i < 3; i++ {
        tailParity, err := enc.FlushParity()
        if err == nil {
            sendShard(conn, tailParity)
        }
    }
}
```

---

## Production Deployment Checklist

Before deploying `go-rlnc` into production, review this configuration checklist:

- [ ] **Window Size ($W$):**
  - **VoIP / Real-time gaming (Ultra-low latency):** Set $W = 16 \dots 32$.
  - **Video streaming / VPN tunnels:** Set $W = 32 \dots 64$.
  - **High-loss / satellite channels:** Set $W = 64 \dots 128$ with Extended Headers.
- [ ] **Parity Redundancy Tuning:**
  - Measure packet loss rate ($L$).
  - Configure parity generation ratio $R > \frac{L}{1 - L}$ to guarantee full decodability.
- [ ] **Tail Burst Protection:**
  - Call [`FlushParity()`](#slidingwindowencoder) whenever packet ingestion pauses or finishes.
- [ ] **Zero-Copy Concurrency:**
  - If using `ZeroCopy: true`, verify you are running **one decoder per network stream/worker** to prevent buffer overwriting during callbacks.
- [ ] **Checksum Verification:**
  - On public or untrusted networks, set `Checksum: true` to prevent pollution attacks and corrupted packets from poisoning the Gauss-Jordan matrix.

---

## Error Handling

All sentinel errors returned by `go-rlnc` can be matched via `errors.Is`:

| Error | Description | Recommended Handling |
|---|---|---|
| `ErrZeroPayload` | Payload passed to `Push` is empty or window is idle. | Ignore or log; do not transmit. |
| `ErrPayloadTooLarge` | Payload exceeds configured `SymbolSize`. | Fragment packet before encoding. |
| `ErrCorruptHeader` | Wire header is truncated or malformed. | Discard invalid wire packet. |
| `ErrInvalidVersion` | Shard header version does not match `0x01`. | Check protocol compatibility. |
| `ErrBufferTooSmall` | Destination slice provided to `EncodeTo` is too small. | Allocate at least `shard.TotalWireSize()`. |
| `ErrChecksumMismatch`| Wire CRC32 does not match payload. | Discard corrupted packet. |
| `ErrLinearlyDependent`| Shard does not provide new innovation (redundant). | Harmless; solver discarded redundant equation. |

---

## Testing, Fuzzing & Validation

### Run Unit Tests
```bash
go test -v ./...
```

### Run Erasure Recovery Simulation Tests
```bash
go test -v -run TestMatrixInversion ./...
```

### Run Benchmarks & Verify Zero Allocations
```bash
go test -bench=. -benchmem -run=^$ ./...
```

### Run Continuous Fuzzing
```bash
go test -fuzz=FuzzDecodeShard -fuzztime=60s ./...
go test -fuzz=FuzzDecoderPushRawShard -fuzztime=60s ./...
```

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
