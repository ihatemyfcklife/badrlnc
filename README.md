# go-rlnc

[![Go Reference](https://pkg.go.dev/badge/github.com/vectis-net/rlnc.svg)](https://pkg.go.dev/github.com/vectis-net/rlnc)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Go Version](https://img.shields.io/badge/Go-1.23+-00ADD8?logo=go)](go.mod)
[![Zero Alloc](https://img.shields.io/badge/Allocations-0%20allocs%2Fop-brightgreen.svg)]()

**go-rlnc** is an ultra-high-performance, zero-allocation Go implementation of **Sliding-Window Random Linear Network Coding (RLNC)** over the Galois Field $\text{GF}(2)$, backed by an on-the-fly **incremental Gauss-Jordan solver** with recursive cascade back-substitution.

Designed for line-rate real-time applications (packet streaming, low-latency tunnels, multi-path networking, VoIP, gaming), it eliminates block boundaries, completely removes block synchronization delays, and recovers lost packets with microsecond latency.

---

## Key Features

- **Decoupled & Standalone:** Operates on generic byte slices (`[]byte`) of configurable symbol size. No dependencies on external libraries or specific protocol headers.
- **Strictly 0 Allocations (`0 allocs/op`):** Zero heap allocations on the systematic fast-path and parity generation path.
- **SIMD Hardware Acceleration:** Up to **35+ GB/s** vector XOR throughput utilizing AVX-512, AVX2, and ARM NEON machine intrinsics via `crypto/subtle`.
- **Incremental Gauss-Jordan Solver:** Resolves reduced row echelon form (RREF) continuously. Solves linearly independent packets on-the-fly without waiting for block completion.
- **Recursive Cascade Back-Substitution:** When a single packet is resolved, all historical dependent equations are immediately solved across the window in $O(1)$ per pivot.
- **Exported Utilities:**
  - `Bitset256`: Fast 256-bit vector operations over $\text{GF}(2)$ on the stack.
  - `XOR(dst, a, b)`: High-performance SIMD XOR vector reduction.
  - `InOrderResequencer`: Zero-allocation jitter buffer eliminating TCP Duplicate ACKs.
  - `ShardBufferPool` & `PacketBufferPool`: Optimized zero-boxing buffer pools.

---

## Benchmarks

Measured on an AMD Ryzen 5 3600 (6-Core, 3.6 GHz, Windows/AMD64, Go 1.26.1):

| Benchmark | Throughput / Latency | Memory Allocs |
|---|---|---|
| **SIMD XOR (1400B)** | **31,578 MB/s (31.5 GB/s)** (44.3 ns) | **0 B/op, 0 allocs/op** |
| **SIMD XOR Multi-4 (1400B)** | **35,696 MB/s (35.7 GB/s)** (156.9 ns) | **0 B/op, 0 allocs/op** |
| **Encoder Systematic `Push()`** | **13.3 Mpps** (75.2 ns/op) | **0 B/op, 0 allocs/op** |
| **Encoder `GenerateParity()`** | **27.2 Mpps** (36.7 ns/op) | **0 B/op, 0 allocs/op** |
| **Systematic Fast-Path (Enc + Dec)** | **1,440 ns/op** | **0 B/op, 0 allocs/op** |
| **Incremental RREF Decoding** | **3,273 Kpps (3.27 Mpps)** (305.5 ns/op) | **0 B/op, 0 allocs/op** |
| **Bitset256 Operations** | **26.1 ns/op** | **0 B/op, 0 allocs/op** |

### Erasure Recovery Matrix

Validated across uniform and burst loss profiles ($W=32$, $N=120$ packets):

| Channel Loss Rate | Parity Redundancy | Recovery Rate | Innovation Efficiency |
|---|---|---|---|
| **10% Loss** | 25% Parity | **100.0%** | Optimal |
| **20% Loss** | 50% Parity | **100.0%** | Optimal |
| **30% Loss** | 66% Parity | **100.0%** | Optimal |
| **40% Loss** | 100% Parity | **100.0%** | Optimal |
| **20% Loss (Burst 4-pkts)** | 50% Parity | **100.0%** | Optimal |

---

## Installation

```bash
go get github.com/vectis-net/rlnc
```

*(Requires Go 1.23 or higher, zero external dependencies).*

---

## Quickstart

### 1. Sender (Sliding Window Encoder)

```go
package main

import (
	"fmt"
	"time"

	"github.com/vectis-net/rlnc"
)

func main() {
	encoder := rlnc.NewSlidingEncoder(rlnc.EncoderConfig{
		WindowSize:        32,                  // Number of source packets in active sliding window
		SymbolSize:        1400,                // Maximum payload size per packet
		InactivityTimeout: 150 * time.Millisecond,
	})

	payload := []byte("Hello, High-Performance RLNC!")

	// 1. Emit systematic shard (0 heap allocations)
	sysShard, err := encoder.Push(payload)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Emitted Systematic Shard: Seq=%d, Length=%d\n", sysShard.Seq(), len(sysShard.Data))

	// 2. Proactively or reactively generate redundant parity combinations
	parityShard, err := encoder.GenerateParity()
	if err != nil {
		panic(err)
	}
	// Note: parityShard.Data reuses an internal buffer for 0-alloc performance.
	// Use parityShard.Clone() if storing or queueing multiple parity shards in memory!
	fmt.Printf("Emitted Parity Shard: BaseSeq=%d, Mask=%s\n", parityShard.BaseSeq, parityShard.Mask)
}
```

### 2. Receiver (Incremental Gauss-Jordan Decoder)

```go
package main

import (
	"fmt"

	"github.com/vectis-net/rlnc"
)

func main() {
	// Mode A (Recommended for high-performance streaming pipelines):
	// Use OnDecoded callback for safe, zero-copy packet emission without slice aliasing.
	decoder := rlnc.NewIncrementalDecoder(rlnc.DecoderConfig{
		Capacity:   2048,
		SymbolSize: 1400,
		ZeroCopy:   true,
		OnDecoded: func(seq uint64, packet []byte) {
			fmt.Printf("Streamed Packet #%d: %s\n", seq, string(packet))
		},
	})

	// Shards can be received out of order or after packet drops
	var shard rlnc.Shard
	_, err := decoder.PushShard(shard)
	if err != nil {
		panic(err)
	}

	// Mode B (Batch return via PushShard):
	// - ZeroCopy: false (default): returned [][]byte and its payloads are fresh heap copies.
	// - ZeroCopy: true: returned [][]byte is an internal scratch slice (d.recoveredScratch).
	//   Both the outer slice and its byte buffers are reset on the very next PushShard call!
	//   Process them synchronously within the loop, or use Mode A (OnDecoded) above.
}
```

### 3. Wire Serialization (Zero-Alloc)

```go
// Sender: serialize to wire buffer
wireBuf := rlnc.GetShardBuffer()
defer rlnc.PutShardBuffer(wireBuf)

n, err := shard.EncodeTo(wireBuf)
netConn.Write(wireBuf[:n])

// Receiver: parse directly from wire
shard, err := rlnc.DecodeShard(wireBuf[:n])
recovered, err := decoder.PushShard(shard)
```

### 4. In-Order Resequencer

```go
// Eliminate packet jitter and prevent TCP DupACKs
reseq := rlnc.NewInOrderResequencer(
	15*time.Millisecond, // Max wait before skipping gap
	1024,                // Max pending buffer capacity
	func(seq uint64, packet []byte) {
		// Delivered strictly in monotonic order: 0, 1, 2, 3...
		tunDevice.Write(packet)
	},
)
defer reseq.Close()

// Ingest decoded packets directly
reseq.Push(seq, packet)
```

---

## Architectural Deep Dive

### Sliding-Window vs Block FEC

Traditional FEC codes (Reed-Solomon, RaptorQ) partition streams into discrete blocks $[K, N]$. Receivers cannot decode any packet until the block boundary is reached, incurring significant serialization latency:

$$\text{Latency}_{\text{block}} \propto K \times \text{RTT}$$

**Sliding-Window RLNC** maintains a moving convolution window $W$. Every incoming source packet is immediately emitted systematically ($0$ delay). Parity symbols are generated on-the-fly across the active window. The receiver solves linear combinations incrementally via **Row Echelon Form (RREF)**:

$$\mathbf{M} \cdot \mathbf{x} = \mathbf{y}$$

Where coefficients are computed in the finite field $\text{GF}(2)$ (Boolean XOR algebra).

### Incremental Gauss-Jordan Reduction

1. **$O(1)$ Solved Variable Filtering:** Incoming shards are compared against the circular ring of already reconstructed packets using word-level bit operations (`TrailingZeros64`). Set bits for solved packets are cleared and their payloads subtracted via SIMD XOR.
2. **Systematic Fast-Path:** When a single unknown variable remains in the equation, the packet is solved immediately. If the pivot slot held a pending linear equation, it is preserved via carry-over reduction.
3. **Recursive Cascade Back-Substitution:** Solving sequence $s$ back-substitutes $s$ into all older active pivot equations containing it. If an older equation is reduced to 1 variable, it resolves immediately, triggering a recursive cascade that clears backlogged erasures in a single step.

---

## Wire Format

`go-rlnc` provides a compact, 8-byte aligned binary wire format with bounds check elimination (BCE):

### Compact Header (24 bytes, for windows $\le 64$):
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
|                       Reserved (uint32)                       |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         Data Payload ...                      |
```

### Extended Header (48 bytes, for windows up to 256):
Sets `FlagExtendedMask (0x02)`. Appends `Mask[1]`, `Mask[2]`, and `Mask[3]` (24 bytes) at offsets `24..47` before the payload.

---

## Testing & Validation

Run the complete test suite:
```bash
go test -v .
```

Run erasure matrix simulations under 10%, 20%, 30%, and 40% loss:
```bash
go test -v -run TestMatrixInversion .
```

Run benchmarks and memory allocation profiling:
```bash
go test -bench="." -benchmem
```

---

## License

MIT License. Copyright (c) 2026 Vectis Contributors.
