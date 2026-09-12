package rlnc

import (
	"sync"
	"sync/atomic"
	"time"
)

// DecoderConfig defines configuration parameters for IncrementalDecoder.
type DecoderConfig struct {
	// Capacity specifies the circular slot capacity for pivots and solved packet records.
	// Default is DefaultCapacity (2048).
	Capacity int

	// SymbolSize specifies the maximum raw payload size in bytes per packet.
	// Default is DefaultSymbolSize (1400 bytes).
	SymbolSize int

	// InactivityTimeout is the maximum duration an unsolved pivot remains active before eviction.
	// Default is 150ms.
	InactivityTimeout time.Duration

	// ZeroCopy enables zero-allocation packet delivery in PushShard.
	//
	// WARNING (Slice & Buffer Aliasing):
	// When true, PushShard returns its internal scratch slice ([][]byte) whose elements point
	// directly to internal decoder ring buffers (0 allocs/op).
	// Crucially, BOTH the returned outer slice ([][]byte) AND its underlying memory are
	// reset and overwritten on the very next call to PushShard.
	//
	// Recommended Alternatives:
	//   1. For event-driven zero-copy streaming, configure OnDecoded(seq, packet) instead.
	//      OnDecoded delivers packets immediately as they are solved without slice aliasing.
	//   2. If using PushShard with ZeroCopy: true, process the returned [][]byte synchronously
	//      before calling PushShard again, or make an explicit copy if retaining across calls.
	//
	// When false (default), each recovered packet and the outer slice are safely cloned into
	// independent heap allocations.
	ZeroCopy bool

	// OnDecoded is an optional callback invoked immediately when a source packet is recovered.
	// Recommended for high-performance streaming pipelines under ZeroCopy mode.
	OnDecoded func(seq uint64, packet []byte)
}

// IncrementalDecoder implements an on-the-fly GF(2) linear solver with cascade back-substitution.
// It reconstructs lost source packets immediately upon receiving innovative shards without block delays.
// It is fully safe for concurrent use.
type IncrementalDecoder struct {
	mu                sync.Mutex
	capacity          int
	symbolSize        int
	inactivityTimeout time.Duration
	zeroCopy          bool
	onDecoded         func(seq uint64, packet []byte)

	pivots           []PivotEntry
	solvedRing       []solvedRecord
	scratch          []byte
	carryOverBuf     []byte
	recoveredScratch [][]byte

	// Telemetry
	shardsReceived   atomic.Uint64
	shardsInnovative atomic.Uint64
	shardsRedundant  atomic.Uint64
	packetsDecoded   atomic.Uint64
}

// NewIncrementalDecoder creates an IncrementalDecoder with the specified configuration.
func NewIncrementalDecoder(cfg DecoderConfig) *IncrementalDecoder {
	capacity := cfg.Capacity
	if capacity <= 0 {
		capacity = DefaultCapacity
	}

	symbolSize := cfg.SymbolSize
	if symbolSize <= 0 {
		symbolSize = DefaultSymbolSize
	}

	internalSymbolCapacity := LengthPrefixSize + symbolSize

	pivots := make([]PivotEntry, capacity)
	solved := make([]solvedRecord, capacity)

	for i := 0; i < capacity; i++ {
		pivots[i].data = make([]byte, internalSymbolCapacity)
		solved[i].data = make([]byte, internalSymbolCapacity)
	}

	return &IncrementalDecoder{
		capacity:          capacity,
		symbolSize:        symbolSize,
		inactivityTimeout: cfg.InactivityTimeout,
		zeroCopy:          cfg.ZeroCopy,
		onDecoded:         cfg.OnDecoded,
		pivots:            pivots,
		solvedRing:        solved,
		scratch:           make([]byte, internalSymbolCapacity),
		carryOverBuf:      make([]byte, internalSymbolCapacity),
		recoveredScratch:  make([][]byte, 0, 64),
	}
}

// PushShard ingests a Shard (systematic or parity), incrementally reduces it via Gauss-Jordan
// elimination over GF(2), cascades back-substitution into older pivots, and returns any packets
// recovered during this step.
//
// Memory Lifetime & ZeroCopy Semantics:
//   - Default (ZeroCopy: false): Both the outer slice ([][]byte) and each packet ([]byte) are
//     freshly allocated copies. Safe to store, queue, or retain indefinitely.
//   - ZeroCopy (ZeroCopy: true): Strictly 0 heap allocations. The returned slice ([][]byte)
//     is an internal reusable scratch buffer (d.recoveredScratch). Its backing array and elements
//     are reset and overwritten on the very next call to PushShard.
//     Do NOT retain the returned [][]byte slice across calls to PushShard!
//     Instead, either process the returned packets synchronously within the same loop,
//     or configure DecoderConfig.OnDecoded for idiomatic, event-driven zero-copy streaming.
func (d *IncrementalDecoder) PushShard(shard Shard) (recovered [][]byte, err error) {
	if len(shard.Data) == 0 {
		return nil, ErrZeroPayload
	}
	if len(shard.Data) < LengthPrefixSize {
		return nil, ErrCorruptHeader
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	d.shardsReceived.Add(1)
	d.recoveredScratch = d.recoveredScratch[:0]

	// Process shard through the incremental Gauss-Jordan solver
	d.processShardLocked(shard)

	if len(d.recoveredScratch) == 0 {
		return nil, nil
	}

	if d.zeroCopy {
		return d.recoveredScratch, nil
	}

	res := make([][]byte, len(d.recoveredScratch))
	copy(res, d.recoveredScratch)
	return res, nil
}

// PushRawShard parses a serialized shard from raw bytes and feeds it to PushShard.
// Wire fast-path: 0 allocations when ZeroCopy is enabled.
func (d *IncrementalDecoder) PushRawShard(raw []byte) (recovered [][]byte, err error) {
	shard, err := DecodeShard(raw)
	if err != nil {
		return nil, err
	}
	return d.PushShard(shard)
}

// Reset clears all pivots, solved records, and telemetry metrics in the decoder.
func (d *IncrementalDecoder) Reset() {
	d.mu.Lock()
	defer d.mu.Unlock()

	for i := range d.pivots {
		d.pivots[i].active = false
		d.pivots[i].solved = false
		d.pivots[i].seq = 0
		d.pivots[i].len = 0
	}
	for i := range d.solvedRing {
		d.solvedRing[i].solved = false
		d.solvedRing[i].seq = 0
		d.solvedRing[i].len = 0
	}

	d.shardsReceived.Store(0)
	d.shardsInnovative.Store(0)
	d.shardsRedundant.Store(0)
	d.packetsDecoded.Store(0)
	d.recoveredScratch = d.recoveredScratch[:0]
}

// Stats returns decoder telemetry: total shards received, innovative shards, redundant shards,
// and successfully decoded packets.
func (d *IncrementalDecoder) Stats() (received, innovative, redundant, decoded uint64) {
	return d.shardsReceived.Load(),
		d.shardsInnovative.Load(),
		d.shardsRedundant.Load(),
		d.packetsDecoded.Load()
}
