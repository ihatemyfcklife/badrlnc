package rlnc

import (
	"sync"
	"sync/atomic"
)

// DecoderConfig defines configuration parameters for IncrementalDecoder.
type DecoderConfig struct {
	// Capacity specifies the circular slot capacity for pivots and solved packet records.
	// Default is DefaultCapacity (2048).
	Capacity int

	// SymbolSize specifies the maximum raw payload size in bytes per packet.
	// Default is DefaultSymbolSize (1400 bytes).
	SymbolSize int

	// ZeroCopy enables strictly 0-allocation packet delivery via OnDecoded streaming.
	//
	// Safety & Aliasing Guarantee:
	// To eliminate silent memory corruption and slice aliasing bugs, ZeroCopy mode
	// requires configuring the OnDecoded callback (NewIncrementalDecoder will panic
	// if ZeroCopy is true and OnDecoded is nil).
	//
	// Behavior & Concurrency:
	//   - ZeroCopy: false (default): Recovered packets and the outer slice are safely cloned
	//     into fresh heap allocations and returned from PushShard. Fully safe for concurrent
	//     use across goroutines.
	//   - ZeroCopy: true: Recovered packets are streamed directly to OnDecoded without
	//     allocating slices. PushShard returns (nil, nil) with strictly 0 heap allocations.
	//     Concurrency note: Callbacks are intentionally executed outside the mutex lock to eliminate
	//     lock contention and allow re-entrant decoder operations. In ZeroCopy mode, packet slices
	//     point directly to internal circular buffers and are only valid for the duration of the
	//     callback invocation. If multiple goroutines push shards concurrently to the SAME decoder instance,
	//     external synchronization is recommended to avoid circular ring buffer aliasing on wrap-around.
	//     In high-throughput multi-worker architectures, dedicating one IncrementalDecoder per
	//     network stream/worker is the recommended pattern.
	ZeroCopy bool

	// OnDecoded is invoked immediately when a source packet is recovered.
	// Required when ZeroCopy is true; optional in default safe mode.
	OnDecoded func(seq uint64, packet []byte)
}

// IncrementalDecoder implements an on-the-fly GF(2) linear solver with cascade back-substitution.
// It reconstructs lost source packets immediately upon receiving innovative shards without block delays.
// Internal solver state transitions are fully thread-safe.
type decodedItem struct {
	seq uint64
	pkt []byte
}

type IncrementalDecoder struct {
	mu         sync.Mutex
	capacity   int
	symbolSize int
	zeroCopy   bool
	onDecoded  func(seq uint64, packet []byte)

	pivots           []pivotEntry
	solvedRing       []solvedRecord
	scratch          []byte
	carryOverBuf     []byte
	recoveredScratch [][]byte
	pendingEmit      []decodedItem

	// Telemetry
	shardsReceived   atomic.Uint64
	shardsInnovative atomic.Uint64
	shardsRedundant  atomic.Uint64
	packetsDecoded   atomic.Uint64
	pivotsEvicted    atomic.Uint64
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

	pivots := make([]pivotEntry, capacity)
	solved := make([]solvedRecord, capacity)

	for i := 0; i < capacity; i++ {
		pivots[i].data = make([]byte, internalSymbolCapacity)
		solved[i].data = make([]byte, internalSymbolCapacity)
	}

	if cfg.ZeroCopy && cfg.OnDecoded == nil {
		panic("rlnc: ZeroCopy mode requires configuring OnDecoded callback to prevent slice aliasing")
	}

	return &IncrementalDecoder{
		capacity:         capacity,
		symbolSize:       symbolSize,
		zeroCopy:         cfg.ZeroCopy,
		onDecoded:        cfg.OnDecoded,
		pivots:           pivots,
		solvedRing:       solved,
		scratch:          make([]byte, internalSymbolCapacity),
		carryOverBuf:     make([]byte, internalSymbolCapacity),
		recoveredScratch: make([][]byte, 0, 64),
		pendingEmit:      make([]decodedItem, 0, 16),
	}
}

// PushShard ingests a Shard (systematic or parity), incrementally reduces it via Gauss-Jordan
// elimination over GF(2), cascades back-substitution into older pivots, and returns any packets
// recovered during this step.
//
// Delivery & Memory Modes:
//   - ZeroCopy: false (default): PushShard returns recovered packets as freshly cloned slices ([][]byte).
//     Each packet is an independent heap copy safe to retain, queue, or pass to async workers.
//   - ZeroCopy: true: Strictly 0 heap allocations. Recovered packets are delivered exclusively
//     and immediately through the configured OnDecoded callback. PushShard returns (nil, nil)
//     to prevent any possibility of slice aliasing or use-after-free bugs.
func (d *IncrementalDecoder) PushShard(shard Shard) (recovered [][]byte, err error) {
	if len(shard.Data) == 0 {
		return nil, ErrZeroPayload
	}
	if len(shard.Data) < LengthPrefixSize {
		return nil, ErrCorruptHeader
	}

	d.mu.Lock()
	d.shardsReceived.Add(1)
	if !d.zeroCopy {
		d.recoveredScratch = d.recoveredScratch[:0]
	}
	d.pendingEmit = d.pendingEmit[:0]

	// Process shard through the incremental Gauss-Jordan solver
	d.processShardLocked(shard)

	var stackEmit [16]decodedItem
	var toEmit []decodedItem
	nEmit := len(d.pendingEmit)
	if nEmit > 0 {
		if nEmit <= len(stackEmit) {
			copy(stackEmit[:], d.pendingEmit)
			toEmit = stackEmit[:nEmit]
		} else {
			toEmit = make([]decodedItem, nEmit)
			copy(toEmit, d.pendingEmit)
		}
	}

	var res [][]byte
	if !d.zeroCopy && len(d.recoveredScratch) > 0 {
		res = make([][]byte, len(d.recoveredScratch))
		copy(res, d.recoveredScratch)
	}
	d.mu.Unlock()

	// Invoke callback outside the mutex lock to prevent lock contention and deadlocks
	if d.onDecoded != nil && len(toEmit) > 0 {
		for i := range toEmit {
			d.onDecoded(toEmit[i].seq, toEmit[i].pkt)
		}
	}

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
	d.pivotsEvicted.Store(0)
	d.recoveredScratch = d.recoveredScratch[:0]
}

// PivotsEvicted returns the total number of active, unsolved pivots evicted due to ring buffer capacity wrap-around.
func (d *IncrementalDecoder) PivotsEvicted() uint64 {
	return d.pivotsEvicted.Load()
}

// Stats returns decoder telemetry: total shards received, innovative shards, redundant shards,
// and successfully decoded packets.
func (d *IncrementalDecoder) Stats() (received, innovative, redundant, decoded uint64) {
	return d.shardsReceived.Load(),
		d.shardsInnovative.Load(),
		d.shardsRedundant.Load(),
		d.packetsDecoded.Load()
}
