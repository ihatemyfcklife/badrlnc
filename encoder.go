package badrlnc

import (
	"crypto/rand"
	"encoding/binary"
	"sync"
	"sync/atomic"
	"time"
)

// EncoderConfig defines configuration parameters for SlidingWindowEncoder.
type EncoderConfig struct {
	// WindowSize specifies the maximum number of packets kept in the active encoding window (1..256).
	// Default is DefaultWindowSize (32).
	WindowSize int

	// SymbolSize specifies the maximum raw payload size in bytes per packet.
	// Default is DefaultSymbolSize (1400 bytes).
	SymbolSize int

	// InactivityTimeout is the maximum duration a packet remains eligible for parity combinations.
	// Packets older than this duration are automatically evicted to avoid stale combinations.
	// When left 0, it defaults to 150ms. Set to < 0 or set DisableInactivityTimeout: true to disable.
	InactivityTimeout time.Duration

	// DisableInactivityTimeout explicitly disables inactivity packet eviction, allowing packets
	// in the sliding window to remain eligible indefinitely until evicted by window size overflow.
	DisableInactivityTimeout bool

	// Seed is an optional initial seed for the SplitMix64 PRNG generating parity coefficients.
	// When 0 (default), a cryptographically secure random 64-bit seed is automatically generated.
	Seed uint64

	// ZeroCopy specifies whether emitted Shards reuse internal encoder buffers.
	// When false (default): Push() and GenerateParity() return freshly cloned Data slices.
	//   The caller owns the memory and can retain, queue, or send across goroutines indefinitely.
	// When true: Shard.Data points directly to internal encoder ring and parity buffers.
	//   Strictly 0 heap allocations on Push() and GenerateParity().
	//   Callers must clone or serialize before subsequent encoder calls.
	ZeroCopy bool

	// Checksum enables automatic CRC32-Castagnoli checksum generation for all emitted shards.
	// When enabled, shards are encoded with FlagChecksum (0x04) in wire headers,
	// enabling DecodeShard to detect bit flips and pollution attacks with hardware acceleration.
	Checksum bool

	// InitialSeq specifies the starting sequence number for the first pushed packet.
	// Default is 0. Enables seamless generation-based encoding partitioned by batches.
	InitialSeq uint64
}

type encoderEntry struct {
	seq       uint64
	len       int
	data      []byte
	valid     bool
	timestamp time.Time
}

// SlidingWindowEncoder maintains a continuous sliding window of W packets
// and generates random linear network coding (RLNC) combinations over GF(2) at multi-gigabit wire speed.
// It is fully safe for concurrent use.
type SlidingWindowEncoder struct {
	mu                sync.RWMutex
	windowSize        int
	symbolSize        int
	inactivityTimeout time.Duration
	zeroCopy          bool
	checksum          bool
	entries           []encoderEntry
	nextSeq           uint64
	totalIn           atomic.Uint64
	totalOut          atomic.Uint64
	rngState          uint64
	lastPushTime      time.Time
	parityBuf         []byte
}

// NewSlidingEncoder initializes a SlidingWindowEncoder with the provided configuration.
func NewSlidingEncoder(cfg EncoderConfig) *SlidingWindowEncoder {
	windowSize := cfg.WindowSize
	if windowSize <= 0 || windowSize > MaxExtendedWindowSize {
		windowSize = DefaultWindowSize
	}

	symbolSize := cfg.SymbolSize
	if symbolSize <= 0 {
		symbolSize = DefaultSymbolSize
	}

	var inactivityTimeout time.Duration
	if cfg.DisableInactivityTimeout || cfg.InactivityTimeout < 0 {
		inactivityTimeout = 0
	} else if cfg.InactivityTimeout == 0 {
		inactivityTimeout = 150 * time.Millisecond
	} else {
		inactivityTimeout = cfg.InactivityTimeout
	}

	seed := cfg.Seed
	if seed == 0 {
		var b [8]byte
		if _, err := rand.Read(b[:]); err == nil {
			seed = binary.BigEndian.Uint64(b[:])
		}
		if seed == 0 {
			seed = uint64(time.Now().UnixNano())
		}
	}

	internalSymbolCapacity := LengthPrefixSize + symbolSize

	entries := make([]encoderEntry, windowSize)
	for i := 0; i < windowSize; i++ {
		entries[i] = encoderEntry{
			data: make([]byte, internalSymbolCapacity),
		}
	}

	return &SlidingWindowEncoder{
		windowSize:        windowSize,
		symbolSize:        symbolSize,
		inactivityTimeout: inactivityTimeout,
		zeroCopy:          cfg.ZeroCopy,
		checksum:          cfg.Checksum,
		entries:           entries,
		nextSeq:           cfg.InitialSeq,
		rngState:          seed,
		parityBuf:         make([]byte, internalSymbolCapacity),
	}
}

// nextRand returns a 64-bit pseudo-random number using SplitMix64 with 0 heap allocations.
func (e *SlidingWindowEncoder) nextRand() uint64 {
	e.rngState += 0x9e3779b97f4a7c15
	z := e.rngState
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// Push ingests a raw payload into the sliding window and returns its corresponding SystematicShard.
// Memory Ownership:
//   - Default (ZeroCopy: false): SystematicShard.Data is a freshly allocated clone owned by the caller.
//   - ZeroCopy (ZeroCopy: true): SystematicShard.Data points directly to the encoder's internal circular
//     window buffer (strictly 0 heap allocations). Callers must clone or serialize before subsequent calls.
func (e *SlidingWindowEncoder) Push(payload []byte) (SystematicShard, error) {
	n := len(payload)
	if n == 0 {
		return SystematicShard{}, ErrZeroPayload
	}
	if n > e.symbolSize {
		return SystematicShard{}, ErrPayloadTooLarge
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	seq := e.nextSeq
	e.nextSeq++

	idx := int(seq % uint64(e.windowSize))
	entry := &e.entries[idx]

	totalLen := LengthPrefixSize + n
	binary.BigEndian.PutUint16(entry.data[0:LengthPrefixSize], uint16(n))
	copy(entry.data[LengthPrefixSize:totalLen], payload)

	// Zero out the remaining payload tail to ensure clean GF(2) XOR operations
	if totalLen < len(entry.data) {
		ClearBytes(entry.data[totalLen:])
	}

	entry.seq = seq
	entry.len = totalLen
	entry.valid = true
	entry.timestamp = time.Now()
	e.lastPushTime = entry.timestamp

	e.totalIn.Add(1)
	e.totalOut.Add(1)

	outData := entry.data[:totalLen]
	if !e.zeroCopy {
		cp := make([]byte, totalLen)
		copy(cp, outData)
		outData = cp
	}

	return SystematicShard{
		BaseSeq:  seq,
		Mask:     NewBitset256FromUint64(1),
		Data:     outData,
		IsParity: false,
		Checksum: e.checksum,
	}, nil
}

// GenerateParity produces an innovative GF(2) linear combination across active packets in the sliding window.
// Memory Ownership:
//   - Default (ZeroCopy: false): ParityShard.Data is a freshly allocated clone owned by the caller.
//   - ZeroCopy (ZeroCopy: true): ParityShard.Data points directly to the encoder's internal scratch buffer
//     (strictly 0 heap allocations). Callers must clone or serialize before subsequent calls.
// If the window has been idle longer than InactivityTimeout, ErrZeroPayload is returned.
func (e *SlidingWindowEncoder) GenerateParity() (ParityShard, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.generateParityLocked(false)
}

// FlushParity produces an innovative GF(2) linear combination across all remaining valid packets
// in the sliding window, explicitly ignoring InactivityTimeout.
// This is critical for protecting the tail packets of a transmission burst before entering idle state.
// Memory Ownership follows the configured ZeroCopy setting identical to GenerateParity.
func (e *SlidingWindowEncoder) FlushParity() (ParityShard, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.generateParityLocked(true)
}

func (e *SlidingWindowEncoder) generateParityLocked(ignoreInactivity bool) (ParityShard, error) {
	if e.nextSeq == 0 {
		return ParityShard{}, ErrZeroPayload
	}

	now := time.Now()

	// If no packet was pushed within inactivityTimeout, the window is idle
	if !ignoreInactivity && e.inactivityTimeout > 0 && !e.lastPushTime.IsZero() && now.Sub(e.lastPushTime) > e.inactivityTimeout {
		return ParityShard{}, ErrZeroPayload
	}

	var minSeq uint64
	if e.nextSeq > uint64(e.windowSize) {
		minSeq = e.nextSeq - uint64(e.windowSize)
	} else {
		minSeq = 0
	}

	latestSeq := e.nextSeq - 1

	// Advance minSeq to ignore stale packets older than inactivityTimeout
	if !ignoreInactivity && e.inactivityTimeout > 0 {
		for minSeq < latestSeq {
			idx := int(minSeq % uint64(e.windowSize))
			if e.entries[idx].valid && now.Sub(e.entries[idx].timestamp) <= e.inactivityTimeout {
				break
			}
			minSeq++
		}
	} else {
		for minSeq < latestSeq {
			idx := int(minSeq % uint64(e.windowSize))
			if e.entries[idx].valid && e.entries[idx].seq == minSeq {
				break
			}
			minSeq++
		}
	}

	activeCount := int(latestSeq - minSeq + 1)
	if activeCount <= 0 {
		return ParityShard{}, ErrZeroPayload
	}
	if activeCount > e.windowSize {
		activeCount = e.windowSize
	}

	baseSeq := minSeq

	// Generate mask: if single active packet, mask = 1
	var mask Bitset256
	if activeCount == 1 {
		mask.SetBit(0)
	} else {
		// Fill mask bits up to activeCount using SplitMix64 words
		remaining := activeCount
		wordIdx := 0
		for remaining > 0 {
			r := e.nextRand()
			if remaining >= 64 {
				mask[wordIdx] = r
				remaining -= 64
			} else {
				mask[wordIdx] = r & ((uint64(1) << remaining) - 1)
				remaining = 0
			}
			wordIdx++
		}

		// Ensure the latest packet is ALWAYS included in the parity combination
		mask.SetBit(activeCount - 1)

		// Ensure at least 2 packets are included if activeCount >= 2
		if mask.Weight() < 2 {
			if !mask.TestBit(0) {
				mask.SetBit(0)
			} else if activeCount > 1 && !mask.TestBit(1) {
				mask.SetBit(1)
			}
		}
	}

	ClearBytes(e.parityBuf)
	maxLen := 0

	// Combine all packets selected by the bitmask via SIMD XOR
	for b := 0; b < activeCount; b++ {
		if !mask.TestBit(b) {
			continue
		}
		targetSeq := baseSeq + uint64(b)
		idx := int(targetSeq % uint64(e.windowSize))
		entry := &e.entries[idx]

		if !entry.valid || entry.seq != targetSeq {
			// Clear bit if packet was already evicted
			mask.ClearBit(b)
			continue
		}

		if entry.len > maxLen {
			maxLen = entry.len
		}
		XORBytes(e.parityBuf, entry.data, entry.len)
	}

	if mask.IsZero() || maxLen == 0 {
		return ParityShard{}, ErrZeroPayload
	}

	e.totalOut.Add(1)

	outData := e.parityBuf[:maxLen]
	if !e.zeroCopy {
		cp := make([]byte, maxLen)
		copy(cp, outData)
		outData = cp
	}

	return ParityShard{
		BaseSeq:  baseSeq,
		Mask:     mask,
		Data:     outData,
		IsParity: true,
		Checksum: e.checksum,
	}, nil
}

// WindowState returns the current base sequence, latest sequence, and active packet count.
func (e *SlidingWindowEncoder) WindowState() (baseSeq, latestSeq uint64, count int) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.nextSeq == 0 {
		return 0, 0, 0
	}
	latestSeq = e.nextSeq - 1
	if e.nextSeq > uint64(e.windowSize) {
		baseSeq = e.nextSeq - uint64(e.windowSize)
	} else {
		baseSeq = 0
	}
	count = int(latestSeq - baseSeq + 1)
	return baseSeq, latestSeq, count
}

// ActivePacketCount returns the number of active packets currently retained in the sliding window.
func (e *SlidingWindowEncoder) ActivePacketCount() int {
	_, _, count := e.WindowState()
	return count
}

// WindowSize returns the configured window capacity W.
func (e *SlidingWindowEncoder) WindowSize() int {
	return e.windowSize
}

// Reset clears the sliding window encoder state and resets the sequence counter.
func (e *SlidingWindowEncoder) Reset() {
	e.mu.Lock()
	defer e.mu.Unlock()

	for i := range e.entries {
		e.entries[i].valid = false
		e.entries[i].seq = 0
		e.entries[i].len = 0
	}
	e.nextSeq = 0
	e.lastPushTime = time.Time{}
}

// Stats returns the total count of packets ingested (in) and shards emitted (out).
func (e *SlidingWindowEncoder) Stats() (in, out uint64) {
	return e.totalIn.Load(), e.totalOut.Load()
}
