package rlnc

import (
	"sync"
	"time"
)

type reseqItem struct {
	seq      uint64
	pkt      []byte
	isPooled bool
}

// ResequencerConfig defines configuration parameters for InOrderResequencer.
type ResequencerConfig struct {
	// MaxWait specifies the maximum duration to wait for a missing sequence number before skipping (default 15ms).
	MaxWait time.Duration

	// MaxPending specifies the maximum capacity of pending out-of-order buffer before forced skip (default 1024).
	MaxPending int

	// ZeroCopy specifies whether emitted packets reuse internal / pooled buffers.
	// When false (default): packets emitted to OnEmit are freshly cloned slices.
	//   The caller owns the memory and can retain, queue, or send across goroutines indefinitely.
	// When true: packets emitted to OnEmit point directly to internal ephemeral or pooled slices.
	//   Strictly 0 heap allocations. Pooled buffers are recycled via PutPacketBuffer immediately
	//   after OnEmit returns; callers must copy if they need to retain the data.
	ZeroCopy bool

	// AllowLateDelivery specifies whether late packets arriving after a gap timeout skip
	// (seq < expectedSeq) should still be delivered to OnEmit (violating strict monotonic sequence ordering).
	// Default is false: late packets arriving behind expectedSeq are counted as stale
	// and discarded to preserve strictly monotonic in-order delivery for TCP/TUN stacks.
	// Set to true only if your higher layer explicitly tolerates or expects out-of-order retrograde packets.
	AllowLateDelivery bool

	// OnEmit is invoked with the packet and its sequence number strictly in order.
	OnEmit func(seq uint64, packet []byte)
}

type pendingPacket struct {
	data     []byte
	isPooled bool
}

// InOrderResequencer guarantees strictly monotonic in-order packet delivery.
// In asymmetric multi-path routing or delayed RLNC Gaussian substitution, packets may arrive with slight jitter.
// By reordering packets before delivering to the higher layer (e.g. TUN device or TCP stack),
// it completely eliminates TCP Duplicate ACKs and prevents congestion window collapse.
// It is safe for concurrent use.
type InOrderResequencer struct {
	mu                sync.Mutex
	maxWait           time.Duration
	maxPending        int
	zeroCopy          bool
	allowLateDelivery bool
	expectedSeq       uint64
	initialized       bool
	pending           map[uint64]pendingPacket
	emittedRing       [2048]uint64
	timer             *time.Timer
	onEmit            func(seq uint64, packet []byte)
	isClosed          bool
	pendingEmit       []reseqItem

	// Telemetry
	delivered    uint64
	reordered    uint64
	gapsSkipped  uint64
	stalePackets uint64
}

// NewInOrderResequencerWithConfig initializes an InOrderResequencer with full configuration options.
func NewInOrderResequencerWithConfig(cfg ResequencerConfig) *InOrderResequencer {
	maxWait := cfg.MaxWait
	if maxWait <= 0 {
		maxWait = 15 * time.Millisecond
	}
	maxPending := cfg.MaxPending
	if maxPending <= 0 {
		maxPending = 1024
	}

	r := &InOrderResequencer{
		maxWait:           maxWait,
		maxPending:        maxPending,
		zeroCopy:          cfg.ZeroCopy,
		allowLateDelivery: cfg.AllowLateDelivery,
		pending:           make(map[uint64]pendingPacket),
		onEmit:            cfg.OnEmit,
		pendingEmit:       make([]reseqItem, 0, 16),
	}
	for i := range r.emittedRing {
		r.emittedRing[i] = ^uint64(0)
	}
	return r
}

// NewInOrderResequencer initializes an InOrderResequencer with safe memory ownership by default.
//   - maxWait: maximum duration to wait for a missing sequence number before skipping (default 15ms).
//   - maxPending: maximum capacity of pending out-of-order buffer before forced skip (default 1024).
//   - onEmit: callback invoked with the packet and its sequence number strictly in order.
func NewInOrderResequencer(maxWait time.Duration, maxPending int, onEmit func(seq uint64, packet []byte)) *InOrderResequencer {
	return NewInOrderResequencerWithConfig(ResequencerConfig{
		MaxWait:           maxWait,
		MaxPending:        maxPending,
		ZeroCopy:          false,
		AllowLateDelivery: false,
		OnEmit:            onEmit,
	})
}

// Push ingests a decoded packet. If it matches expectedSeq, it is emitted immediately on the 0-delay fast path.
// If it arrived out of order, it is buffered in the pool until the gap is filled or maxWait expires.
func (r *InOrderResequencer) Push(seq uint64, packet []byte) {
	r.mu.Lock()
	if r.isClosed {
		r.mu.Unlock()
		return
	}

	r.pendingEmit = r.pendingEmit[:0]

	if !r.initialized {
		if seq <= 128 {
			// Normal stream start: expect packets starting at 0
			r.expectedSeq = 0
		} else {
			// Reconnection mid-stream: synchronize to initial packet
			r.expectedSeq = seq
		}
		r.initialized = true
	} else if r.delivered == 0 && seq < r.expectedSeq {
		r.expectedSeq = seq
	}

	diff := int64(seq - r.expectedSeq)

	// Detect remote sequence aberrant forward jump (e.g. sequence number discontinuity > 100,000)
	if diff > 100000 {
		r.drainAllPendingLocked()
		r.expectedSeq = seq
		diff = 0
	}

	if diff == 0 {
		// Zero-delay fast-path: packet is strictly in order.
		r.pendingEmit = append(r.pendingEmit, reseqItem{seq: seq, pkt: packet, isPooled: false})
		r.emittedRing[seq%2048] = seq
		r.expectedSeq++
		r.delivered++
		r.drainConsecutiveLocked()
	} else if diff < 0 {
		// Late packet arriving after a gap was already skipped (seq < expectedSeq).
		// By default (AllowLateDelivery: false), retrograde packets are discarded to guarantee
		// strictly monotonic in-order delivery without TCP duplicate ACKs or congestion window collapse.
		if r.allowLateDelivery && diff > -1024 && r.emittedRing[seq%2048] != seq {
			r.pendingEmit = append(r.pendingEmit, reseqItem{seq: seq, pkt: packet, isPooled: false})
			r.emittedRing[seq%2048] = seq
			r.delivered++
		} else {
			r.stalePackets++
		}
	} else {
		// diff > 0: future packet arrived out of order.
		if _, exists := r.pending[seq]; !exists {
			var buf []byte
			var isPooled bool
			if len(packet) <= MaxPacketBufferSize {
				buf = GetPacketBuffer()
				n := copy(buf, packet)
				buf = buf[:n]
				isPooled = true
			} else {
				buf = make([]byte, len(packet))
				copy(buf, packet)
				isPooled = false
			}
			r.pending[seq] = pendingPacket{data: buf, isPooled: isPooled}
			r.reordered++

			if len(r.pending) >= r.maxPending {
				// Buffer limit reached: force skip to lowest pending sequence
				r.gapsSkipped++
				r.skipToLowestPendingLocked()
			} else if r.timer == nil {
				// Ensure timeout timer is running
				r.timer = time.AfterFunc(r.maxWait, r.onTimeout)
			}
		}
	}

	var stackEmit [16]reseqItem
	var toEmit []reseqItem
	nEmit := len(r.pendingEmit)
	if nEmit > 0 {
		if nEmit <= len(stackEmit) {
			copy(stackEmit[:], r.pendingEmit)
			toEmit = stackEmit[:nEmit]
		} else {
			toEmit = make([]reseqItem, nEmit)
			copy(toEmit, r.pendingEmit)
		}
	}
	r.mu.Unlock()

	// Invoke callback outside mutex lock and return buffers to pool
	for i := range toEmit {
		if r.onEmit != nil {
			var out []byte
			if r.zeroCopy {
				out = toEmit[i].pkt
			} else {
				out = make([]byte, len(toEmit[i].pkt))
				copy(out, toEmit[i].pkt)
			}
			r.onEmit(toEmit[i].seq, out)
		}
		if toEmit[i].isPooled {
			PutPacketBuffer(toEmit[i].pkt)
		}
	}
}

func (r *InOrderResequencer) onTimeout() {
	r.mu.Lock()
	if r.isClosed || len(r.pending) == 0 {
		r.timer = nil
		r.mu.Unlock()
		return
	}

	r.pendingEmit = r.pendingEmit[:0]
	r.gapsSkipped++
	r.skipToLowestPendingLocked()

	var stackEmit [16]reseqItem
	var toEmit []reseqItem
	nEmit := len(r.pendingEmit)
	if nEmit > 0 {
		if nEmit <= len(stackEmit) {
			copy(stackEmit[:], r.pendingEmit)
			toEmit = stackEmit[:nEmit]
		} else {
			toEmit = make([]reseqItem, nEmit)
			copy(toEmit, r.pendingEmit)
		}
	}
	r.mu.Unlock()

	for i := range toEmit {
		if r.onEmit != nil {
			var out []byte
			if r.zeroCopy {
				out = toEmit[i].pkt
			} else {
				out = make([]byte, len(toEmit[i].pkt))
				copy(out, toEmit[i].pkt)
			}
			r.onEmit(toEmit[i].seq, out)
		}
		if toEmit[i].isPooled {
			PutPacketBuffer(toEmit[i].pkt)
		}
	}
}

func (r *InOrderResequencer) skipToLowestPendingLocked() {
	var minSeq uint64
	var found bool

	for seq := range r.pending {
		if !found || int64(seq-minSeq) < 0 {
			minSeq = seq
			found = true
		}
	}

	if found {
		r.expectedSeq = minSeq
		r.drainConsecutiveLocked()
	}

	if len(r.pending) > 0 && !r.isClosed {
		if r.timer != nil {
			r.timer.Stop()
		}
		r.timer = time.AfterFunc(r.maxWait, r.onTimeout)
	} else {
		if r.timer != nil {
			r.timer.Stop()
			r.timer = nil
		}
	}
}

func (r *InOrderResequencer) drainConsecutiveLocked() {
	for {
		item, exists := r.pending[r.expectedSeq]
		if !exists {
			break
		}
		delete(r.pending, r.expectedSeq)
		r.pendingEmit = append(r.pendingEmit, reseqItem{
			seq:      r.expectedSeq,
			pkt:      item.data,
			isPooled: item.isPooled,
		})
		r.emittedRing[r.expectedSeq%2048] = r.expectedSeq
		r.expectedSeq++
		r.delivered++
	}

	if len(r.pending) == 0 && r.timer != nil {
		r.timer.Stop()
		r.timer = nil
	}
}

func (r *InOrderResequencer) drainAllPendingLocked() {
	if r.timer != nil {
		r.timer.Stop()
		r.timer = nil
	}
	for len(r.pending) > 0 {
		r.skipToLowestPendingLocked()
	}
}

// Close stops running timers and flushes all pending buffered packets.
func (r *InOrderResequencer) Close() {
	r.mu.Lock()
	r.isClosed = true
	r.pendingEmit = r.pendingEmit[:0]
	r.drainAllPendingLocked()

	var stackEmit [16]reseqItem
	var toEmit []reseqItem
	nEmit := len(r.pendingEmit)
	if nEmit > 0 {
		if nEmit <= len(stackEmit) {
			copy(stackEmit[:], r.pendingEmit)
			toEmit = stackEmit[:nEmit]
		} else {
			toEmit = make([]reseqItem, nEmit)
			copy(toEmit, r.pendingEmit)
		}
	}
	r.mu.Unlock()

	for i := range toEmit {
		if r.onEmit != nil {
			var out []byte
			if r.zeroCopy {
				out = toEmit[i].pkt
			} else {
				out = make([]byte, len(toEmit[i].pkt))
				copy(out, toEmit[i].pkt)
			}
			r.onEmit(toEmit[i].seq, out)
		}
		if toEmit[i].isPooled {
			PutPacketBuffer(toEmit[i].pkt)
		}
	}
}

// Reset clears all pending records and resets the expected sequence.
func (r *InOrderResequencer) Reset() {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.timer != nil {
		r.timer.Stop()
		r.timer = nil
	}
	for _, item := range r.pending {
		if item.isPooled {
			PutPacketBuffer(item.data)
		}
	}
	r.pending = make(map[uint64]pendingPacket)
	for i := range r.emittedRing {
		r.emittedRing[i] = ^uint64(0)
	}
	r.expectedSeq = 0
	r.initialized = false
	r.delivered = 0
	r.reordered = 0
	r.gapsSkipped = 0
	r.stalePackets = 0
}

// StalePackets returns the total number of retrograde packets discarded because they arrived after a gap was skipped.
func (r *InOrderResequencer) StalePackets() uint64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.stalePackets
}

// Stats returns resequencer telemetry: delivered packets, reordered packets, skipped gaps,
// and current pending count.
func (r *InOrderResequencer) Stats() (delivered, reordered, skipped uint64, pending int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.delivered, r.reordered, r.gapsSkipped, len(r.pending)
}
