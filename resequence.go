package rlnc

import (
	"sync"
	"time"
)

// InOrderResequencer guarantees strictly monotonic in-order packet delivery.
// In asymmetric multi-path routing or delayed RLNC Gaussian substitution, packets may arrive with slight jitter.
// By reordering packets before delivering to the higher layer (e.g. TUN device or TCP stack),
// it completely eliminates TCP Duplicate ACKs and prevents congestion window collapse.
// It is safe for concurrent use.
type InOrderResequencer struct {
	mu          sync.Mutex
	maxWait     time.Duration
	maxPending  int
	expectedSeq uint64
	initialized bool
	pending     map[uint64][]byte
	emittedRing [2048]uint64
	timer       *time.Timer
	onEmit      func(seq uint64, packet []byte)
	isClosed    bool

	// Telemetry
	delivered   uint64
	reordered   uint64
	gapsSkipped uint64
}

// NewInOrderResequencer initializes an InOrderResequencer.
//   - maxWait: maximum duration to wait for a missing sequence number before skipping (default 15ms).
//   - maxPending: maximum capacity of pending out-of-order buffer before forced skip (default 1024).
//   - onEmit: callback invoked with the packet and its sequence number strictly in order.
func NewInOrderResequencer(maxWait time.Duration, maxPending int, onEmit func(seq uint64, packet []byte)) *InOrderResequencer {
	if maxWait <= 0 {
		maxWait = 15 * time.Millisecond
	}
	if maxPending <= 0 {
		maxPending = 1024
	}

	r := &InOrderResequencer{
		maxWait:    maxWait,
		maxPending: maxPending,
		pending:    make(map[uint64][]byte),
		onEmit:     onEmit,
	}
	for i := range r.emittedRing {
		r.emittedRing[i] = ^uint64(0)
	}
	return r
}

// Push ingests a decoded packet. If it matches expectedSeq, it is emitted immediately on the 0-delay fast path.
// If it arrived out of order, it is buffered in the pool until the gap is filled or maxWait expires.
func (r *InOrderResequencer) Push(seq uint64, packet []byte) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.isClosed {
		return
	}

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
		// Adjust initial sequence if an earlier sequence arrives before any delivery
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
		// Emit immediately without buffer copy.
		if r.onEmit != nil {
			r.onEmit(seq, packet)
		}
		r.emittedRing[seq%2048] = seq
		r.expectedSeq++
		r.delivered++
		r.drainConsecutiveLocked()
		return
	}

	if diff < 0 {
		// Late packet after gap skip: deliver if never previously emitted
		if diff > -1024 && r.emittedRing[seq%2048] != seq {
			if r.onEmit != nil {
				r.onEmit(seq, packet)
			}
			r.emittedRing[seq%2048] = seq
			r.delivered++
		}
		return
	}

	// diff > 0: future packet arrived out of order.
	if _, exists := r.pending[seq]; exists {
		return // Ignore duplicate out-of-order packet without allocating
	}

	// Buffer in pooled slice.
	buf := GetPacketBuffer()
	n := copy(buf, packet)
	r.pending[seq] = buf[:n]
	r.reordered++

	if len(r.pending) >= r.maxPending {
		// Buffer limit reached: force skip to lowest pending sequence
		r.gapsSkipped++
		r.skipToLowestPendingLocked()
		return
	}

	// Ensure timeout timer is running
	if r.timer == nil {
		r.timer = time.AfterFunc(r.maxWait, r.onTimeout)
	}
}

func (r *InOrderResequencer) onTimeout() {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.isClosed || len(r.pending) == 0 {
		r.timer = nil
		return
	}

	r.gapsSkipped++
	r.skipToLowestPendingLocked()
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
		nextPkt, exists := r.pending[r.expectedSeq]
		if !exists {
			break
		}
		delete(r.pending, r.expectedSeq)
		if r.onEmit != nil {
			r.onEmit(r.expectedSeq, nextPkt)
		}
		r.emittedRing[r.expectedSeq%2048] = r.expectedSeq
		PutPacketBuffer(nextPkt)
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
	defer r.mu.Unlock()
	r.isClosed = true
	r.drainAllPendingLocked()
}

// Reset clears all pending records and resets the expected sequence.
func (r *InOrderResequencer) Reset() {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.timer != nil {
		r.timer.Stop()
		r.timer = nil
	}
	for _, pkt := range r.pending {
		PutPacketBuffer(pkt)
	}
	r.pending = make(map[uint64][]byte)
	for i := range r.emittedRing {
		r.emittedRing[i] = ^uint64(0)
	}
	r.expectedSeq = 0
	r.initialized = false
	r.delivered = 0
	r.reordered = 0
	r.gapsSkipped = 0
}

// Stats returns resequencer telemetry: delivered packets, reordered packets, skipped gaps,
// and current pending count.
func (r *InOrderResequencer) Stats() (delivered, reordered, skipped uint64, pending int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.delivered, r.reordered, r.gapsSkipped, len(r.pending)
}
