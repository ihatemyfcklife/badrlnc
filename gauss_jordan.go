package rlnc

import (
	"encoding/binary"
	"math/bits"
)

// PivotEntry stores a single linear equation in Row Echelon Form over GF(2).
type PivotEntry struct {
	seq    uint64
	mask   Bitset256
	len    int
	data   []byte
	active bool
	solved bool
}

// solvedRecord caches recently reconstructed source packets to eliminate them from incoming shards in O(1).
type solvedRecord struct {
	seq    uint64
	len    int
	data   []byte
	solved bool
}

// processShardLocked incrementally reduces an incoming shard via Gauss-Jordan elimination.
func (d *IncrementalDecoder) processShardLocked(shard Shard) {
	mask := shard.Mask
	payloadLen := len(shard.Data)
	if payloadLen > len(d.scratch) {
		payloadLen = len(d.scratch)
	}

	ClearBytes(d.scratch)
	copy(d.scratch[:payloadLen], shard.Data[:payloadLen])

	// Step 1: Filter out already-solved packets from the incoming shard mask in O(1).
	// We iterate strictly over set bits using word-level trailing zeros for maximum performance.
	for w := 0; w < 4; w++ {
		word := mask[w]
		for word != 0 {
			tz := bits.TrailingZeros64(word)
			bitIdx := w*64 + tz
			targetSeq := shard.BaseSeq + uint64(bitIdx)
			rec := &d.solvedRing[targetSeq%uint64(d.capacity)]
			if rec.solved && rec.seq == targetSeq {
				// Cancel out this solved packet variable over GF(2)
				mask.ClearBit(bitIdx)
				XORBytes(d.scratch, rec.data, rec.len)
				if rec.len > payloadLen {
					payloadLen = rec.len
				}
			}
			word &^= (uint64(1) << tz)
		}
	}

	if mask.IsZero() {
		d.shardsRedundant.Add(1)
		return
	}

	// Step 2: Fast-path for single-variable shards (systematic or fully reduced to 1 variable)
	if mask.Weight() == 1 {
		tz := mask.TrailingZeros()
		targetSeq := shard.BaseSeq + uint64(tz)
		p := &d.pivots[targetSeq%uint64(d.capacity)]

		var carryOverMask Bitset256
		var carryOverLen int
		hasCarryOver := false

		if p.active && p.seq == targetSeq && !p.solved && p.mask.Weight() > 1 && p.mask.TestBit(0) {
			carryOverMask = p.mask
			carryOverMask.ClearBit(0)
			carryOverLen = p.len
			if payloadLen > carryOverLen {
				carryOverLen = payloadLen
			}
			copy(d.carryOverBuf[:p.len], p.data[:p.len])
			if carryOverLen > p.len {
				ClearBytes(d.carryOverBuf[p.len:carryOverLen])
			}
			XORBytes(d.carryOverBuf[:carryOverLen], d.scratch[:payloadLen], payloadLen)
			hasCarryOver = true
		}

		p.active = false
		p.solved = true
		p.seq = targetSeq
		d.shardsInnovative.Add(1)
		d.recordAndEmitSolved(targetSeq, d.scratch[:payloadLen])
		d.backSubstituteSolvedSeq(targetSeq)

		if hasCarryOver {
			d.reduceAndInsert(targetSeq, carryOverMask, d.carryOverBuf[:carryOverLen], carryOverLen)
		}
		return
	}

	// Step 3: Align mask to its lowest variable
	firstBit := mask.TrailingZeros()
	curSeq := shard.BaseSeq + uint64(firstBit)
	curMask := mask.ShiftRight(firstBit)

	// Step 4: Progressive reduction against existing pivots (Row Echelon Form)
	d.reduceAndInsert(curSeq, curMask, d.scratch, payloadLen)
}

// reduceAndInsert reduces curMask and scratch against existing pivots and inserts the resulting pivot.
func (d *IncrementalDecoder) reduceAndInsert(curSeq uint64, curMask Bitset256, scratch []byte, payloadLen int) bool {
	for {
		if curMask.IsZero() {
			d.shardsRedundant.Add(1)
			return false
		}

		tz := curMask.TrailingZeros()
		if tz > 0 {
			curMask = curMask.ShiftRight(tz)
			curSeq += uint64(tz)
		}

		// Eliminate leading variable curSeq if an active pivot already exists
		p := &d.pivots[curSeq%uint64(d.capacity)]
		if p.active && p.seq == curSeq {
			curMask.XOR(p.mask)
			XORBytes(scratch, p.data, p.len)
			if p.len > payloadLen {
				payloadLen = p.len
			}
			continue
		}

		// Eliminate any variables in curMask that are already solved in solvedRing
		for w := 0; w < 4; w++ {
			word := curMask[w]
			if w == 0 {
				word &^= 1 // Skip bit 0 (leading variable)
			}
			for word != 0 {
				t := bits.TrailingZeros64(word)
				k := w*64 + t
				subSeq := curSeq + uint64(k)
				rec := &d.solvedRing[subSeq%uint64(d.capacity)]
				if rec.solved && rec.seq == subSeq {
					curMask.ClearBit(k)
					XORBytes(scratch, rec.data, rec.len)
					if rec.len > payloadLen {
						payloadLen = rec.len
					}
				}
				word &^= (uint64(1) << t)
			}
		}

		// Eliminate any higher variables in curMask using existing subsequent pivots
		for w := 0; w < 4; w++ {
			minBit := 0
			if w == 0 {
				minBit = 1 // Skip bit 0 (leading variable)
			}
			for minBit < 64 {
				var mask uint64
				if minBit > 0 {
					mask = (^uint64(0)) >> (64 - minBit)
				}
				word := curMask[w] &^ mask
				if word == 0 {
					break
				}
				t := bits.TrailingZeros64(word)
				k := w*64 + t
				subSeq := curSeq + uint64(k)
				subP := &d.pivots[subSeq%uint64(d.capacity)]
				if subP.active && subP.seq == subSeq {
					shifted := subP.mask.ShiftLeft(k)
					curMask.XOR(shifted)
					XORBytes(scratch, subP.data, subP.len)
					if subP.len > payloadLen {
						payloadLen = subP.len
					}
				}
				minBit = t + 1
			}
		}

		if curMask.IsZero() {
			d.shardsRedundant.Add(1)
			return false
		}

		newTz := curMask.TrailingZeros()
		if newTz > 0 {
			curMask = curMask.ShiftRight(newTz)
			curSeq += uint64(newTz)
			continue
		}

		// Found empty pivot slot at curSeq
		p = &d.pivots[curSeq%uint64(d.capacity)]
		p.seq = curSeq
		p.mask = curMask
		p.len = payloadLen
		p.active = true
		p.solved = false
		copy(p.data[:payloadLen], scratch[:payloadLen])
		ClearBytes(p.data[payloadLen:])

		d.shardsInnovative.Add(1)

		// Check if newly inserted pivot is fully solved (single variable remaining)
		if p.mask.Weight() == 1 && p.mask.TestBit(0) {
			p.solved = true
			d.recordAndEmitSolved(p.seq, p.data[:p.len])
			d.backSubstituteSolvedSeq(p.seq)
		}

		return true
	}
}

// backSubstituteSolvedSeq substitutes a known solved packet into all older pivots containing it.
// Because the solved packet has ONLY 1 variable, substitution strictly clears bits and never introduces new variables.
func (d *IncrementalDecoder) backSubstituteSolvedSeq(solvedSeq uint64) {
	rec := &d.solvedRing[solvedSeq%uint64(d.capacity)]
	if !rec.solved || rec.seq != solvedSeq {
		return
	}

	lookback := uint64(256)
	if uint64(d.capacity) < lookback {
		lookback = uint64(d.capacity)
	}
	if solvedSeq < lookback {
		lookback = solvedSeq
	}

	for delta := uint64(1); delta <= lookback; delta++ {
		prevSeq := solvedSeq - delta
		prevP := &d.pivots[prevSeq%uint64(d.capacity)]
		if prevP.active && prevP.seq == prevSeq && !prevP.solved && int(delta) < 256 && prevP.mask.TestBit(int(delta)) {
			prevP.mask.ClearBit(int(delta))
			XORBytes(prevP.data, rec.data, rec.len)
			if rec.len > prevP.len {
				prevP.len = rec.len
			}

			// If prevP now has only bit 0 remaining, it has just been solved!
			if prevP.mask.Weight() == 1 && prevP.mask.TestBit(0) {
				prevP.solved = true
				d.recordAndEmitSolved(prevP.seq, prevP.data[:prevP.len])
				// Cascade back-substitution recursively for this newly solved pivot
				d.backSubstituteSolvedSeq(prevP.seq)
			}
		}
	}
}

// recordAndEmitSolved registers the packet into solvedRing, invokes OnDecoded, and buffers the payload.
func (d *IncrementalDecoder) recordAndEmitSolved(seq uint64, data []byte) {
	if len(data) < LengthPrefixSize {
		return
	}

	pktLen := int(binary.BigEndian.Uint16(data[0:LengthPrefixSize]))
	if pktLen <= 0 || pktLen > len(data)-LengthPrefixSize {
		return
	}

	rec := &d.solvedRing[seq%uint64(d.capacity)]
	totalRecordLen := LengthPrefixSize + pktLen
	if totalRecordLen > len(rec.data) {
		return
	}

	rec.seq = seq
	rec.len = totalRecordLen
	rec.solved = true
	copy(rec.data[:totalRecordLen], data[:totalRecordLen])

	d.packetsDecoded.Add(1)

	recoveredPayload := rec.data[LengthPrefixSize:totalRecordLen]
	if d.onDecoded != nil {
		d.onDecoded(seq, recoveredPayload)
	}

	if d.zeroCopy {
		// Zero-allocation: points to internal buffer
		d.recoveredScratch = append(d.recoveredScratch, recoveredPayload)
	} else {
		// Safe clone: caller owns memory
		cp := make([]byte, pktLen)
		copy(cp, recoveredPayload)
		d.recoveredScratch = append(d.recoveredScratch, cp)
	}
}
