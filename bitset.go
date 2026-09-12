package rlnc

import (
	"fmt"
	"math/bits"
)

// Bitset256 represents a 256-bit vector for GF(2) discrete linear algebra operations.
// The underlying fixed-size [4]uint64 array guarantees zero heap allocations on the stack.
type Bitset256 [4]uint64

// NewBitset256 creates a Bitset256 with the specified bit indices set to 1.
func NewBitset256(indices ...int) Bitset256 {
	var b Bitset256
	for _, idx := range indices {
		b.SetBit(idx)
	}
	return b
}

// NewBitset256FromUint64 creates a Bitset256 with word 0 set to v and words 1..3 set to 0.
func NewBitset256FromUint64(v uint64) Bitset256 {
	return Bitset256{v, 0, 0, 0}
}

// IsZero returns true if all 256 bits are 0.
func (b Bitset256) IsZero() bool {
	return (b[0] | b[1] | b[2] | b[3]) == 0
}

// Weight returns the Hamming weight (number of set bits) in the 256-bit vector.
func (b Bitset256) Weight() int {
	return bits.OnesCount64(b[0]) +
		bits.OnesCount64(b[1]) +
		bits.OnesCount64(b[2]) +
		bits.OnesCount64(b[3])
}

// TrailingZeros returns the index of the lowest set bit (0..255), or -1 if the bitset is zero.
func (b Bitset256) TrailingZeros() int {
	if b[0] != 0 {
		return bits.TrailingZeros64(b[0])
	}
	if b[1] != 0 {
		return 64 + bits.TrailingZeros64(b[1])
	}
	if b[2] != 0 {
		return 128 + bits.TrailingZeros64(b[2])
	}
	if b[3] != 0 {
		return 192 + bits.TrailingZeros64(b[3])
	}
	return -1
}

// LeadingZeros returns the number of leading zero bits before the highest set bit (0..256).
// If all bits are zero, it returns 256.
func (b Bitset256) LeadingZeros() int {
	if b[3] != 0 {
		return bits.LeadingZeros64(b[3])
	}
	if b[2] != 0 {
		return 64 + bits.LeadingZeros64(b[2])
	}
	if b[1] != 0 {
		return 128 + bits.LeadingZeros64(b[1])
	}
	if b[0] != 0 {
		return 192 + bits.LeadingZeros64(b[0])
	}
	return 256
}

// HighestBit returns the index of the highest set bit (0..255), or -1 if the bitset is zero.
func (b Bitset256) HighestBit() int {
	return 255 - b.LeadingZeros()
}

// SetBit sets bit at index i (0..255) to 1. Out-of-bounds indices are ignored.
func (b *Bitset256) SetBit(i int) {
	if i < 0 || i >= 256 {
		return
	}
	b[i/64] |= (uint64(1) << (i % 64))
}

// ClearBit clears bit at index i (0..255) to 0. Out-of-bounds indices are ignored.
func (b *Bitset256) ClearBit(i int) {
	if i < 0 || i >= 256 {
		return
	}
	b[i/64] &^= (uint64(1) << (i % 64))
}

// TestBit returns true if bit at index i is 1. Out-of-bounds indices return false.
func (b Bitset256) TestBit(i int) bool {
	if i < 0 || i >= 256 {
		return false
	}
	return (b[i/64] & (uint64(1) << (i % 64))) != 0
}

// XOR computes b ^= other in-place over GF(2).
func (b *Bitset256) XOR(other Bitset256) {
	b[0] ^= other[0]
	b[1] ^= other[1]
	b[2] ^= other[2]
	b[3] ^= other[3]
}

// ShiftLeft shifts bits towards higher indices (bit i -> i + shift).
// Bits shifted past index 255 are discarded.
func (b Bitset256) ShiftLeft(shift int) Bitset256 {
	if shift <= 0 {
		return b
	}
	if shift >= 256 {
		return Bitset256{}
	}

	var res Bitset256
	w := shift / 64
	s := shift % 64

	if s == 0 {
		for i := 3; i >= w; i-- {
			res[i] = b[i-w]
		}
		return res
	}

	for i := 3; i >= w; i-- {
		val := b[i-w] << s
		if i-w-1 >= 0 {
			val |= b[i-w-1] >> (64 - s)
		}
		res[i] = val
	}
	return res
}

// ShiftRight shifts bits towards lower indices (bit i -> i - shift).
// Bits shifted past index 0 are discarded.
func (b Bitset256) ShiftRight(shift int) Bitset256 {
	if shift <= 0 {
		return b
	}
	if shift >= 256 {
		return Bitset256{}
	}

	var res Bitset256
	w := shift / 64
	s := shift % 64

	if s == 0 {
		for i := 0; i < 4-w; i++ {
			res[i] = b[i+w]
		}
		return res
	}

	for i := 0; i < 4-w; i++ {
		val := b[i+w] >> s
		if i+w+1 < 4 {
			val |= b[i+w+1] << (64 - s)
		}
		res[i] = val
	}
	return res
}

// Uint64 returns the lower 64 bits of the bitset (word 0).
func (b Bitset256) Uint64() uint64 {
	return b[0]
}

// IsUint64 returns true if words 1, 2, and 3 are zero (the bitset fits entirely within 64 bits).
func (b Bitset256) IsUint64() bool {
	return (b[1] | b[2] | b[3]) == 0
}

// String formats the bitset as a 64-character hexadecimal string representing the 256 bits.
func (b Bitset256) String() string {
	return fmt.Sprintf("0x%016x_%016x_%016x_%016x", b[3], b[2], b[1], b[0])
}
