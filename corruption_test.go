package rlnc

import (
	"math"
	"testing"
)

// TestCorruption_InvalidHeaders verifies that malformed headers are rejected cleanly without panicking.
func TestCorruption_InvalidHeaders(t *testing.T) {
	dec := NewIncrementalDecoder(DecoderConfig{Capacity: 64, SymbolSize: 1400})

	testCases := []struct {
		name string
		raw  []byte
	}{
		{
			name: "Zero-length slice",
			raw:  []byte{},
		},
		{
			name: "Truncated header (12 bytes instead of 24)",
			raw:  make([]byte, 12),
		},
		{
			name: "Exact 23 bytes (1 byte short of CompactHeaderSize)",
			raw:  make([]byte, 23),
		},
		{
			name: "Invalid Protocol Version (0x00)",
			raw: func() []byte {
				b := make([]byte, CompactHeaderSize+32)
				b[19] = 0x00
				return b
			}(),
		},
		{
			name: "Invalid Protocol Version (0xFF)",
			raw: func() []byte {
				b := make([]byte, CompactHeaderSize+32)
				b[19] = 0xFF
				return b
			}(),
		},
		{
			name: "Aberrant DataLen (claims 65535 bytes but buffer is 40 bytes)",
			raw: func() []byte {
				b := make([]byte, CompactHeaderSize+16)
				s := Shard{
					BaseSeq: 100,
					Mask:    NewBitset256FromUint64(1),
					Data:    make([]byte, 50),
				}
				_, _ = s.EncodeTo(b[:CompactHeaderSize])
				b[16] = 0xff
				b[17] = 0xff
				return b
			}(),
		},
		{
			name: "Extended header flag set but truncated (30 bytes instead of 48)",
			raw: func() []byte {
				b := make([]byte, 30)
				b[18] = FlagExtendedMask
				b[19] = ProtocolVersion1
				return b
			}(),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("PANIC on %s: %v", tc.name, r)
				}
			}()

			_, err := dec.PushRawShard(tc.raw)
			if err == nil {
				t.Errorf("expected error for %s, got nil", tc.name)
			}
		})
	}
}

// TestCorruption_BitFlips verifies resilience against random and deterministic bit flips in serialized shards.
func TestCorruption_BitFlips(t *testing.T) {
	enc := NewSlidingEncoder(EncoderConfig{WindowSize: 32, SymbolSize: 1400})
	dec := NewIncrementalDecoder(DecoderConfig{Capacity: 128, SymbolSize: 1400})

	sysShard, err := enc.Push([]byte("BitFlip Resilience Test Payload 1234567890"))
	if err != nil {
		t.Fatal(err)
	}

	buf := make([]byte, sysShard.TotalWireSize())
	n, err := sysShard.EncodeTo(buf)
	if err != nil {
		t.Fatal(err)
	}

	// 1. Bit flips in header fields (must never panic)
	for byteIdx := 0; byteIdx < sysShard.HeaderSize(); byteIdx++ {
		for bit := 0; bit < 8; bit++ {
			corrupted := make([]byte, n)
			copy(corrupted, buf[:n])
			corrupted[byteIdx] ^= (1 << bit)

			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Fatalf("PANIC on header bit flip at byte %d, bit %d: %v", byteIdx, bit, r)
					}
				}()
				_, _ = dec.PushRawShard(corrupted)
			}()
		}
	}

	// 2. Bit flips in payload length prefix
	for bit := 0; bit < 8; bit++ {
		corrupted := make([]byte, n)
		copy(corrupted, buf[:n])
		corrupted[sysShard.HeaderSize()] ^= (1 << bit)

		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("PANIC on payload bit flip at bit %d: %v", bit, r)
				}
			}()
			_, _ = dec.PushRawShard(corrupted)
		}()
	}
}

// TestCorruption_ExtremeMasksAndBaseSeqs tests boundary sequence numbers and masks.
func TestCorruption_ExtremeMasksAndBaseSeqs(t *testing.T) {
	dec := NewIncrementalDecoder(DecoderConfig{Capacity: 128, SymbolSize: 1400})

	testCases := []struct {
		name    string
		baseSeq uint64
		mask    Bitset256
		dataLen int
	}{
		{"Mask=0 (Zero innovation)", 100, Bitset256{}, 32},
		{"BaseSeq=0", 0, NewBitset256FromUint64(1), 32},
		{"BaseSeq=MaxUint64", math.MaxUint64, NewBitset256FromUint64(1), 32},
		{"BaseSeq=MaxUint64 with 64-bit mask", math.MaxUint64 - 30, NewBitset256FromUint64(math.MaxUint64), 32},
		{"All 256 bits set in Mask", 500, Bitset256{math.MaxUint64, math.MaxUint64, math.MaxUint64, math.MaxUint64}, 64},
		{"Single high bit set in Mask (Bit 255)", 1000, NewBitset256(255), 32},
		{"Single low bit set in Mask (Bit 0)", 1000, NewBitset256(0), 32},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("PANIC on %s: %v", tc.name, r)
				}
			}()

			shardData := make([]byte, LengthPrefixSize+tc.dataLen)
			shardData[0] = byte(tc.dataLen >> 8)
			shardData[1] = byte(tc.dataLen)

			s := Shard{
				BaseSeq:  tc.baseSeq,
				Mask:     tc.mask,
				Data:     shardData,
				IsParity: true,
			}

			_, err := dec.PushShard(s)
			t.Logf("[%s] err=%v", tc.name, err)
		})
	}
}

// TestCorruption_DuplicateShards verifies handling of repeated identical shards.
func TestCorruption_DuplicateShards(t *testing.T) {
	enc := NewSlidingEncoder(EncoderConfig{WindowSize: 16, SymbolSize: 1400})
	dec := NewIncrementalDecoder(DecoderConfig{Capacity: 64, SymbolSize: 1400})

	shard, err := enc.Push([]byte("Duplicate payload test packet"))
	if err != nil {
		t.Fatal(err)
	}

	// First push is innovative
	rec1, err := dec.PushShard(shard)
	if err != nil || len(rec1) != 1 {
		t.Fatalf("expected 1 packet on first push: err=%v, rec=%d", err, len(rec1))
	}

	// 100 duplicate pushes must all be non-innovative / redundant without errors or panics
	for i := 0; i < 100; i++ {
		rec, err := dec.PushShard(shard)
		if err != nil {
			t.Fatalf("unexpected error on duplicate %d: %v", i, err)
		}
		if len(rec) != 0 {
			t.Fatalf("duplicate shard should yield 0 recovered packets on iter %d, got %d", i, len(rec))
		}
	}

	_, _, red, decCount := dec.Stats()
	if decCount != 1 {
		t.Fatalf("expected 1 decoded packet, got %d", decCount)
	}
	if red != 100 {
		t.Fatalf("expected 100 redundant shards, got %d", red)
	}
}
