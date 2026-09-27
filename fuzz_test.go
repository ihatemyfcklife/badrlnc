package badrlnc

import (
	"bytes"
	"testing"
)

// FuzzDecodeShard verifies that DecodeShard never panics on arbitrary malformed inputs,
// and that valid shards satisfy round-trip serialization and can be ingested safely.
func FuzzDecodeShard(f *testing.F) {
	// Seed corpus with valid compact shard
	compactShard := Shard{
		BaseSeq:  100,
		Mask:     NewBitset256FromUint64(0b101),
		Data:     []byte{0x00, 0x05, 'h', 'e', 'l', 'l', 'o'},
		IsParity: true,
		Checksum: true,
	}
	compactBuf := make([]byte, compactShard.TotalWireSize())
	_, _ = compactShard.EncodeTo(compactBuf)
	f.Add(compactBuf)

	// Seed corpus with valid extended shard
	extShard := Shard{
		BaseSeq:  200,
		Mask:     NewBitset256(1, 70, 150),
		Data:     []byte{0x00, 0x04, 't', 'e', 's', 't'},
		IsParity: true,
		Checksum: true,
	}
	extBuf := make([]byte, extShard.TotalWireSize())
	_, _ = extShard.EncodeTo(extBuf)
	f.Add(extBuf)

	// Seed corpus with truncated/boundary inputs
	f.Add([]byte{})
	f.Add(make([]byte, CompactHeaderSize))
	f.Add(make([]byte, ExtendedHeaderSize))

	f.Fuzz(func(t *testing.T, data []byte) {
		shard, err := DecodeShard(data)
		if err != nil {
			return
		}

		// Valid shard invariant checks
		hdrSize := shard.HeaderSize()
		if hdrSize != CompactHeaderSize && hdrSize != ExtendedHeaderSize {
			t.Fatalf("unexpected header size: %d", hdrSize)
		}

		if shard.TotalWireSize() != hdrSize+len(shard.Data) {
			t.Fatalf("total wire size mismatch")
		}

		// Re-encoding must succeed without panic
		outBuf := make([]byte, shard.TotalWireSize())
		n, err := shard.EncodeTo(outBuf)
		if err != nil {
			t.Fatalf("re-encode failed for valid shard: %v", err)
		}

		redecoded, err := DecodeShard(outBuf[:n])
		if err != nil {
			t.Fatalf("re-decode failed: %v", err)
		}

		if redecoded.BaseSeq != shard.BaseSeq || redecoded.Mask != shard.Mask || !bytes.Equal(redecoded.Data, shard.Data) {
			t.Fatalf("round-trip data mismatch")
		}

		// Push to decoder must never panic
		dec := NewIncrementalDecoder(DecoderConfig{Capacity: 64, SymbolSize: 2000})
		_, _ = dec.PushShard(shard)
	})
}

// FuzzDecoderPushRawShard verifies that IncrementalDecoder.PushRawShard never panics
// on arbitrary stream input and maintains invariant solver metrics.
func FuzzDecoderPushRawShard(f *testing.F) {
	enc := NewSlidingEncoder(EncoderConfig{WindowSize: 16, SymbolSize: 1400, Checksum: true})
	s1, _ := enc.Push([]byte("fuzz-stream-seed-packet-1"))
	b1, _ := s1.MarshalBinary()
	f.Add(b1)

	p1, _ := enc.GenerateParity()
	bp1, _ := p1.MarshalBinary()
	f.Add(bp1)

	f.Fuzz(func(t *testing.T, raw []byte) {
		dec := NewIncrementalDecoder(DecoderConfig{
			Capacity:   128,
			SymbolSize: 2000,
			OnDecoded:  func(seq uint64, pkt []byte) {},
		})

		_, _ = dec.PushRawShard(raw)
	})
}
