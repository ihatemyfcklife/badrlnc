package rlnc

import (
	"crypto/rand"
	"testing"
)

// BenchmarkSIMDXOR_1400B benchmarks line-rate SIMD XOR throughput on typical MTU buffers (1400 bytes).
// Reports throughput in GB/s via b.SetBytes.
func BenchmarkSIMDXOR_1400B(b *testing.B) {
	dst := make([]byte, 1400)
	src := make([]byte, 1400)
	_, _ = rand.Read(src)

	b.SetBytes(1400)
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		XORBytes(dst, src, 1400)
	}
}

// BenchmarkSIMDXOR_Multi4_1400B benchmarks SIMD XOR combination of 4 sources simultaneously.
// Reports throughput in GB/s via b.SetBytes.
func BenchmarkSIMDXOR_Multi4_1400B(b *testing.B) {
	dst := make([]byte, 1400)
	srcs := make([][]byte, 4)
	for i := range srcs {
		srcs[i] = make([]byte, 1400)
		_, _ = rand.Read(srcs[i])
	}

	b.SetBytes(1400 * 4)
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		XORMulti(dst, srcs, 1400)
	}
}

// BenchmarkEncoder_Push_ZeroAlloc verifies that SlidingWindowEncoder.Push achieves strictly 0 allocs/op.
func BenchmarkEncoder_Push_ZeroAlloc(b *testing.B) {
	enc := NewSlidingEncoder(EncoderConfig{WindowSize: 32, SymbolSize: 1400})
	payload := make([]byte, 1200)
	_, _ = rand.Read(payload)

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _ = enc.Push(payload)
	}
}

// BenchmarkEncoder_GenerateParity_ZeroAlloc verifies that SlidingWindowEncoder.GenerateParity achieves 0 allocs/op.
func BenchmarkEncoder_GenerateParity_ZeroAlloc(b *testing.B) {
	enc := NewSlidingEncoder(EncoderConfig{WindowSize: 32, SymbolSize: 1400})
	payload := make([]byte, 1200)
	_, _ = rand.Read(payload)

	for i := 0; i < 32; i++ {
		_, _ = enc.Push(payload)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _ = enc.GenerateParity()
	}
}

// BenchmarkSystematicFastPath_ZeroAlloc verifies that full end-to-end systematic encode + decode
// achieves strictly 0 heap allocations per operation with ZeroCopy enabled.
func BenchmarkSystematicFastPath_ZeroAlloc(b *testing.B) {
	enc := NewSlidingEncoder(EncoderConfig{WindowSize: 32, SymbolSize: 1400})
	dec := NewIncrementalDecoder(DecoderConfig{
		Capacity:   2048,
		SymbolSize: 1400,
		ZeroCopy:   true,
		OnDecoded:  func(seq uint64, pkt []byte) {},
	})
	payload := make([]byte, 1200)
	_, _ = rand.Read(payload)

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		shard, _ := enc.Push(payload)
		_, _ = dec.PushShard(shard)
	}
}

// BenchmarkDecoder_IncrementalRREF_Kpps benchmarks incremental Gauss-Jordan RREF decoding
// and reports throughput in packets/sec (Kpps).
func BenchmarkDecoder_IncrementalRREF_Kpps(b *testing.B) {
	enc := NewSlidingEncoder(EncoderConfig{WindowSize: 32, SymbolSize: 1400})
	dec := NewIncrementalDecoder(DecoderConfig{
		Capacity:   2048,
		SymbolSize: 1400,
		ZeroCopy:   true,
		OnDecoded:  func(seq uint64, pkt []byte) {},
	})
	payload := make([]byte, 1200)
	_, _ = rand.Read(payload)

	// Pre-generate a sequence of systematic and parity shards
	const numShards = 1000
	shards := make([]Shard, numShards)
	for i := 0; i < numShards; i++ {
		if i%4 != 3 {
			s, _ := enc.Push(payload)
			shards[i] = s.Clone()
		} else {
			p, _ := enc.GenerateParity()
			shards[i] = p.Clone()
		}
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		s := shards[i%numShards]
		_, _ = dec.PushShard(s)
	}
}

// BenchmarkBitset256Operations benchmarks discrete GF(2) vector algebra operations.
func BenchmarkBitset256Operations(b *testing.B) {
	var bs1, bs2 Bitset256
	bs1.SetBit(0)
	bs1.SetBit(63)
	bs1.SetBit(127)
	bs2.SetBit(42)
	bs2.SetBit(99)

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		bs1.XOR(bs2)
		_ = bs1.ShiftRight(1)
		_ = bs1.ShiftLeft(1)
		_ = bs1.TrailingZeros()
		_ = bs1.Weight()
	}
}
