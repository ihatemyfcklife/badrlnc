package rlnc

import (
	"bytes"
	"crypto/rand"
	"sync"
	"testing"
	"time"
)

func makeRandomBytes(n int) []byte {
	buf := make([]byte, n)
	_, _ = rand.Read(buf)
	return buf
}

// 1. Test Bitset256 operations
func TestBitset256(t *testing.T) {
	var b Bitset256
	if !b.IsZero() {
		t.Fatal("expected zero bitset")
	}

	b.SetBit(0)
	b.SetBit(63)
	b.SetBit(64)
	b.SetBit(127)
	b.SetBit(255)

	if b.Weight() != 5 {
		t.Fatalf("expected weight 5, got %d", b.Weight())
	}

	if !b.TestBit(0) || !b.TestBit(63) || !b.TestBit(64) || !b.TestBit(127) || !b.TestBit(255) {
		t.Fatal("expected set bits to be true")
	}
	if b.TestBit(1) || b.TestBit(65) || b.TestBit(254) {
		t.Fatal("expected unset bits to be false")
	}

	if b.TrailingZeros() != 0 {
		t.Fatalf("expected trailing zeros 0, got %d", b.TrailingZeros())
	}

	b.ClearBit(0)
	if b.TrailingZeros() != 63 {
		t.Fatalf("expected trailing zeros 63, got %d", b.TrailingZeros())
	}

	// Test Shifts
	var s Bitset256
	s.SetBit(1)
	s.SetBit(65)
	s.SetBit(130)

	shiftedL := s.ShiftLeft(3)
	if !shiftedL.TestBit(4) || !shiftedL.TestBit(68) || !shiftedL.TestBit(133) {
		t.Fatal("ShiftLeft(3) failed")
	}

	shiftedR := shiftedL.ShiftRight(3)
	if !shiftedR.TestBit(1) || !shiftedR.TestBit(65) || !shiftedR.TestBit(130) {
		t.Fatal("ShiftRight(3) failed")
	}

	shiftedW := s.ShiftLeft(64)
	if !shiftedW.TestBit(65) || !shiftedW.TestBit(129) || !shiftedW.TestBit(194) {
		t.Fatal("ShiftLeft(64) failed")
	}

	// Constructors and conversions
	bFromU64 := NewBitset256FromUint64(0x12345678)
	if !bFromU64.IsUint64() || bFromU64.Uint64() != 0x12345678 {
		t.Fatalf("NewBitset256FromUint64 failed: got %x", bFromU64.Uint64())
	}

	bFromIndices := NewBitset256(5, 70, 200)
	if !bFromIndices.TestBit(5) || !bFromIndices.TestBit(70) || !bFromIndices.TestBit(200) {
		t.Fatal("NewBitset256 indices test failed")
	}
	if bFromIndices.IsUint64() {
		t.Fatal("bFromIndices should not fit in uint64")
	}

	str := bFromU64.String()
	if len(str) == 0 {
		t.Fatal("String() returned empty string")
	}
}

// 2. Test XOR SIMD acceleration
func TestXOR(t *testing.T) {
	sizes := []int{1, 7, 8, 15, 16, 31, 32, 63, 64, 100, 1400, 1500}
	for _, sz := range sizes {
		a := makeRandomBytes(sz)
		b := makeRandomBytes(sz)
		expected := make([]byte, sz)
		for i := 0; i < sz; i++ {
			expected[i] = a[i] ^ b[i]
		}

		// Test XOR(dst, a, b)
		dst := make([]byte, sz)
		n := XOR(dst, a, b)
		if n != sz || !bytes.Equal(dst, expected) {
			t.Fatalf("XOR mismatch for size %d", sz)
		}

		// Test XORBytes(dst, src, sz)
		dstBytes := make([]byte, sz)
		copy(dstBytes, a)
		XORBytes(dstBytes, b, sz)
		if !bytes.Equal(dstBytes, expected) {
			t.Fatalf("XORBytes mismatch for size %d", sz)
		}
	}
}

// 3. Test Shard serialization and deserialization
func TestShardSerialization(t *testing.T) {
	// Compact 24-byte header test
	shardCompact := Shard{
		BaseSeq:  0x1122334455667788,
		Mask:     NewBitset256FromUint64(0xaabbccddeeff0011),
		Data:     makeRandomBytes(1200),
		IsParity: true,
	}

	if shardCompact.HeaderSize() != CompactHeaderSize {
		t.Fatalf("expected compact header size %d, got %d", CompactHeaderSize, shardCompact.HeaderSize())
	}

	rawCompact := make([]byte, shardCompact.TotalWireSize())
	n, err := shardCompact.EncodeTo(rawCompact)
	if err != nil || n != shardCompact.TotalWireSize() {
		t.Fatalf("EncodeTo compact failed: %v", err)
	}

	decodedCompact, err := DecodeShard(rawCompact)
	if err != nil {
		t.Fatalf("DecodeShard compact failed: %v", err)
	}
	if decodedCompact.BaseSeq != shardCompact.BaseSeq ||
		decodedCompact.Mask != shardCompact.Mask ||
		decodedCompact.IsParity != shardCompact.IsParity ||
		!bytes.Equal(decodedCompact.Data, shardCompact.Data) {
		t.Fatal("decoded compact shard does not match original")
	}

	// Extended 48-byte header test
	shardExtended := Shard{
		BaseSeq:  0x9988776655443322,
		Mask:     NewBitset256(10, 100, 200),
		Data:     makeRandomBytes(500),
		IsParity: true,
	}

	if shardExtended.HeaderSize() != ExtendedHeaderSize {
		t.Fatalf("expected extended header size %d, got %d", ExtendedHeaderSize, shardExtended.HeaderSize())
	}

	rawExtended, err := shardExtended.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary extended failed: %v", err)
	}

	var unmarshaledExtended Shard
	err = unmarshaledExtended.UnmarshalBinary(rawExtended)
	if err != nil {
		t.Fatalf("UnmarshalBinary extended failed: %v", err)
	}
	if unmarshaledExtended.BaseSeq != shardExtended.BaseSeq ||
		unmarshaledExtended.Mask != shardExtended.Mask ||
		unmarshaledExtended.IsParity != shardExtended.IsParity ||
		!bytes.Equal(unmarshaledExtended.Data, shardExtended.Data) {
		t.Fatal("unmarshaled extended shard does not match original")
	}
}

// 4. Test SlidingWindowEncoder & IncrementalDecoder Single Packet
func TestEncoderDecoder_SinglePacket(t *testing.T) {
	enc := NewSlidingEncoder(EncoderConfig{WindowSize: 16, SymbolSize: 1400})
	dec := NewIncrementalDecoder(DecoderConfig{Capacity: 64, SymbolSize: 1400})

	pkt := []byte("Hello, High-Performance RLNC in GF(2)!")
	sysShard, err := enc.Push(pkt)
	if err != nil {
		t.Fatalf("Push failed: %v", err)
	}

	if sysShard.Seq() != 0 {
		t.Fatalf("expected seq 0, got %d", sysShard.Seq())
	}

	recovered, err := dec.PushShard(sysShard)
	if err != nil {
		t.Fatalf("PushShard failed: %v", err)
	}
	if len(recovered) != 1 {
		t.Fatalf("expected 1 recovered packet, got %d", len(recovered))
	}
	if !bytes.Equal(recovered[0], pkt) {
		t.Fatalf("recovered packet mismatch: got %q, want %q", recovered[0], pkt)
	}
}

// 5. Test Continuous Stream over sliding window W=32
func TestEncoderDecoder_ContinuousStream(t *testing.T) {
	const numPackets = 100
	enc := NewSlidingEncoder(EncoderConfig{WindowSize: 32, SymbolSize: 1400})
	dec := NewIncrementalDecoder(DecoderConfig{Capacity: 256, SymbolSize: 1400})

	packets := make([][]byte, numPackets)
	for i := 0; i < numPackets; i++ {
		sz := 64 + (i*37)%1200
		packets[i] = makeRandomBytes(sz)
		packets[i][0] = byte(i >> 24)
		packets[i][1] = byte(i >> 16)
		packets[i][2] = byte(i >> 8)
		packets[i][3] = byte(i)
	}

	for i := 0; i < numPackets; i++ {
		shard, err := enc.Push(packets[i])
		if err != nil {
			t.Fatalf("Push %d failed: %v", i, err)
		}

		recovered, err := dec.PushShard(shard)
		if err != nil {
			t.Fatalf("PushShard %d failed: %v", i, err)
		}
		if len(recovered) != 1 {
			t.Fatalf("expected 1 recovered packet at index %d, got %d", i, len(recovered))
		}
		if !bytes.Equal(recovered[0], packets[i]) {
			t.Fatalf("packet %d payload mismatch", i)
		}
	}

	rx, inno, red, decoded := dec.Stats()
	if rx != numPackets || inno != numPackets || red != 0 || decoded != numPackets {
		t.Fatalf("unexpected decoder stats: rx=%d inno=%d red=%d decoded=%d", rx, inno, red, decoded)
	}
}

// 6. Test Multi-Variable linear combination reduction and redundant symbol rejection
func TestEncoderDecoder_MultiVariableReduction(t *testing.T) {
	enc := NewSlidingEncoder(EncoderConfig{WindowSize: 16, SymbolSize: 1400})
	dec := NewIncrementalDecoder(DecoderConfig{Capacity: 64, SymbolSize: 1400})

	p1 := []byte("packet-alpha-12345")
	p2 := []byte("packet-bravo-67890")

	shard1, err := enc.Push(p1)
	if err != nil {
		t.Fatal(err)
	}
	shard2, err := enc.Push(p2)
	if err != nil {
		t.Fatal(err)
	}

	linShard, err := enc.GenerateParity()
	if err != nil {
		t.Fatal(err)
	}

	// Ingest linear parity shard first (cannot solve yet, under-determined)
	recLin, err := dec.PushShard(linShard)
	if err != nil {
		t.Fatal(err)
	}
	if len(recLin) != 0 {
		t.Fatalf("linear shard alone should not solve packets, got %d", len(recLin))
	}

	// Ingest shard 1: solves p1 AND triggers back-substitution to solve p2!
	rec1, err := dec.PushShard(shard1)
	if err != nil {
		t.Fatal(err)
	}
	if len(rec1) != 2 {
		t.Fatalf("expected 2 recovered packets (cascade), got %d", len(rec1))
	}
	if !bytes.Equal(rec1[0], p1) || !bytes.Equal(rec1[1], p2) {
		t.Fatal("recovered packets do not match original payloads")
	}

	// Ingesting shard 2 now must be detected as redundant
	rec2, err := dec.PushShard(shard2)
	if err != nil {
		t.Fatal(err)
	}
	if len(rec2) != 0 {
		t.Fatalf("duplicate shard 2 should yield 0 recovered packets, got %d", len(rec2))
	}

	// Ingesting duplicate linear shard must be redundant
	recDup, err := dec.PushShard(linShard)
	if err != nil {
		t.Fatal(err)
	}
	if len(recDup) != 0 {
		t.Fatalf("duplicate linear shard should yield 0 recovered packets, got %d", len(recDup))
	}

	_, _, red, _ := dec.Stats()
	if red != 2 {
		t.Fatalf("expected 2 redundant shards, got %d", red)
	}
}

// 7. Test Inactivity Timeout in SlidingWindowEncoder
func TestSlidingWindow_InactivityTimeout(t *testing.T) {
	enc := NewSlidingEncoder(EncoderConfig{
		WindowSize:        16,
		SymbolSize:        1400,
		InactivityTimeout: 50 * time.Millisecond,
	})

	_, err := enc.Push([]byte("test-packet"))
	if err != nil {
		t.Fatal(err)
	}

	// Immediately generate parity -> should succeed
	_, err = enc.GenerateParity()
	if err != nil {
		t.Fatalf("immediate GenerateParity should succeed: %v", err)
	}

	// Wait for inactivity timeout
	time.Sleep(70 * time.Millisecond)

	// Now window should be considered idle
	_, err = enc.GenerateParity()
	if err != ErrZeroPayload {
		t.Fatalf("expected ErrZeroPayload after inactivity timeout, got %v", err)
	}
}

// 8. Test InOrderResequencer
func TestInOrderResequencer(t *testing.T) {
	var mu sync.Mutex
	var delivered []uint64

	reseq := NewInOrderResequencer(20*time.Millisecond, 100, func(seq uint64, pkt []byte) {
		mu.Lock()
		defer mu.Unlock()
		delivered = append(delivered, seq)
	})
	defer reseq.Close()

	// Push out of order: 2, 0, 1, 4, 3
	order := []uint64{2, 0, 1, 4, 3}
	for _, s := range order {
		pkt := []byte{byte(s)}
		reseq.Push(s, pkt)
	}

	time.Sleep(10 * time.Millisecond)

	mu.Lock()
	defer mu.Unlock()
	if len(delivered) != 5 {
		t.Fatalf("expected 5 delivered packets, got %d: %v", len(delivered), delivered)
	}
	for i, v := range delivered {
		if v != uint64(i) {
			t.Fatalf("expected index %d to be seq %d, got %d", i, i, v)
		}
	}
}

// 9. Test InOrderResequencer Timeout and Gap Skipping
func TestInOrderResequencer_TimeoutAndGapSkip(t *testing.T) {
	var mu sync.Mutex
	var delivered []uint64

	reseq := NewInOrderResequencer(10*time.Millisecond, 10, func(seq uint64, pkt []byte) {
		mu.Lock()
		defer mu.Unlock()
		delivered = append(delivered, seq)
	})
	defer reseq.Close()

	// Deliver 0, skip 1, deliver 2
	reseq.Push(0, []byte{0})
	reseq.Push(2, []byte{2})

	// Wait for timeout to fire and skip missing sequence 1
	time.Sleep(30 * time.Millisecond)

	mu.Lock()
	if len(delivered) != 2 || delivered[0] != 0 || delivered[1] != 2 {
		mu.Unlock()
		t.Fatalf("expected [0, 2] delivered after gap skip, got %v", delivered)
	}
	mu.Unlock()

	del, reord, skipped, pending := reseq.Stats()
	if del != 2 || reord != 1 || skipped != 1 || pending != 0 {
		t.Fatalf("unexpected resequencer stats: del=%d, reord=%d, skipped=%d, pending=%d", del, reord, skipped, pending)
	}
}

// 10. Test Buffer Pools
func TestBufferPools(t *testing.T) {
	sPool := NewShardBufferPool()
	sBuf := sPool.Get()
	if len(sBuf) < MaxShardSize {
		t.Fatalf("expected shard buffer cap >= %d, got %d", MaxShardSize, len(sBuf))
	}
	sPool.Put(sBuf)
	// Test recycling emptied slice (len == 0, cap >= MaxShardSize)
	sBufEmpty := sPool.Get()[:0]
	sPool.Put(sBufEmpty)

	pPool := NewPacketBufferPool()
	pBuf := pPool.Get()
	if len(pBuf) < MaxPacketBufferSize {
		t.Fatalf("expected packet buffer cap >= %d, got %d", MaxPacketBufferSize, len(pBuf))
	}
	pBufEmpty := pBuf[:0]
	pPool.Put(pBufEmpty)
	pPool.Put(make([]byte, 10)) // Too small, should not panic

	gShard := GetShardBuffer()
	PutShardBuffer(gShard)

	gPkt := GetPacketBuffer()
	PutPacketBuffer(gPkt)
}

// 11. Test Encoder and Decoder Lifecycle, Reset, and Error branches
func TestLifecycleAndErrors(t *testing.T) {
	// Encoder error cases
	enc := NewSlidingEncoder(EncoderConfig{WindowSize: 8, SymbolSize: 100})
	if enc.WindowSize() != 8 {
		t.Fatalf("expected window size 8, got %d", enc.WindowSize())
	}

	_, err := enc.Push(nil)
	if err != ErrZeroPayload {
		t.Fatalf("expected ErrZeroPayload, got %v", err)
	}

	tooLarge := make([]byte, 101)
	_, err = enc.Push(tooLarge)
	if err != ErrPayloadTooLarge {
		t.Fatalf("expected ErrPayloadTooLarge, got %v", err)
	}

	// Generate parity on empty encoder
	_, err = enc.GenerateParity()
	if err != ErrZeroPayload {
		t.Fatalf("expected ErrZeroPayload on empty encoder, got %v", err)
	}

	// Push valid packet
	_, err = enc.Push([]byte("lifecycle-packet"))
	if err != nil {
		t.Fatal(err)
	}
	if enc.ActivePacketCount() != 1 {
		t.Fatalf("expected active count 1, got %d", enc.ActivePacketCount())
	}
	base, latest, count := enc.WindowState()
	if base != 0 || latest != 0 || count != 1 {
		t.Fatalf("unexpected window state: base=%d, latest=%d, count=%d", base, latest, count)
	}

	in, out := enc.Stats()
	if in != 1 || out != 1 {
		t.Fatalf("expected stats in=1, out=1, got in=%d, out=%d", in, out)
	}

	enc.Reset()
	if enc.ActivePacketCount() != 0 {
		t.Fatalf("expected 0 active packets after reset, got %d", enc.ActivePacketCount())
	}

	// Decoder error and reset cases
	dec := NewIncrementalDecoder(DecoderConfig{Capacity: 32, SymbolSize: 100})
	_, err = dec.PushShard(Shard{})
	if err != ErrZeroPayload {
		t.Fatalf("expected ErrZeroPayload on empty shard data, got %v", err)
	}

	_, err = dec.PushShard(Shard{Data: []byte{1}}) // less than LengthPrefixSize
	if err != ErrCorruptHeader {
		t.Fatalf("expected ErrCorruptHeader on short data, got %v", err)
	}

	dec.Reset()
	rx, inno, red, decCount := dec.Stats()
	if rx != 0 || inno != 0 || red != 0 || decCount != 0 {
		t.Fatalf("expected 0 stats after decoder reset")
	}

	// Resequencer reset and jump cases
	var emittedSeqs []uint64
	reseq := NewInOrderResequencer(5*time.Millisecond, 10, func(seq uint64, pkt []byte) {
		emittedSeqs = append(emittedSeqs, seq)
	})
	reseq.Push(50, []byte("mid-stream-reconnect")) // seq <= 128 starts at seq 0, wait for jump
	reseq.Push(1000000, []byte("huge-jump"))
	reseq.Reset()
	del, reord, skipped, pending := reseq.Stats()
	if del != 0 || reord != 0 || skipped != 0 || pending != 0 {
		t.Fatalf("expected 0 stats after resequencer reset")
	}

	// Shard EncodeTo buffer too small
	s := Shard{BaseSeq: 1, Mask: NewBitset256FromUint64(1), Data: []byte("data")}
	dstSmall := make([]byte, 10)
	_, err = s.EncodeTo(dstSmall)
	if err != ErrBufferTooSmall {
		t.Fatalf("expected ErrBufferTooSmall, got %v", err)
	}

	// XORRow and XORMulti
	dst := make([]byte, 8)
	src1 := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	src2 := []byte{8, 7, 6, 5, 4, 3, 2, 1}
	XORRow(dst, src1, 8)
	if !bytes.Equal(dst, src1) {
		t.Fatal("XORRow failed")
	}
	XORMulti(dst, [][]byte{src2}, 8)
	expectedXOR := make([]byte, 8)
	for i := 0; i < 8; i++ {
		expectedXOR[i] = src1[i] ^ src2[i]
	}
	if !bytes.Equal(dst, expectedXOR) {
		t.Fatal("XORMulti failed")
	}
	ClearBytes(dst)
	if !bytes.Equal(dst, make([]byte, 8)) {
		t.Fatal("ClearBytes failed")
	}
}

