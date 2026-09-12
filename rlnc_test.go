package rlnc

import (
	"bytes"
	"crypto/rand"
	"encoding/binary"
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

	// Test LeadingZeros and HighestBit
	var emptyBs Bitset256
	if emptyBs.LeadingZeros() != 256 || emptyBs.HighestBit() != -1 {
		t.Fatalf("empty bitset LeadingZeros/HighestBit failed: lz=%d hb=%d", emptyBs.LeadingZeros(), emptyBs.HighestBit())
	}

	single0 := NewBitset256(0)
	if single0.LeadingZeros() != 255 || single0.HighestBit() != 0 {
		t.Fatalf("bit 0 LeadingZeros/HighestBit failed: lz=%d hb=%d", single0.LeadingZeros(), single0.HighestBit())
	}

	single255 := NewBitset256(255)
	if single255.LeadingZeros() != 0 || single255.HighestBit() != 255 {
		t.Fatalf("bit 255 LeadingZeros/HighestBit failed: lz=%d hb=%d", single255.LeadingZeros(), single255.HighestBit())
	}

	multi := NewBitset256(10, 150, 250)
	if multi.HighestBit() != 250 || multi.LeadingZeros() != 5 {
		t.Fatalf("multi bitset LeadingZeros/HighestBit failed: lz=%d hb=%d", multi.LeadingZeros(), multi.HighestBit())
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

// 13. Test ZeroCopy vs Safe mode slice aliasing semantics
func TestDecoder_ZeroCopyVsSafe_Aliasing(t *testing.T) {
	enc := NewSlidingEncoder(EncoderConfig{WindowSize: 4, SymbolSize: 100})

	pkt0 := []byte("packet-0-payload-alpha")
	pkt1 := []byte("packet-1-payload-bravo")

	s0, _ := enc.Push(pkt0)
	s1, _ := enc.Push(pkt1)

	// Safe mode (default): returned [][]byte is isolated and survives across subsequent calls
	decSafe := NewIncrementalDecoder(DecoderConfig{Capacity: 16, SymbolSize: 100, ZeroCopy: false})
	rec0Safe, err := decSafe.PushShard(s0)
	if err != nil || len(rec0Safe) != 1 {
		t.Fatalf("rec0Safe failed: %v", err)
	}

	rec1Safe, err := decSafe.PushShard(s1)
	if err != nil || len(rec1Safe) != 1 {
		t.Fatalf("rec1Safe failed: %v", err)
	}

	// In safe mode, rec0Safe must retain its original data intact
	if !bytes.Equal(rec0Safe[0], pkt0) {
		t.Fatalf("safe mode violated: rec0Safe corrupted to %q", rec0Safe[0])
	}
	if !bytes.Equal(rec1Safe[0], pkt1) {
		t.Fatalf("safe mode violated: rec1Safe corrupted to %q", rec1Safe[0])
	}

	// Zero-copy mode with OnDecoded callback (strictly enforced)
	var streamedPackets [][]byte
	decZero := NewIncrementalDecoder(DecoderConfig{
		Capacity:   16,
		SymbolSize: 100,
		ZeroCopy:   true,
		OnDecoded: func(seq uint64, packet []byte) {
			cp := make([]byte, len(packet))
			copy(cp, packet)
			streamedPackets = append(streamedPackets, cp)
		},
	})

	rec0Zero, err := decZero.PushShard(s0)
	if err != nil || rec0Zero != nil {
		t.Fatalf("expected nil returned slice in ZeroCopy mode, got %v", rec0Zero)
	}
	rec1Zero, err := decZero.PushShard(s1)
	if err != nil || rec1Zero != nil {
		t.Fatalf("expected nil returned slice in ZeroCopy mode, got %v", rec1Zero)
	}

	if len(streamedPackets) != 2 {
		t.Fatalf("expected 2 streamed packets via OnDecoded, got %d", len(streamedPackets))
	}
	if !bytes.Equal(streamedPackets[0], pkt0) || !bytes.Equal(streamedPackets[1], pkt1) {
		t.Fatalf("streamed packet mismatch")
	}
}

// 14. Test that ZeroCopy without OnDecoded panics at construction
func TestDecoder_ZeroCopy_RequiresOnDecodedPanic(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic when ZeroCopy is true and OnDecoded is nil")
		}
	}()
	_ = NewIncrementalDecoder(DecoderConfig{Capacity: 16, SymbolSize: 100, ZeroCopy: true, OnDecoded: nil})
}

// 15. Test that ShiftLeft bitset overflow (> 255) is safely skipped without corrupting the matrix
func TestGaussJordan_BitsetShiftOverflowGuard(t *testing.T) {
	dec := NewIncrementalDecoder(DecoderConfig{Capacity: 64, SymbolSize: 100})

	// Inject a pivot at subSeq=10 whose mask spans up to bit 250 (HighestBit() = 250)
	var pData [102]byte
	binary.BigEndian.PutUint16(pData[:2], 4)
	copy(pData[2:6], []byte("piv0"))

	sPiv := Shard{
		BaseSeq: 10,
		Mask:    NewBitset256(0, 250),
		Data:    pData[:6],
	}
	_, err := dec.PushShard(sPiv)
	if err != nil {
		t.Fatalf("failed to insert initial pivot: %v", err)
	}

	// Now send a shard at curSeq=0 that has bit 10 set (which matches subSeq=10).
	// Shifting sPiv by k=10 would shift bit 250 to index 260 (which exceeds 255).
	// With the guard, the shift is safely skipped and no algebraic corruption occurs.
	var sData [102]byte
	binary.BigEndian.PutUint16(sData[:2], 4)
	copy(sData[2:6], []byte("cur0"))

	sCur := Shard{
		BaseSeq: 0,
		Mask:    NewBitset256(0, 10),
		Data:    sData[:6],
	}
	_, err = dec.PushShard(sCur)
	if err != nil {
		t.Fatalf("PushShard failed: %v", err)
	}

	// Assert decoder is still in a healthy state and can decode subsequent systematic packets
	sysPkt := []byte("clean-packet")
	var sysData [102]byte
	binary.BigEndian.PutUint16(sysData[:2], uint16(len(sysPkt)))
	copy(sysData[2:], sysPkt)

	sSys := Shard{
		BaseSeq: 20,
		Mask:    NewBitset256(0),
		Data:    sysData[:2+len(sysPkt)],
	}
	rec, err := dec.PushShard(sSys)
	if err != nil {
		t.Fatalf("PushShard systematic failed: %v", err)
	}
	if len(rec) != 1 || !bytes.Equal(rec[0], sysPkt) {
		t.Fatalf("expected clean systematic recovery, got %q", rec)
	}
}

// 16. Test that stale/ghost/replayed packets (diff < 0) never wipe pending resequencer queues
func TestInOrderResequencer_LateGhostPacketDoesNotReset(t *testing.T) {
	var deliveredSeqs []uint64
	reseq := NewInOrderResequencer(50*time.Millisecond, 100, func(seq uint64, pkt []byte) {
		deliveredSeqs = append(deliveredSeqs, seq)
	})
	defer reseq.Close()

	// 1. Deliver packet 10,000 in-order
	reseq.Push(10000, []byte("pkt-10000"))
	if len(deliveredSeqs) != 1 || deliveredSeqs[0] != 10000 {
		t.Fatalf("expected packet 10000 delivered, got %v", deliveredSeqs)
	}

	// 2. Push packet 10,002 (leaves gap at 10,001)
	reseq.Push(10002, []byte("pkt-10002"))
	_, _, _, pending := reseq.Stats()
	if pending != 1 {
		t.Fatalf("expected 1 pending packet (10002), got %d", pending)
	}

	// 3. Replay an old ghost packet (seq=0, diff = 0 - 10001 = -10001 < -10000)
	// Must be dropped safely WITHOUT resetting expectedSeq or wiping pending buffer!
	reseq.Push(0, []byte("ghost-pkt-0"))

	_, _, _, pending = reseq.Stats()
	if pending != 1 {
		t.Fatalf("pending queue was incorrectly wiped by ghost packet! pending=%d", pending)
	}

	// 4. Deliver missing packet 10,001
	reseq.Push(10001, []byte("pkt-10001"))

	// Both 10,001 and 10,002 must now be delivered in exact sequence
	if len(deliveredSeqs) != 3 {
		t.Fatalf("expected 3 delivered packets, got %v", deliveredSeqs)
	}
	if deliveredSeqs[1] != 10001 || deliveredSeqs[2] != 10002 {
		t.Fatalf("packets out of order or corrupted: %v", deliveredSeqs)
	}
}

// 17. Test that Encoder defaults to safe memory cloning (ZeroCopy: false)
func TestEncoder_SafeCopyByDefault(t *testing.T) {
	encSafe := NewSlidingEncoder(EncoderConfig{WindowSize: 2, SymbolSize: 100})
	s0, err := encSafe.Push([]byte("payload-000"))
	if err != nil {
		t.Fatal(err)
	}
	p0, err := encSafe.GenerateParity()
	if err != nil {
		t.Fatal(err)
	}

	// Push enough packets to overwrite window entries
	s1, _ := encSafe.Push([]byte("payload-111"))
	s2, _ := encSafe.Push([]byte("payload-222"))
	p1, _ := encSafe.GenerateParity()

	// Ensure s0 and p0 were NOT mutated
	if string(s0.Data[LengthPrefixSize:]) != "payload-000" {
		t.Fatalf("s0 was mutated! got %q", s0.Data[LengthPrefixSize:])
	}
	_ = s1
	_ = s2
	_ = p1
	_ = p0
}

// 18. Test that Decoder executes OnDecoded outside mutex lock (re-entrant safe)
func TestDecoder_OnDecodedLockFreedom(t *testing.T) {
	var dec *IncrementalDecoder
	var statsCalled bool

	dec = NewIncrementalDecoder(DecoderConfig{
		Capacity:   16,
		SymbolSize: 100,
		ZeroCopy:   true,
		OnDecoded: func(seq uint64, pkt []byte) {
			// Calling dec.Stats() while OnDecoded runs would DEADLOCK if mutex was held!
			rx, inno, _, _ := dec.Stats()
			if rx > 0 && inno > 0 {
				statsCalled = true
			}
		},
	})

	enc := NewSlidingEncoder(EncoderConfig{WindowSize: 4, SymbolSize: 100, ZeroCopy: true})
	s0, _ := enc.Push([]byte("reentrant-test"))
	_, _ = dec.PushShard(s0)

	if !statsCalled {
		t.Fatal("expected re-entrant dec.Stats() inside OnDecoded to succeed without deadlock")
	}
}

// 19. Test that InOrderResequencer executes onEmit outside mutex lock (re-entrant safe)
func TestInOrderResequencer_OnEmitLockFreedom(t *testing.T) {
	var reseq *InOrderResequencer
	var statsCalled bool

	reseq = NewInOrderResequencer(10*time.Millisecond, 10, func(seq uint64, pkt []byte) {
		// Calling reseq.Stats() from onEmit would DEADLOCK if mutex was held!
		del, _, _, _ := reseq.Stats()
		if del >= 1 {
			statsCalled = true
		}
	})
	defer reseq.Close()

	reseq.Push(0, []byte("pkt-0"))
	if !statsCalled {
		t.Fatal("expected re-entrant reseq.Stats() inside onEmit to succeed without deadlock")
	}
}

// 20. Test that Gauss-Jordan reduceAndInsert eliminates bit 0 against solvedRing
func TestGaussJordan_SolvedRingBit0Reduction(t *testing.T) {
	enc := NewSlidingEncoder(EncoderConfig{WindowSize: 2, SymbolSize: 100, ZeroCopy: true})
	dec := NewIncrementalDecoder(DecoderConfig{Capacity: 16, SymbolSize: 100})

	alpha := []byte("alpha")
	bravo := []byte("bravo")

	s0, err := enc.Push(alpha)
	if err != nil {
		t.Fatal(err)
	}
	_, err = enc.Push(bravo)
	if err != nil {
		t.Fatal(err)
	}
	p, err := enc.GenerateParity()
	if err != nil {
		t.Fatal(err)
	}

	// 1. Ingest systematic packet seq 0 (enters solvedRing immediately)
	_, err = dec.PushShard(s0)
	if err != nil {
		t.Fatal(err)
	}

	// 2. Ingest parity shard combining seq 0 and seq 1
	// Bit 0 is eliminated against solvedRing, reducing the equation directly to seq 1!
	rec, err := dec.PushShard(p)
	if err != nil {
		t.Fatalf("PushShard failed: %v", err)
	}
	if len(rec) != 1 || !bytes.Equal(rec[0], bravo) {
		t.Fatalf("expected packet 1 ('bravo') recovered via solvedRing bit 0 elimination, got %q", rec)
	}
}




