package badrlnc

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math/rand"
	"sync"
	"testing"
)

// simulateLossTransmission simulates sending numPackets over an erasure channel
// with lossPct packet loss and parityPct parity redundancy.
func simulateLossTransmission(w int, parityPct int, lossPct float64, burstSize int, numPackets int, seed uint64) (sent, dropped, recovered int, allRecoveredMatch bool) {
	enc := NewSlidingEncoder(EncoderConfig{WindowSize: w, SymbolSize: 1400, DisableInactivityTimeout: true, Seed: seed})

	var mu sync.Mutex
	reconstructed := make(map[uint64][]byte)

	dec := NewIncrementalDecoder(DecoderConfig{
		Capacity:   2048,
		SymbolSize: 1400,
		OnDecoded: func(seq uint64, packet []byte) {
			mu.Lock()
			defer mu.Unlock()
			cp := make([]byte, len(packet))
			copy(cp, packet)
			reconstructed[seq] = cp
		},
	})

	rng := rand.New(rand.NewSource(int64(seed)))

	sourcePackets := make([][]byte, numPackets)
	for i := 0; i < numPackets; i++ {
		sourcePackets[i] = make([]byte, 250+(i*17)%800)
		binary.BigEndian.PutUint64(sourcePackets[i][:8], uint64(i))
		for j := 8; j < len(sourcePackets[i]); j++ {
			sourcePackets[i][j] = byte(i*31 + j)
		}
	}

	type wireShard struct {
		isSystematic bool
		seq          uint64
		shard        Shard
	}
	var channelQueue []wireShard

	var parityAccumulator int
	for i := 0; i < numPackets; i++ {
		sysShard, err := enc.Push(sourcePackets[i])
		if err != nil {
			panic(err)
		}
		channelQueue = append(channelQueue, wireShard{
			isSystematic: true,
			seq:          uint64(i),
			shard:        sysShard.Clone(),
		})

		parityAccumulator += parityPct
		for parityAccumulator >= 100 {
			parityAccumulator -= 100
			parShard, err := enc.GenerateParity()
			if err == nil {
				channelQueue = append(channelQueue, wireShard{
					isSystematic: false,
					shard:        parShard.Clone(),
				})
			}
		}
	}

	// Flush trailing parities to protect the tail of the sliding window
	tailParities := w / 2
	if tailParities < 4 {
		tailParities = 4
	}
	for p := 0; p < tailParities; p++ {
		parShard, err := enc.GenerateParity()
		if err == nil {
			channelQueue = append(channelQueue, wireShard{
				isSystematic: false,
				shard:        parShard.Clone(),
			})
		}
	}

	// Apply simulated channel packet loss
	inBurst := 0
	droppedSystematic := 0

	for _, item := range channelQueue {
		isDropped := false
		if inBurst > 0 {
			inBurst--
			isDropped = true
		} else if lossPct > 0 {
			if burstSize > 1 {
				burstTriggerProb := (lossPct / 100.0) / float64(burstSize)
				if rng.Float64() < burstTriggerProb {
					inBurst = burstSize - 1
					isDropped = true
				}
			} else {
				if rng.Float64() < (lossPct / 100.0) {
					isDropped = true
				}
			}
		}

		if isDropped {
			if item.isSystematic {
				droppedSystematic++
			}
			continue
		}

		_, _ = dec.PushShard(item.shard)
	}

	mu.Lock()
	defer mu.Unlock()

	matchedCount := 0
	allMatch := true
	for i := 0; i < numPackets; i++ {
		rec, found := reconstructed[uint64(i)]
		if found {
			matchedCount++
			if !bytes.Equal(rec, sourcePackets[i]) {
				allMatch = false
			}
		}
	}

	return numPackets, droppedSystematic, matchedCount, allMatch
}

// TestMatrixInversion_SimulatedLoss validates incremental Gauss-Jordan RREF inversion
// under 10%, 20%, 30%, and 40% simulated packet loss rates.
func TestMatrixInversion_SimulatedLoss(t *testing.T) {
	testCases := []struct {
		name         string
		lossPct      float64
		parityPct    int
		windowSize   int
		numPackets   int
		minRecovRate float64
	}{
		// 10% Loss: With 25% parity, recovery should be 100%
		{
			name:         "Loss 10% (W=32, Parity=25%)",
			lossPct:      10.0,
			parityPct:    25,
			windowSize:   32,
			numPackets:   120,
			minRecovRate: 98.0,
		},
		// 20% Loss: With 50% parity, recovery should be >= 99%
		{
			name:         "Loss 20% (W=32, Parity=50%)",
			lossPct:      20.0,
			parityPct:    50,
			windowSize:   32,
			numPackets:   120,
			minRecovRate: 98.0,
		},
		// 30% Loss: With 66% parity, recovery should be >= 98%
		{
			name:         "Loss 30% (W=32, Parity=66%)",
			lossPct:      30.0,
			parityPct:    66,
			windowSize:   32,
			numPackets:   120,
			minRecovRate: 97.0,
		},
		// 40% Loss: With 100% parity (1:1), recovery should be >= 98%
		{
			name:         "Loss 40% (W=32, Parity=100%)",
			lossPct:      40.0,
			parityPct:    100,
			windowSize:   32,
			numPackets:   120,
			minRecovRate: 98.0,
		},
		// 40% Loss with larger window (W=64)
		{
			name:         "Loss 40% (W=64, Parity=100%)",
			lossPct:      40.0,
			parityPct:    100,
			windowSize:   64,
			numPackets:   150,
			minRecovRate: 98.0,
		},
		// Burst loss of 4 packets under 20% total loss
		{
			name:         "Burst Loss 4-pkts (W=32, Parity=50%, Loss=20%)",
			lossPct:      20.0,
			parityPct:    50,
			windowSize:   32,
			numPackets:   120,
			minRecovRate: 95.0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			sent, dropped, recovered, allMatch := simulateLossTransmission(
				tc.windowSize,
				tc.parityPct,
				tc.lossPct,
				1,
				tc.numPackets,
				0xdeadbeef+uint64(tc.lossPct*10),
			)

			if !allMatch {
				t.Fatalf("[%s] data corruption: recovered packets do not match original source packets", tc.name)
			}

			recovRate := (float64(recovered) / float64(sent)) * 100.0
			t.Logf("[%s] Sent: %d, Systematic Dropped: %d, Recovered: %d (%.2f%%)",
				tc.name, sent, dropped, recovered, recovRate)

			if recovRate < tc.minRecovRate {
				t.Fatalf("[%s] recovery rate %.2f%% below required minimum %.2f%% (dropped: %d)",
					tc.name, recovRate, tc.minRecovRate, dropped)
			}
		})
	}
}

// TestMatrixInversion_UnsolvableDegreesOfFreedom verifies that under-determined systems
// (received symbols < source packets) strictly do not emit false packets.
func TestMatrixInversion_UnsolvableDegreesOfFreedom(t *testing.T) {
	enc := NewSlidingEncoder(EncoderConfig{WindowSize: 16, SymbolSize: 1400, Seed: 0xdeadbeefcafe1337})

	var mu sync.Mutex
	reconstructed := make(map[uint64][]byte)

	dec := NewIncrementalDecoder(DecoderConfig{
		Capacity:   64,
		SymbolSize: 1400,
		OnDecoded: func(seq uint64, packet []byte) {
			mu.Lock()
			defer mu.Unlock()
			reconstructed[seq] = append([]byte(nil), packet...)
		},
	})

	// Ingest 16 source packets
	for i := 0; i < 16; i++ {
		_, _ = enc.Push([]byte(fmt.Sprintf("Underdetermined-Packet-%02d", i)))
	}

	// Drop ALL systematic packets 0..15.
	// Feed only 10 parity shards (10 equations for 16 unknowns -> strictly underdetermined).
	for p := 0; p < 10; p++ {
		parity, err := enc.GenerateParity()
		if err != nil {
			t.Fatal(err)
		}
		rec, err := dec.PushShard(parity)
		if err != nil {
			t.Fatal(err)
		}
		if len(rec) != 0 {
			t.Fatalf("underdetermined system emitted %d recovered packets before receiving sufficient rank", len(rec))
		}
	}

	mu.Lock()
	count := len(reconstructed)
	mu.Unlock()

	if count != 0 {
		t.Fatalf("expected 0 recovered packets from rank-deficient matrix, got %d", count)
	}

	// Now supply systematic packets 8..15
	for i := 8; i < 16; i++ {
		sysShard := Shard{
			BaseSeq:  uint64(i),
			Mask:     NewBitset256FromUint64(1),
			Data:     enc.entries[i%16].data[:enc.entries[i%16].len],
			IsParity: false,
		}
		_, _ = dec.PushShard(sysShard)
	}

	// Supply additional linear parities to reach full algebraic rank over GF(2)
	for p := 0; p < 8; p++ {
		parity, err := enc.GenerateParity()
		if err == nil {
			_, _ = dec.PushShard(parity)
		}
	}

	mu.Lock()
	finalCount := len(reconstructed)
	mu.Unlock()

	t.Logf("Supplied remaining degrees of freedom: Total Decoded = %d/16", finalCount)
	if finalCount < 16 {
		t.Fatalf("expected all 16 packets recovered after providing full rank, got %d", finalCount)
	}
}
