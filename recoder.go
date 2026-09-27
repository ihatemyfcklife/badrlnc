package badrlnc

import (
	"crypto/rand"
	"encoding/binary"
	"errors"
	"sync"
	"time"
)

var (
	// ErrInsufficientShards indicates that no innovative shards are available in the recoder pool.
	ErrInsufficientShards = errors.New("badrlnc: insufficient shards available to recode")

	// ErrIncompatibleShards indicates that the selected shards have disjoint windows exceeding 256 bits.
	ErrIncompatibleShards = errors.New("badrlnc: shards cannot be combined (window span exceeds 256 bits)")
)

// RecoderConfig defines configuration parameters for SwarmRecoder.
type RecoderConfig struct {
	// Capacity specifies the maximum number of shards kept in the recoding pool.
	// Default is 128.
	Capacity int

	// SymbolSize specifies the maximum raw payload size in bytes per packet.
	// Default is DefaultSymbolSize (1400 bytes).
	SymbolSize int

	// Checksum specifies whether recoded shards should have CRC32-Castagnoli checksum enabled.
	Checksum bool

	// Seed is an optional initial seed for SplitMix64 PRNG. When 0, a crypto seed is generated.
	Seed uint64
}

// SwarmRecoder buffers partial innovative shards received by an intermediate peer
// and generates new random linear combinations over GF(2) (recoded parity shards)
// without requiring full decoding of the underlying generation or file.
type SwarmRecoder struct {
	mu         sync.Mutex
	capacity   int
	symbolSize int
	checksum   bool
	rngState   uint64
	shards     []Shard
}

// NewSwarmRecoder initializes a SwarmRecoder with the provided configuration.
func NewSwarmRecoder(cfg RecoderConfig) *SwarmRecoder {
	capacity := cfg.Capacity
	if capacity <= 0 {
		capacity = 128
	}

	symbolSize := cfg.SymbolSize
	if symbolSize <= 0 {
		symbolSize = DefaultSymbolSize
	}

	seed := cfg.Seed
	if seed == 0 {
		var b [8]byte
		if _, err := rand.Read(b[:]); err == nil {
			seed = binary.BigEndian.Uint64(b[:])
		}
		if seed == 0 {
			seed = uint64(time.Now().UnixNano())
		}
	}

	return &SwarmRecoder{
		capacity:   capacity,
		symbolSize: symbolSize,
		checksum:   cfg.Checksum,
		rngState:   seed,
		shards:     make([]Shard, 0, capacity),
	}
}

// nextRand returns a 64-bit pseudo-random number using SplitMix64.
func (r *SwarmRecoder) nextRand() uint64 {
	r.rngState += 0x9e3779b97f4a7c15
	z := r.rngState
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// AddShard ingests a shard into the recoder pool.
// If the pool is full, it evicts the oldest shard in FIFO order.
func (r *SwarmRecoder) AddShard(shard Shard) bool {
	if len(shard.Data) == 0 || shard.Mask.IsZero() {
		return false
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	// Check if this exact shard is already present (same BaseSeq and Mask)
	for i := range r.shards {
		if r.shards[i].BaseSeq == shard.BaseSeq && r.shards[i].Mask == shard.Mask {
			return false
		}
	}

	if len(r.shards) >= r.capacity {
		// Evict oldest shard
		copy(r.shards, r.shards[1:])
		r.shards[len(r.shards)-1] = shard.Clone()
	} else {
		r.shards = append(r.shards, shard.Clone())
	}

	return true
}

// Recode generates a new random linear combination over GF(2) from the buffered shards.
// It randomly selects a subset of available shards belonging to compatible BaseSeq ranges,
// sums their masks and payloads using bitwise XOR, and emits an innovative parity shard.
func (r *SwarmRecoder) Recode() (Shard, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	n := len(r.shards)
	if n == 0 {
		return Shard{}, ErrInsufficientShards
	}

	if n == 1 {
		return r.shards[0].Clone(), nil
	}

	// Pick a reference shard at random to anchor the BaseSeq
	refIdx := int(r.nextRand() % uint64(n))
	refBaseSeq := r.shards[refIdx].BaseSeq

	// Find all shards that are compatible with refBaseSeq (span within 256 bits)
	candidates := make([]int, 0, n)
	minBase := refBaseSeq
	maxHighSeq := refBaseSeq + uint64(r.shards[refIdx].Mask.HighestBit())

	for i := range r.shards {
		s := &r.shards[i]
		sMin := s.BaseSeq
		sMax := s.BaseSeq + uint64(s.Mask.HighestBit())

		currMin := minBase
		if sMin < currMin {
			currMin = sMin
		}
		currMax := maxHighSeq
		if sMax > currMax {
			currMax = sMax
		}

		if (currMax - currMin) < 256 {
			candidates = append(candidates, i)
			minBase = currMin
			maxHighSeq = currMax
		}
	}

	if len(candidates) == 0 {
		return r.shards[refIdx].Clone(), nil
	}

	// Pick random non-zero binary coefficients for candidates
	var chosen []int
	for attempts := 0; attempts < 10; attempts++ {
		chosen = chosen[:0]
		for _, idx := range candidates {
			if (r.nextRand() & 1) == 1 {
				chosen = append(chosen, idx)
			}
		}
		if len(chosen) > 0 {
			break
		}
	}

	// Guarantee at least one shard chosen
	if len(chosen) == 0 {
		chosen = append(chosen, candidates[int(r.nextRand()%uint64(len(candidates)))])
	}

	// Calculate overall minimum BaseSeq among chosen shards
	effMinBase := r.shards[chosen[0]].BaseSeq
	for _, idx := range chosen[1:] {
		if r.shards[idx].BaseSeq < effMinBase {
			effMinBase = r.shards[idx].BaseSeq
		}
	}

	var combinedMask Bitset256
	maxLen := 0
	for _, idx := range chosen {
		if len(r.shards[idx].Data) > maxLen {
			maxLen = len(r.shards[idx].Data)
		}
	}

	combinedData := make([]byte, maxLen)

	for _, idx := range chosen {
		s := &r.shards[idx]
		shift := int(s.BaseSeq - effMinBase)
		shifted := s.Mask.ShiftLeft(shift)
		combinedMask.XOR(shifted)
		XORBytes(combinedData, s.Data, len(s.Data))
	}

	if combinedMask.IsZero() {
		// If XOR unexpectedly canceled all bits, return one of the chosen shards directly
		return r.shards[chosen[0]].Clone(), nil
	}

	return Shard{
		BaseSeq:  effMinBase,
		Mask:     combinedMask,
		Data:     combinedData,
		IsParity: true,
		Checksum: r.checksum,
	}, nil
}

// RecodeGeneration generates a recoded parity shard strictly from shards sharing the specified BaseSeq.
func (r *SwarmRecoder) RecodeGeneration(baseSeq uint64) (Shard, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	var matching []int
	for i := range r.shards {
		if r.shards[i].BaseSeq == baseSeq {
			matching = append(matching, i)
		}
	}

	if len(matching) == 0 {
		return Shard{}, ErrInsufficientShards
	}

	if len(matching) == 1 {
		return r.shards[matching[0]].Clone(), nil
	}

	var chosen []int
	for attempts := 0; attempts < 10; attempts++ {
		chosen = chosen[:0]
		for _, idx := range matching {
			if (r.nextRand() & 1) == 1 {
				chosen = append(chosen, idx)
			}
		}
		if len(chosen) > 0 {
			break
		}
	}

	if len(chosen) == 0 {
		chosen = append(chosen, matching[int(r.nextRand()%uint64(len(matching)))])
	}

	var combinedMask Bitset256
	maxLen := 0
	for _, idx := range chosen {
		if len(r.shards[idx].Data) > maxLen {
			maxLen = len(r.shards[idx].Data)
		}
	}

	combinedData := make([]byte, maxLen)
	for _, idx := range chosen {
		s := &r.shards[idx]
		combinedMask.XOR(s.Mask)
		XORBytes(combinedData, s.Data, len(s.Data))
	}

	if combinedMask.IsZero() {
		return r.shards[chosen[0]].Clone(), nil
	}

	return Shard{
		BaseSeq:  baseSeq,
		Mask:     combinedMask,
		Data:     combinedData,
		IsParity: true,
		Checksum: r.checksum,
	}, nil
}

// Len returns the current number of shards in the recoder pool.
func (r *SwarmRecoder) Len() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.shards)
}

// Reset clears all shards from the recoder pool.
func (r *SwarmRecoder) Reset() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.shards = r.shards[:0]
}
