package badrlnc

import (
	"sync"
)

const (
	// DefaultBufferSize is the standard buffer allocation size (2048 bytes),
	// comfortably holding a full MTU IP packet, RLNC shard headers, and crypto tags.
	DefaultBufferSize = 2048

	// MaxShardSize is the maximum buffer size allocated in pools to hold serialized shards.
	MaxShardSize = 2048

	// MaxPacketBufferSize is the buffer capacity allocated for packets in pools.
	MaxPacketBufferSize = 2048
)

// ShardBufferPool manages reusable byte buffers for serialized shards.
// Uses sync.Pool storing array pointers to achieve true 0 heap allocations without interface boxing.
type ShardBufferPool struct {
	pool sync.Pool
}

var defaultShardPool = NewShardBufferPool()

// NewShardBufferPool creates a pool with buffers of MaxShardSize (2048 bytes) capacity.
func NewShardBufferPool() *ShardBufferPool {
	p := &ShardBufferPool{}
	p.pool.New = func() any {
		var buf [MaxShardSize]byte
		return &buf
	}
	return p
}

// Get retrieves a shard buffer with capacity equal to MaxShardSize.
func (p *ShardBufferPool) Get() []byte {
	bufPtr := p.pool.Get().(*[MaxShardSize]byte)
	return bufPtr[:]
}

// Put returns a shard buffer to the pool with zero allocations.
func (p *ShardBufferPool) Put(b []byte) {
	if cap(b) < MaxShardSize {
		return
	}
	bufPtr := (*[MaxShardSize]byte)(b[:MaxShardSize])
	p.pool.Put(bufPtr)
}

// GetShardBuffer borrows a slice of MaxShardSize from the global pool.
func GetShardBuffer() []byte {
	return defaultShardPool.Get()
}

// PutShardBuffer recycles a slice back into the global shard pool.
func PutShardBuffer(b []byte) {
	defaultShardPool.Put(b)
}

// PacketBufferPool manages reusable buffers for raw IP packets and application datagrams.
type PacketBufferPool struct {
	pool sync.Pool
}

var defaultPacketPool = NewPacketBufferPool()

// NewPacketBufferPool creates a pool of packet buffers of MaxPacketBufferSize capacity.
func NewPacketBufferPool() *PacketBufferPool {
	p := &PacketBufferPool{}
	p.pool.New = func() any {
		var buf [MaxPacketBufferSize]byte
		return &buf
	}
	return p
}

// Get borrows a packet buffer from the pool.
func (p *PacketBufferPool) Get() []byte {
	bufPtr := p.pool.Get().(*[MaxPacketBufferSize]byte)
	return bufPtr[:]
}

// Put recycles a packet buffer with zero allocations.
func (p *PacketBufferPool) Put(b []byte) {
	if cap(b) < MaxPacketBufferSize {
		return
	}
	bufPtr := (*[MaxPacketBufferSize]byte)(b[:MaxPacketBufferSize])
	p.pool.Put(bufPtr)
}

// GetPacketBuffer borrows a buffer from the default packet pool.
func GetPacketBuffer() []byte {
	return defaultPacketPool.Get()
}

// PutPacketBuffer recycles a buffer into the default packet pool.
func PutPacketBuffer(b []byte) {
	defaultPacketPool.Put(b)
}
