package rlnc

import (
	"encoding/binary"
	"errors"
)

const (
	// CompactHeaderSize is the size in bytes of the RLNC shard header for 64-bit masks.
	// Binary Layout (24 bytes, 8-byte aligned):
	//  [0..7]   : BaseSeq  (uint64 BigEndian) - Base packet sequence number for bit 0 of Mask
	//  [8..15]  : Mask[0]  (uint64 BigEndian) - Primary 64-bit GF(2) linear combination bitmask
	//  [16..17] : DataLen  (uint16 BigEndian) - Byte length of the shard payload
	//  [18]     : Flags    (uint8)            - Bit 0: FlagParity, Bit 1: FlagExtendedMask
	//  [19]     : Version  (uint8)            - Protocol Version (0x01)
	//  [20..23] : Reserved (uint32 BigEndian) - Alignment and future extensions
	CompactHeaderSize = 24

	// ExtendedHeaderSize is the size in bytes of the RLNC shard header for 256-bit masks.
	// Binary Layout (48 bytes, 8-byte aligned):
	//  [0..23]  : Same as CompactHeaderSize (with FlagExtendedMask set)
	//  [24..31] : Mask[1]  (uint64 BigEndian)
	//  [32..39] : Mask[2]  (uint64 BigEndian)
	//  [40..47] : Mask[3]  (uint64 BigEndian)
	ExtendedHeaderSize = 48

	// ProtocolVersion1 represents the current RLNC wire format version.
	ProtocolVersion1 = 0x01

	// FlagParity indicates that this symbol is a linear combination of multiple source packets.
	FlagParity = 0x01

	// FlagExtendedMask indicates that the shard header contains a 256-bit Bitset256 mask (48 bytes).
	FlagExtendedMask = 0x02

	// LengthPrefixSize is the 2-byte prefix storing variable-length packet size inside the GF(2) symbol.
	LengthPrefixSize = 2

	// DefaultWindowSize is the standard sliding window size for low-latency real-time RLNC.
	DefaultWindowSize = 32

	// MaxWindowSize is the maximum sliding window size for 64-bit masks.
	MaxWindowSize = 64

	// MaxExtendedWindowSize is the maximum sliding window size for Bitset256 masks.
	MaxExtendedWindowSize = 256

	// DefaultSymbolSize is the default maximum symbol payload capacity in bytes.
	DefaultSymbolSize = 1400

	// DefaultCapacity is the default slot capacity for encoder and decoder ring buffers.
	DefaultCapacity = 2048
)

var (
	ErrZeroPayload         = errors.New("rlnc: payload cannot be empty")
	ErrPayloadTooLarge     = errors.New("rlnc: payload exceeds configured SymbolSize")
	ErrCorruptHeader       = errors.New("rlnc: shard header corrupted or smaller than expected size")
	ErrInvalidVersion      = errors.New("rlnc: invalid protocol version")
	ErrBufferTooSmall      = errors.New("rlnc: destination buffer too small for operation")
	ErrWindowOverflow      = errors.New("rlnc: sliding window overflow")
	ErrSequenceOutOfWindow = errors.New("rlnc: sequence number is outside active window")
	ErrNoPivotsAvailable   = errors.New("rlnc: no linear pivots available")
	ErrLinearlyDependent   = errors.New("rlnc: symbol is linearly dependent (zero innovation)")
)

// Shard represents a generic, self-contained Random Linear Network Coding symbol.
// It encapsulates the base sequence number, the GF(2) innovation bitmask, and the payload.
type Shard struct {
	BaseSeq  uint64
	Mask     Bitset256
	Data     []byte
	IsParity bool
}

// SystematicShard is an alias for Shard representing an uncoded source symbol (Mask weight = 1).
type SystematicShard = Shard

// ParityShard is an alias for Shard representing a GF(2) linear combination of source symbols.
type ParityShard = Shard

// Seq returns the packet sequence number for a systematic shard (BaseSeq + trailing zeros).
func (s Shard) Seq() uint64 {
	tz := s.Mask.TrailingZeros()
	if tz < 0 {
		return s.BaseSeq
	}
	return s.BaseSeq + uint64(tz)
}

// HeaderSize returns the serialized wire header size in bytes (24 or 48 bytes).
func (s Shard) HeaderSize() int {
	if s.Mask.IsUint64() {
		return CompactHeaderSize
	}
	return ExtendedHeaderSize
}

// TotalWireSize returns the total byte length of the serialized shard (Header + Data).
func (s Shard) TotalWireSize() int {
	return s.HeaderSize() + len(s.Data)
}

// Clone creates a deep copy of the Shard, allocating a new slice for Data.
func (s Shard) Clone() Shard {
	cp := make([]byte, len(s.Data))
	copy(cp, s.Data)
	return Shard{
		BaseSeq:  s.BaseSeq,
		Mask:     s.Mask,
		Data:     cp,
		IsParity: s.IsParity,
	}
}

// EncodeTo serializes the Shard into dst with strictly zero heap allocations.
// Returns the total bytes written into dst.
func (s Shard) EncodeTo(dst []byte) (int, error) {
	if len(s.Data) > 65535 {
		return 0, ErrPayloadTooLarge
	}

	hdrSize := s.HeaderSize()
	totalLen := hdrSize + len(s.Data)
	if len(dst) < totalLen {
		return 0, ErrBufferTooSmall
	}

	_ = dst[hdrSize-1] // BCE guarantee

	binary.BigEndian.PutUint64(dst[0:8], s.BaseSeq)
	binary.BigEndian.PutUint64(dst[8:16], s.Mask[0])
	binary.BigEndian.PutUint16(dst[16:18], uint16(len(s.Data)))

	var flags uint8
	if s.IsParity || s.Mask.Weight() > 1 {
		flags |= FlagParity
	}
	if hdrSize == ExtendedHeaderSize {
		flags |= FlagExtendedMask
	}
	dst[18] = flags
	dst[19] = ProtocolVersion1
	binary.BigEndian.PutUint32(dst[20:24], 0) // Reserved

	if hdrSize == ExtendedHeaderSize {
		binary.BigEndian.PutUint64(dst[24:32], s.Mask[1])
		binary.BigEndian.PutUint64(dst[32:40], s.Mask[2])
		binary.BigEndian.PutUint64(dst[40:48], s.Mask[3])
	}

	copy(dst[hdrSize:totalLen], s.Data)
	return totalLen, nil
}

// MarshalBinary serializes the Shard into a new byte slice implementing encoding.BinaryMarshaler.
func (s Shard) MarshalBinary() ([]byte, error) {
	buf := make([]byte, s.TotalWireSize())
	n, err := s.EncodeTo(buf)
	if err != nil {
		return nil, err
	}
	return buf[:n], nil
}

// DecodeShard parses a raw byte slice into a Shard with zero heap allocations.
// Shard.Data points directly into the provided src slice.
func DecodeShard(src []byte) (Shard, error) {
	if len(src) < CompactHeaderSize {
		return Shard{}, ErrCorruptHeader
	}
	_ = src[23] // BCE guarantee

	version := src[19]
	if version != ProtocolVersion1 {
		return Shard{}, ErrInvalidVersion
	}

	flags := src[18]
	hdrSize := CompactHeaderSize
	if (flags & FlagExtendedMask) != 0 {
		hdrSize = ExtendedHeaderSize
		if len(src) < ExtendedHeaderSize {
			return Shard{}, ErrCorruptHeader
		}
		_ = src[47] // BCE guarantee
	}

	dataLen := int(binary.BigEndian.Uint16(src[16:18]))
	expectedLen := hdrSize + dataLen
	if len(src) < expectedLen {
		return Shard{}, ErrCorruptHeader
	}

	baseSeq := binary.BigEndian.Uint64(src[0:8])
	var mask Bitset256
	mask[0] = binary.BigEndian.Uint64(src[8:16])

	if hdrSize == ExtendedHeaderSize {
		mask[1] = binary.BigEndian.Uint64(src[24:32])
		mask[2] = binary.BigEndian.Uint64(src[32:40])
		mask[3] = binary.BigEndian.Uint64(src[40:48])
	}

	return Shard{
		BaseSeq:  baseSeq,
		Mask:     mask,
		Data:     src[hdrSize:expectedLen],
		IsParity: (flags & FlagParity) != 0,
	}, nil
}

// UnmarshalBinary decodes raw bytes into the receiver implementing encoding.BinaryUnmarshaler.
func (s *Shard) UnmarshalBinary(data []byte) error {
	decoded, err := DecodeShard(data)
	if err != nil {
		return err
	}
	// Copy data slice to ensure ownership when unmarshaling
	*s = decoded.Clone()
	return nil
}
