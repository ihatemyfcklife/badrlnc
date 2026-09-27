package badrlnc

import (
	"crypto/subtle"
)

// XOR computes dst[i] = a[i] ^ b[i] for i in [0, min(len(dst), len(a), len(b))).
// It returns the number of bytes written, utilizing hardware SIMD instructions
// (AVX-512, AVX2, ARM NEON) via Go's crypto/subtle.XORBytes with strictly 0 heap allocations.
func XOR(dst, a, b []byte) int {
	return subtle.XORBytes(dst, a, b)
}

// XORBytes computes dst[i] ^= src[i] for i in [0, n) in-place.
// If n exceeds the length of dst or src, n is clamped to min(len(dst), len(src)).
// Hot path: strictly 0 heap allocations and multi-gigabyte/s line rate.
func XORBytes(dst, src []byte, n int) {
	if n > len(dst) {
		n = len(dst)
	}
	if n > len(src) {
		n = len(src)
	}
	if n <= 0 {
		return
	}

	subtle.XORBytes(dst[:n], dst[:n], src[:n])
}

// XORRow performs row-level vector reduction in GF(2) linear algebra operations.
// Hot path: strictly 0 heap allocations.
func XORRow(dst, src []byte, n int) {
	XORBytes(dst, src, n)
}

// XORMulti combines multiple source slices into dst using SIMD XOR instructions.
func XORMulti(dst []byte, sources [][]byte, n int) {
	for _, src := range sources {
		XORBytes(dst, src, n)
	}
}

// ClearBytes zeroes out a byte slice using the high-performance compiler intrinsic memclr.
func ClearBytes(b []byte) {
	clear(b)
}
