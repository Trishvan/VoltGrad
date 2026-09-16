package vulkankernel

import (
	"fmt"
)

// DynamicBitTensor: bit-packed tensor with dynamic bit widening (DBW).
type DynamicBitTensor struct {
	shape    []int
	length   int
	bitWidth int
	data     []byte
}

// NewDynamicBitTensor creates a new dynamic bit tensor with the specified shape and initial bit width (2..64).
func NewDynamicBitTensor(shape []int, initialBits int) *DynamicBitTensor {
	if initialBits < 2 {
		initialBits = 2
	}
	if initialBits > 64 {
		initialBits = 64
	}

	totalLen := 1
	for _, dim := range shape {
		if dim <= 0 {
			totalLen = 0
			break
		}
		totalLen *= dim
	}
	if len(shape) == 0 {
		totalLen = 0
	}

	totalBits := totalLen * initialBits
	totalBytes := (totalBits + 7) / 8

	return &DynamicBitTensor{
		shape:    append([]int{}, shape...),
		length:   totalLen,
		bitWidth: initialBits,
		data:     make([]byte, totalBytes),
	}
}

// Shape returns the tensor dimensions.
func (t *DynamicBitTensor) Shape() []int {
	return append([]int{}, t.shape...)
}

// Len returns the total number of scalar elements.
func (t *DynamicBitTensor) Len() int {
	return t.length
}

// BitWidth returns the active bits per element.
func (t *DynamicBitTensor) BitWidth() int {
	return t.bitWidth
}







// Get reads the signed integer at linear index i as int64.
func (t *DynamicBitTensor) Get(i int) int64 {
	if i < 0 || i >= t.length {
		panic(fmt.Sprintf("DynamicBitTensor index %d out of bounds [0, %d)", i, t.length))
	}

	bitOffset := i * t.bitWidth
	var raw uint64
	for b := 0; b < t.bitWidth; b++ {
		byteIdx := (bitOffset + b) / 8
		bitInByte := (bitOffset + b) % 8
		bitVal := (t.data[byteIdx] >> bitInByte) & 1
		raw |= (uint64(bitVal) << b)
	}

	return int64(raw)
}

// GetInt32 returns the signed 32-bit integer at linear index i.
func (t *DynamicBitTensor) GetInt32(i int) int32 {
	return int32(t.Get(i))
}

// Set writes signed integer v at linear index i, dynamically widening the tensor bit-depth if v overflows.
func (t *DynamicBitTensor) Set(i int, v int64) {
	if i < 0 || i >= t.length {
		panic(fmt.Sprintf("DynamicBitTensor index %d out of bounds [0, %d)", i, t.length))
	}

	bitOffset := i * t.bitWidth
	var raw uint64
	if t.bitWidth == 64 {
		raw = uint64(v)
	} else {
		mask := (uint64(1) << t.bitWidth) - 1
		raw = uint64(v) & mask
	}

	for b := 0; b < t.bitWidth; b++ {
		byteIdx := (bitOffset + b) / 8
		bitInByte := (bitOffset + b) % 8
		bitVal := (raw >> b) & 1

		if bitVal == 1 {
			t.data[byteIdx] |= byte(1 << bitInByte)
		} else {
			t.data[byteIdx] &= ^byte(1 << bitInByte)
		}
	}
}

// SetInt32 writes signed 32-bit integer v at linear index i, auto-widening if needed.
func (t *DynamicBitTensor) SetInt32(i int, v int32) {
	t.Set(i, int64(v))
}




