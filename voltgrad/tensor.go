package voltgrad

import (
	"fmt"
	"math"
)

// DynamicBitTensor represents an arbitrary bit-width signed integer tensor with Dynamic Bit Widening (DBW).
// It stores weights, gradients, or activations packed into compact bitstreams.
//
// Whenever any parameter update or gradient accumulation exceeds the current bit-width representation
// limit [-2^(W-1), 2^(W-1)-1], the tensor automatically and dynamically widens its bit allocation
// (e.g., 2-bit -> 3-bit -> ... -> 16-bit -> 32-bit -> 64-bit) and re-packs the underlying raw bitstream
// with zero clipping and zero quantization noise.
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

// MemoryBytes returns the actual physical bytes consumed by the packed bitstream.
func (t *DynamicBitTensor) MemoryBytes() int {
	return len(t.data)
}

// UncompressedInt32Bytes returns the memory bytes that standard 32-bit integers would require.
func (t *DynamicBitTensor) UncompressedInt32Bytes() int {
	return t.length * 4
}

// CompressionRatio returns the memory savings ratio over 32-bit storage.
func (t *DynamicBitTensor) CompressionRatio() float64 {
	if len(t.data) == 0 {
		return 1.0
	}
	return float64(t.UncompressedInt32Bytes()) / float64(len(t.data))
}

// BitsNeeded64 calculates the minimum bit width (2..64) required to represent signed integer v.
func BitsNeeded64(v int64) int {
	for w := 2; w < 64; w++ {
		minVal, maxVal := MinMax64(w)
		if v >= minVal && v <= maxVal {
			return w
		}
	}
	return 64
}

// BitsNeeded32 calculates the minimum bit width required to represent signed 32-bit integer v.
func BitsNeeded32(v int32) int {
	return BitsNeeded64(int64(v))
}

// MinMax64 returns the representable [min, max] range for signed integers of bit width W (2..64).
func MinMax64(width int) (minVal, maxVal int64) {
	if width <= 1 {
		return 0, 0
	}
	if width >= 64 {
		return math.MinInt64, math.MaxInt64
	}
	half := int64(1) << (width - 1)
	return -half, half - 1
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

	if t.bitWidth == 64 {
		return int64(raw)
	}

	mask := (uint64(1) << t.bitWidth) - 1
	signBit := uint64(1) << (t.bitWidth - 1)
	if raw&signBit != 0 {
		return int64(raw | ^mask)
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

	needed := BitsNeeded64(v)
	if needed > t.bitWidth {
		t.Widen(needed)
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

// Add accumulates delta into element i, auto-widening if the result overflows.
func (t *DynamicBitTensor) Add(i int, delta int64) int64 {
	curr := t.Get(i)
	res := curr + delta
	t.Set(i, res)
	return res
}

// Widen dynamically re-packs all elements into a wider bit-width buffer without precision loss.
func (t *DynamicBitTensor) Widen(newBitWidth int) {
	if newBitWidth <= t.bitWidth {
		return
	}
	if newBitWidth > 64 {
		newBitWidth = 64
	}

	oldValues := make([]int64, t.length)
	for i := 0; i < t.length; i++ {
		oldValues[i] = t.Get(i)
	}

	t.bitWidth = newBitWidth
	totalBytes := (t.length*newBitWidth + 7) / 8
	t.data = make([]byte, totalBytes)

	for i, v := range oldValues {
		bitOffset := i * newBitWidth
		var raw uint64
		if newBitWidth == 64 {
			raw = uint64(v)
		} else {
			mask := (uint64(1) << newBitWidth) - 1
			raw = uint64(v) & mask
		}

		for b := 0; b < newBitWidth; b++ {
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
}

// ToInt32Slice unrolls the packed tensor into a contiguous []int32 slice (for Vulkan buffer uploads).
func (t *DynamicBitTensor) ToInt32Slice() []int32 {
	out := make([]int32, t.length)
	for i := 0; i < t.length; i++ {
		out[i] = t.GetInt32(i)
	}
	return out
}

// ToInt64Slice unrolls the packed tensor into a contiguous []int64 slice without truncation.
func (t *DynamicBitTensor) ToInt64Slice() []int64 {
	out := make([]int64, t.length)
	for i := 0; i < t.length; i++ {
		out[i] = t.Get(i)
	}
	return out
}

// FromInt32Slice copies contiguous values into the tensor and updates DBW bit-depth to the minimal fit.
func (t *DynamicBitTensor) FromInt32Slice(values []int32) {
	if len(values) != t.length {
		panic(fmt.Sprintf("slice length %d does not match tensor length %d", len(values), t.length))
	}
	maxBits := 2
	for _, v := range values {
		b := BitsNeeded32(v)
		if b > maxBits {
			maxBits = b
		}
	}
	if maxBits > t.bitWidth {
		t.Widen(maxBits)
	}
	for i, v := range values {
		t.SetInt32(i, v)
	}
}

// FromInt64Slice copies contiguous 64-bit values into the tensor and updates DBW bit-depth to the minimal fit.
func (t *DynamicBitTensor) FromInt64Slice(values []int64) {
	if len(values) != t.length {
		panic(fmt.Sprintf("slice length %d does not match tensor length %d", len(values), t.length))
	}
	maxBits := 2
	for _, v := range values {
		b := BitsNeeded64(v)
		if b > maxBits {
			maxBits = b
		}
	}
	if maxBits > t.bitWidth {
		t.Widen(maxBits)
	}
	for i, v := range values {
		t.Set(i, v)
	}
}

// Clone returns an independent deep copy of the DynamicBitTensor.
func (t *DynamicBitTensor) Clone() *DynamicBitTensor {
	dataCopy := make([]byte, len(t.data))
	copy(dataCopy, t.data)
	shapeCopy := make([]int, len(t.shape))
	copy(shapeCopy, t.shape)
	return &DynamicBitTensor{
		shape:    shapeCopy,
		length:   t.length,
		bitWidth: t.bitWidth,
		data:     dataCopy,
	}
}

// Words returns the underlying packed bitstream as a contiguous []uint32 slice for direct GPU buffer binding.
func (t *DynamicBitTensor) Words() []uint32 {
	numWords := (len(t.data) + 3) / 4
	if numWords == 0 {
		return []uint32{0}
	}
	words := make([]uint32, numWords)
	for i := 0; i < len(t.data); i++ {
		wordIdx := i / 4
		byteInWord := i % 4
		words[wordIdx] |= uint32(t.data[i]) << (byteInWord * 8)
	}
	return words
}

// SetWords updates the packed bitstream directly from GPU-computed []uint32 words at the specified bit-width.
func (t *DynamicBitTensor) SetWords(words []uint32, bitWidth int) {
	t.bitWidth = bitWidth
	totalBytes := (t.length*bitWidth + 7) / 8
	t.data = make([]byte, totalBytes)
	for i := 0; i < totalBytes; i++ {
		wordIdx := i / 4
		byteInWord := i % 4
		if wordIdx < len(words) {
			t.data[i] = byte((words[wordIdx] >> (byteInWord * 8)) & 0xFF)
		}
	}
}
