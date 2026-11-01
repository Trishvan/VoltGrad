package voltgrad

import (
	"testing"
)

func BenchmarkCPUMatMul(b *testing.B) {
	inDim, outDim := 64, 64
	w := NewDynamicBitTensor([]int{outDim, inDim}, 8)
	for i := 0; i < w.Len(); i++ {
		w.Set(i, int64((i%15)-7))
	}
	x := make([]int32, inDim)
	for i := range x {
		x[i] = int32((i % 20) - 10)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = MatMulVec64(w, x, 16)
	}
}

func BenchmarkGPUMatMul(b *testing.B) {
	hw, err := NewHardwareContext()
	if err != nil {
		b.Skipf("Skipping Vulkan benchmark: %v", err)
	}
	defer hw.Close()

	gpu, err := NewGpuEngine(hw)
	if err != nil {
		b.Fatalf("Failed to init GPU engine: %v", err)
	}
	defer gpu.Close()

	inDim, outDim := 64, 64
	w := NewDynamicBitTensor([]int{outDim, inDim}, 8)
	for i := 0; i < w.Len(); i++ {
		w.Set(i, int64((i%15)-7))
	}
	x := make([]int32, inDim)
	for i := range x {
		x[i] = int32((i % 20) - 10)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = gpu.MatMulForward(w, x, nil, 16)
	}
}

func BenchmarkBatchedGEMM(b *testing.B) {
	hw, err := NewHardwareContext()
	if err != nil {
		b.Skipf("Skipping Vulkan benchmark: %v", err)
	}
	defer hw.Close()

	gpu, err := NewGpuEngine(hw)
	if err != nil {
		b.Fatalf("Failed to init GPU engine: %v", err)
	}
	defer gpu.Close()

	batchSize := 32
	inDim, outDim := 64, 64
	w := NewDynamicBitTensor([]int{outDim, inDim}, 8)
	for i := 0; i < w.Len(); i++ {
		w.Set(i, int64((i%15)-7))
	}
	x := make([]int32, batchSize*inDim)
	for i := range x {
		x[i] = int32((i % 20) - 10)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = gpu.MatMulBatchedForward(w, x, batchSize, nil, 16)
	}
}

func BenchmarkDBWOptimizerStep(b *testing.B) {
	hw, err := NewHardwareContext()
	if err != nil {
		b.Skipf("Skipping Vulkan benchmark: %v", err)
	}
	defer hw.Close()

	gpu, err := NewGpuEngine(hw)
	if err != nil {
		b.Fatalf("Failed to init GPU engine: %v", err)
	}
	defer gpu.Close()

	n := 1024
	p := NewDynamicBitTensor([]int{n}, 4)
	g := NewDynamicBitTensor([]int{n}, 16)
	v := NewDynamicBitTensor([]int{n}, 16)

	config := DefaultOptimizerConfig()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = gpu.OptimizerStep(p, g, v, config)
	}
}

func BenchmarkPersistentGpuTensorSync(b *testing.B) {
	hw, err := NewHardwareContext()
	if err != nil {
		b.Skipf("Skipping Vulkan benchmark: %v", err)
	}
	defer hw.Close()

	n := 1024
	t := NewDynamicBitTensor([]int{n}, 8)
	pt, err := NewPersistentGpuTensor(hw, t)
	if err != nil {
		b.Fatalf("NewPersistentGpuTensor failed: %v", err)
	}
	defer pt.Free()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = pt.SyncToGPU()
		_ = pt.SyncToHost()
	}
}
