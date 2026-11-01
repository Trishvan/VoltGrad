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

func BenchmarkMatMulForwardResident(b *testing.B) {
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
	pt, err := NewPersistentGpuTensor(hw, w)
	if err != nil {
		b.Fatalf("Failed creating persistent tensor: %v", err)
	}
	defer pt.Free()

	x := make([]int32, inDim)
	for i := range x {
		x[i] = int32((i % 20) - 10)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = gpu.MatMulForwardResident(pt.BufferPacked(), nil, x, outDim, inDim, w.BitWidth(), 16)
	}
}

func BenchmarkBatchedGEMMResident(b *testing.B) {
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
	pt, err := NewPersistentGpuTensor(hw, w)
	if err != nil {
		b.Fatalf("Failed creating persistent tensor: %v", err)
	}
	defer pt.Free()

	x := make([]int32, batchSize*inDim)
	for i := range x {
		x[i] = int32((i % 20) - 10)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = gpu.MatMulBatchedForwardResident(pt.BufferPacked(), nil, x, batchSize, outDim, inDim, w.BitWidth(), 16)
	}
}

func BenchmarkDBWOptimizerStepResident(b *testing.B) {
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
	pt, err := NewPersistentGpuTensor(hw, p)
	if err != nil {
		b.Fatalf("Failed creating persistent tensor: %v", err)
	}
	defer pt.Free()

	config := DefaultOptimizerConfig()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = gpu.OptimizerStepResident(pt.BufferInt64(), pt.BufferGrad(), pt.BufferVel(), pt.BufferPacked(), n, 4, config)
	}
}

func BenchmarkTrainingStepCPU(b *testing.B) {
	batchSize := 32
	model := NewExampleMLPArchitecture()
	model.fc1.InitWeightsIdentity(Q16One)
	model.fc2.InitWeightsIdentity(Q16One)
	config := DefaultTrainingConfig()
	opt := NewDynamicBitOptimizer(config.OptimizerConfig, model)
	cpuCtx := NewComputeContext(nil)

	flatX := make([]int32, batchSize*4)
	flatY := make([]int32, batchSize*2)
	for i := range flatX {
		flatX[i] = 32768
	}
	for i := range flatY {
		flatY[i] = 65536
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.ZeroGrad()
		preds, _ := model.ForwardBatched(cpuCtx, flatX, batchSize)
		_, dLoss, _ := ComputeLoss(LossMSE, preds, flatY, config.OptimizerConfig.FractionalBits)
		_ = model.BackwardBatched(cpuCtx, dLoss, batchSize)
		_, _ = opt.Step(model)
	}
}

func BenchmarkTrainingStepGPUPersistent(b *testing.B) {
	hw, err := NewHardwareContext()
	if err != nil {
		b.Skipf("Skipping Vulkan benchmark: %v", err)
	}
	defer hw.Close()

	gpu, err := NewGpuEngine(hw)
	if err != nil {
		b.Fatalf("Failed creating GPU engine: %v", err)
	}
	defer gpu.Close()

	batchSize := 32
	model := NewExampleMLPArchitecture()
	model.fc1.InitWeightsIdentity(Q16One)
	model.fc2.InitWeightsIdentity(Q16One)
	if err := model.BindGPU(hw); err != nil {
		b.Fatalf("BindGPU failed: %v", err)
	}
	defer model.FreeGPU()

	config := DefaultTrainingConfig()
	opt := NewDynamicBitOptimizer(config.OptimizerConfig, model)
	ctx := NewComputeContext(nil)
	ctx.SetGpuEngine(gpu)

	flatX := make([]int32, batchSize*4)
	flatY := make([]int32, batchSize*2)
	for i := range flatX {
		flatX[i] = 32768
	}
	for i := range flatY {
		flatY[i] = 65536
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.ZeroGrad()
		preds, _ := model.ForwardBatched(ctx, flatX, batchSize)
		_, dLoss, _ := gpu.MSELoss(preds, flatY, config.OptimizerConfig.FractionalBits)
		_ = model.BackwardBatched(ctx, dLoss, batchSize)
		_, _ = opt.StepGPU(gpu, model)
	}
}
