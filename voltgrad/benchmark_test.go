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

// BenchmarkMLPArchitecture defines a configurable 2-layer MLP for scaling benchmarks.
type BenchmarkMLPArchitecture struct {
	fc1 *LinearLayer
	fc2 *LinearLayer
}

func NewBenchmarkMLPArchitecture(inDim, hiddenDim, outDim int) *BenchmarkMLPArchitecture {
	return &BenchmarkMLPArchitecture{
		fc1: NewLinearLayer("fc1", inDim, hiddenDim, 8, true),
		fc2: NewLinearLayer("fc2", hiddenDim, outDim, 8, true),
	}
}

func (m *BenchmarkMLPArchitecture) Name() string { return "BenchmarkMLP" }

func (m *BenchmarkMLPArchitecture) Layers() []LayerConfig {
	return []LayerConfig{
		{Name: "fc1", InDim: m.fc1.InDim, OutDim: m.fc1.OutDim, InitialBitWidth: 8, UseBias: true},
		{Name: "fc2", InDim: m.fc2.InDim, OutDim: m.fc2.OutDim, InitialBitWidth: 8, UseBias: true},
	}
}

func (m *BenchmarkMLPArchitecture) Parameters() []*DynamicBitTensor {
	return []*DynamicBitTensor{m.fc1.Weight, m.fc1.Bias, m.fc2.Weight, m.fc2.Bias}
}

func (m *BenchmarkMLPArchitecture) Gradients() []*DynamicBitTensor {
	return []*DynamicBitTensor{m.fc1.GradW, m.fc1.GradBias, m.fc2.GradW, m.fc2.GradBias}
}

func (m *BenchmarkMLPArchitecture) ZeroGrad() {
	m.fc1.ZeroGrad()
	m.fc2.ZeroGrad()
}

func (m *BenchmarkMLPArchitecture) BindGPU(hw *HardwareContext) error {
	if err := m.fc1.BindGPU(hw); err != nil {
		return err
	}
	if err := m.fc2.BindGPU(hw); err != nil {
		m.fc1.FreeGPU()
		return err
	}
	return nil
}

func (m *BenchmarkMLPArchitecture) SyncToHost() error {
	if err := m.fc1.SyncToHost(); err != nil {
		return err
	}
	return m.fc2.SyncToHost()
}

func (m *BenchmarkMLPArchitecture) FreeGPU() {
	m.fc1.FreeGPU()
	m.fc2.FreeGPU()
}

func (m *BenchmarkMLPArchitecture) Forward(ctx *ComputeContext, x []int32) ([]int32, error) {
	h1, err := m.fc1.ForwardCtx(ctx, x, Q16FractionalBits)
	if err != nil {
		return nil, err
	}
	h1Act := ReLU(h1)
	ctx.SetScratch("input_x", x)
	ctx.SetScratch("h1_act", h1Act)
	return m.fc2.ForwardCtx(ctx, h1Act, Q16FractionalBits)
}

func (m *BenchmarkMLPArchitecture) Backward(ctx *ComputeContext, dLoss []int32) error {
	x, _ := ctx.GetScratch("input_x")
	h1Act, _ := ctx.GetScratch("h1_act")
	dH1Act, err := m.fc2.BackwardCtx(ctx, dLoss, h1Act.([]int32), Q16FractionalBits)
	if err != nil {
		return err
	}
	dH1 := ReLUBackward(dH1Act, h1Act.([]int32))
	_, err = m.fc1.BackwardCtx(ctx, dH1, x.([]int32), Q16FractionalBits)
	return err
}

func (m *BenchmarkMLPArchitecture) ForwardBatched(ctx *ComputeContext, x []int32, batchSize int) ([]int32, error) {
	h1, err := m.fc1.ForwardBatchedCtx(ctx, x, batchSize, Q16FractionalBits)
	if err != nil {
		return nil, err
	}
	var h1Act []int32
	if ctx != nil && ctx.IsGPU() {
		h1Act, err = ctx.GpuEngine().ReLUForward(h1)
		if err != nil {
			return nil, err
		}
	} else {
		h1Act = ReLU(h1)
	}
	ctx.SetScratch("batched_x", x)
	ctx.SetScratch("batched_h1_act", h1Act)
	return m.fc2.ForwardBatchedCtx(ctx, h1Act, batchSize, Q16FractionalBits)
}

func (m *BenchmarkMLPArchitecture) BackwardBatched(ctx *ComputeContext, dLoss []int32, batchSize int) error {
	x, _ := ctx.GetScratch("batched_x")
	h1Act, _ := ctx.GetScratch("batched_h1_act")
	dH1Act, err := m.fc2.BackwardBatchedCtx(ctx, dLoss, h1Act.([]int32), batchSize, Q16FractionalBits)
	if err != nil {
		return err
	}
	var dH1 []int32
	var err2 error
	if ctx != nil && ctx.IsGPU() {
		dH1, err2 = ctx.GpuEngine().ReLUBackward(dH1Act, h1Act.([]int32))
		if err2 != nil {
			return err2
		}
	} else {
		dH1 = ReLUBackward(dH1Act, h1Act.([]int32))
	}
	_, err = m.fc1.BackwardBatchedCtx(ctx, dH1, x.([]int32), batchSize, Q16FractionalBits)
	return err
}

func BenchmarkBatchedGEMM_Large_128x128_CPU(b *testing.B) {
	batchSize := 64
	inDim, outDim := 128, 128
	w := NewDynamicBitTensor([]int{outDim, inDim}, 8)
	for i := 0; i < w.Len(); i++ {
		w.Set(i, int64((i%15)-7))
	}
	x := make([]int32, batchSize*inDim)
	for i := range x {
		x[i] = int32((i % 20) - 10)
	}
	layer := NewLinearLayer("gemm", inDim, outDim, 8, false)
	layer.Weight = w

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = layer.ForwardBatched(x, batchSize, 16)
	}
}

func BenchmarkBatchedGEMM_Large_128x128_Transient(b *testing.B) {
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

	batchSize := 64
	inDim, outDim := 128, 128
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

func BenchmarkBatchedGEMM_Large_128x128_Resident(b *testing.B) {
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

	batchSize := 64
	inDim, outDim := 128, 128
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

func BenchmarkBatchedGEMM_Large_128x128_Resident_B128(b *testing.B) {
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

	batchSize := 128
	inDim, outDim := 128, 128
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

func BenchmarkTrainingStep_LargeModel_CPU(b *testing.B) {
	batchSize := 64
	model := NewBenchmarkMLPArchitecture(128, 128, 64)
	model.fc1.InitWeightsIdentity(Q16One)
	model.fc2.InitWeightsIdentity(Q16One)
	config := DefaultTrainingConfig()
	opt := NewDynamicBitOptimizer(config.OptimizerConfig, model)
	cpuCtx := NewComputeContext(nil)

	flatX := make([]int32, batchSize*128)
	flatY := make([]int32, batchSize*64)
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

func BenchmarkTrainingStep_LargeModel_GPUPersistent(b *testing.B) {
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

	batchSize := 64
	model := NewBenchmarkMLPArchitecture(128, 128, 64)
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

	flatX := make([]int32, batchSize*128)
	flatY := make([]int32, batchSize*64)
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
