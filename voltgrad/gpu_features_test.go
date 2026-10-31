package voltgrad

import (
	"reflect"
	"testing"
)

// TestGPUMatMulLinearLayer explicitly tests GPU-backed matrix multiplication against CPU reference.
func TestGPUMatMulLinearLayer(t *testing.T) {
	hw, err := NewHardwareContext()
	if err != nil {
		t.Fatalf("Failed to init hardware: %v", err)
	}
	defer hw.Close()

	gpu, err := NewGpuEngine(hw)
	if err != nil {
		t.Fatalf("Failed to init GPU engine: %v", err)
	}
	defer gpu.Close()

	// 1. Create a LinearLayer: InDim=4, OutDim=3, 4-bit weights
	layer := NewLinearLayer("test_fc", 4, 3, 4, true)
	layer.Weight.SetInt32(0, 3)
	layer.Weight.SetInt32(1, -2)
	layer.Weight.SetInt32(2, 5)
	layer.Weight.SetInt32(3, -4)
	layer.Weight.SetInt32(4, 1)
	layer.Weight.SetInt32(5, 7)
	layer.Weight.SetInt32(6, -6)
	layer.Weight.SetInt32(7, 2)
	layer.Weight.SetInt32(8, 0)
	layer.Weight.SetInt32(9, -7)
	layer.Weight.SetInt32(10, 4)
	layer.Weight.SetInt32(11, -1)

	layer.Bias.SetInt32(0, 100)
	layer.Bias.SetInt32(1, -50)
	layer.Bias.SetInt32(2, 25)

	x := []int32{1000, -500, 2000, -1500}
	shift := uint(8)

	// CPU Reference Forward
	cpuOut, err := layer.Forward(x, shift)
	if err != nil {
		t.Fatalf("CPU Forward failed: %v", err)
	}

	// GPU MatMul Forward
	gpuOut, err := gpu.MatMulForward(layer.Weight, x, layer.Bias, shift)
	if err != nil {
		t.Fatalf("GPU MatMulForward failed: %v", err)
	}

	t.Logf("Forward Pass: CPU = %v, GPU = %v", cpuOut, gpuOut)
	if !reflect.DeepEqual(cpuOut, gpuOut) {
		t.Fatalf("GPU Forward pass mismatch: CPU=%v, GPU=%v", cpuOut, gpuOut)
	}

	// Backward Pass Verification
	dOut := []int32{300, -150, 450}

	// CPU Reference Backward
	cpuGradW := layer.GradW.Clone()
	cpuGradBias := layer.GradBias.Clone()
	cpuDX, err := MatMulVecBackward(dOut, x, layer.Weight, cpuGradW, shift)
	if err != nil {
		t.Fatalf("CPU MatMulVecBackward failed: %v", err)
	}
	for i := 0; i < layer.OutDim; i++ {
		cpuGradBias.Add(i, int64(dOut[i]))
	}

	// GPU MatMul Backward
	gpuGradW := layer.GradW.Clone()
	gpuGradBias := layer.GradBias.Clone()
	gpuDX, err := gpu.MatMulBackward(dOut, x, layer.Weight, gpuGradW, gpuGradBias, shift)
	if err != nil {
		t.Fatalf("GPU MatMulBackward failed: %v", err)
	}

	t.Logf("Backward dX: CPU = %v, GPU = %v", cpuDX, gpuDX)
	if !reflect.DeepEqual(cpuDX, gpuDX) {
		t.Fatalf("GPU Backward dX mismatch: CPU=%v, GPU=%v", cpuDX, gpuDX)
	}

	if !reflect.DeepEqual(cpuGradW.ToInt32Slice(), gpuGradW.ToInt32Slice()) {
		t.Fatalf("GPU GradW mismatch: CPU=%v, GPU=%v", cpuGradW.ToInt32Slice(), gpuGradW.ToInt32Slice())
	}

	if !reflect.DeepEqual(cpuGradBias.ToInt32Slice(), gpuGradBias.ToInt32Slice()) {
		t.Fatalf("GPU GradBias mismatch: CPU=%v, GPU=%v", cpuGradBias.ToInt32Slice(), gpuGradBias.ToInt32Slice())
	}

	t.Log(">>> GPU Matrix Multiplication for LinearLayer VERIFIED 100% BIT-FOR-BIT! <<<")
}

// TestGPUOptimizerWithDBW explicitly tests the GPU optimizer against the CPU optimizer reference.
func TestGPUOptimizerWithDBW(t *testing.T) {
	hw, err := NewHardwareContext()
	if err != nil {
		t.Fatalf("Failed to init hardware: %v", err)
	}
	defer hw.Close()

	gpu, err := NewGpuEngine(hw)
	if err != nil {
		t.Fatalf("Failed to init GPU engine: %v", err)
	}
	defer gpu.Close()

	// Start with 2-bit weights (range -2..1)
	paramGPU := NewDynamicBitTensor([]int{4}, 2)
	paramGPU.SetInt32(0, 1)
	paramGPU.SetInt32(1, -1)
	paramGPU.SetInt32(2, 0)
	paramGPU.SetInt32(3, -2)

	paramCPU := paramGPU.Clone()

	// Gradient that causes values to widen
	grad := NewDynamicBitTensor([]int{4}, 16)
	grad.SetInt32(0, -65536)
	grad.SetInt32(1, 131072)
	grad.SetInt32(2, -262144)
	grad.SetInt32(3, 524288)

	velocityGPU := NewDynamicBitTensor([]int{4}, 2)
	velocityCPU := NewDynamicBitTensor([]int{4}, 2)

	cfg := OptimizerConfig{
		Type:            OptimizerSGDMomentum,
		LearningRateQ16: 3276, // 0.05
		MomentumBeta:    58982, // 0.9
		FractionalBits:  Q16FractionalBits,
	}

	initialBits := paramGPU.BitWidth()

	// 1. Run CPU reference step
	cpuOpt := &DynamicBitOptimizer{
		config:     cfg,
		velocities: [][]*DynamicBitTensor{{velocityCPU}},
	}
	mockModelCPU := &singleTensorModel{param: paramCPU, grad: grad}
	_, err = cpuOpt.Step(mockModelCPU)
	if err != nil {
		t.Fatalf("CPU optimizer step failed: %v", err)
	}

	// 2. Run GPU optimizer step
	newBits, err := gpu.OptimizerStep(paramGPU, grad, velocityGPU, cfg)
	if err != nil {
		t.Fatalf("GPU OptimizerStep failed: %v", err)
	}

	t.Logf("Initial bit width: %d, New bit width after GPU DBW: %d", initialBits, newBits)
	if newBits <= initialBits {
		t.Fatalf("Expected GPU DBW to widen parameter bit-width, got %d", newBits)
	}

	// Verify GPU output matches CPU output bit-for-bit
	cpuWeights := paramCPU.ToInt32Slice()
	gpuWeights := paramGPU.ToInt32Slice()
	t.Logf("Weights: CPU = %v, GPU = %v", cpuWeights, gpuWeights)
	if !reflect.DeepEqual(cpuWeights, gpuWeights) {
		t.Fatalf("GPU weights mismatch: CPU=%v, GPU=%v", cpuWeights, gpuWeights)
	}

	cpuVel := velocityCPU.ToInt32Slice()
	gpuVel := velocityGPU.ToInt32Slice()
	t.Logf("Velocities: CPU = %v, GPU = %v", cpuVel, gpuVel)
	if !reflect.DeepEqual(cpuVel, gpuVel) {
		t.Fatalf("GPU velocities mismatch: CPU=%v, GPU=%v", cpuVel, gpuVel)
	}

	t.Log(">>> GPU Optimizer with Dynamic Bit Widening (DBW) VERIFIED 100% BIT-FOR-BIT! <<<")
}

type singleTensorModel struct {
	param *DynamicBitTensor
	grad  *DynamicBitTensor
}

func (m *singleTensorModel) Name() string                          { return "singleTensorModel" }
func (m *singleTensorModel) Layers() []LayerConfig                { return nil }
func (m *singleTensorModel) Parameters() []*DynamicBitTensor      { return []*DynamicBitTensor{m.param} }
func (m *singleTensorModel) Gradients() []*DynamicBitTensor       { return []*DynamicBitTensor{m.grad} }
func (m *singleTensorModel) ZeroGrad()                             {}
func (m *singleTensorModel) Forward(ctx *ComputeContext, x []int32) ([]int32, error) { return nil, nil }
func (m *singleTensorModel) Backward(ctx *ComputeContext, dLoss []int32) error       { return nil }
