package voltgrad

import (
	"fmt"
	"testing"
)

// ExampleMLPArchitecture demonstrates how ANY user defines their custom model architecture
// in just a few lines of Go code without touching GPU drivers or quantization pipelines.
type ExampleMLPArchitecture struct {
	fc1 *LinearLayer
	fc2 *LinearLayer
}

// NewExampleMLPArchitecture creates a 2-layer MLP: InDim (4) -> Hidden (8) -> OutDim (2).
// Parameters intentionally start at 2-bit storage to demonstrate Dynamic Bit Widening during training!
func NewExampleMLPArchitecture() *ExampleMLPArchitecture {
	return &ExampleMLPArchitecture{
		fc1: NewLinearLayer("fc1", 4, 8, 2, true),
		fc2: NewLinearLayer("fc2", 8, 2, 2, true),
	}
}

func (m *ExampleMLPArchitecture) Name() string {
	return "Example2LayerMLP_4x8x2"
}

func (m *ExampleMLPArchitecture) Layers() []LayerConfig {
	return []LayerConfig{
		{Name: "fc1", InDim: 4, OutDim: 8, InitialBitWidth: 2, UseBias: true},
		{Name: "fc2", InDim: 8, OutDim: 2, InitialBitWidth: 2, UseBias: true},
	}
}

func (m *ExampleMLPArchitecture) Parameters() []*DynamicBitTensor {
	return []*DynamicBitTensor{
		m.fc1.Weight,
		m.fc1.Bias,
		m.fc2.Weight,
		m.fc2.Bias,
	}
}

func (m *ExampleMLPArchitecture) Gradients() []*DynamicBitTensor {
	return []*DynamicBitTensor{
		m.fc1.GradW,
		m.fc1.GradBias,
		m.fc2.GradW,
		m.fc2.GradBias,
	}
}

func (m *ExampleMLPArchitecture) ZeroGrad() {
	m.fc1.ZeroGrad()
	m.fc2.ZeroGrad()
}

func (m *ExampleMLPArchitecture) Forward(ctx *ComputeContext, x []int32) ([]int32, error) {
	// 1. FC1 Forward (accelerated on GPU)
	h1, err := m.fc1.ForwardCtx(ctx, x, Q16FractionalBits)
	if err != nil {
		return nil, err
	}
	// 2. ReLU Activation
	h1Act := ReLU(h1)

	// Save for backward
	ctx.SetScratch("input_x", x)
	ctx.SetScratch("h1_pre_act", h1)
	ctx.SetScratch("h1_act", h1Act)

	// 3. FC2 Forward (accelerated on GPU)
	out, err := m.fc2.ForwardCtx(ctx, h1Act, Q16FractionalBits)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (m *ExampleMLPArchitecture) Backward(ctx *ComputeContext, dLoss []int32) error {
	x, _ := ctx.GetScratch("input_x")
	h1Pre, _ := ctx.GetScratch("h1_pre_act")
	h1Act, _ := ctx.GetScratch("h1_act")

	// 1. FC2 Backward (accelerated on GPU)
	dH1Act, err := m.fc2.BackwardCtx(ctx, dLoss, h1Act.([]int32), Q16FractionalBits)
	if err != nil {
		return err
	}

	// 2. ReLU Backward
	dH1 := ReLUBackward(dH1Act, h1Pre.([]int32))

	// 3. FC1 Backward (accelerated on GPU)
	_, err = m.fc1.BackwardCtx(ctx, dH1, x.([]int32), Q16FractionalBits)
	return err
}

func TestVulkanTrainingPipelineWithDBW(t *testing.T) {
	// 1. Initialize user custom architecture (starting with 2-bit weights!)
	model := NewExampleMLPArchitecture()

	// Initialize weights
	model.fc1.InitWeightsIdentity(Q16One)
	model.fc2.InitWeightsIdentity(Q16One)

	// 2. Prepare synthetic batch of dataset vectors
	numSamples := 64
	inputs := make([][]int32, numSamples)
	targets := make([][]int32, numSamples)

	for i := 0; i < numSamples; i++ {
		// Inputs in Q16 (~0.5, ~1.0, ~-0.2, ~0.8)
		inputs[i] = []int32{32768, 65536, -13107, 52428}
		// Targets in Q16
		targets[i] = []int32{65536, 131072}
	}

	// 3. Configure training
	config := DefaultTrainingConfig()
	config.Epochs = 5
	config.BatchSize = 16
	config.OptimizerConfig.LearningRateQ16 = 3276 // ~0.05
	config.Verbose = true

	// 4. Instantiate Vulkan Pipeline
	pipeline, err := NewTrainingPipeline(config, model)
	if err != nil {
		t.Fatalf("Failed to initialize VulkanTrainingPipeline: %v", err)
	}
	defer pipeline.Close()

	initialBitWidths := make([]int, len(model.Parameters()))
	for i, p := range model.Parameters() {
		initialBitWidths[i] = p.BitWidth()
	}
	t.Logf("Initial Parameter Bit-Widths: %v", initialBitWidths)

	// 5. Train!
	metrics, err := pipeline.Train(model, inputs, targets)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	if len(metrics) != config.Epochs {
		t.Errorf("Expected %d epoch metrics, got %d", config.Epochs, len(metrics))
	}

	// 6. Verify Dynamic Bit Widening took effect
	finalBitWidths := metrics[len(metrics)-1].ParamBitWidths
	t.Logf("Final Parameter Bit-Widths after DBW: %v", finalBitWidths)

	widened := false
	for i := range finalBitWidths {
		if finalBitWidths[i] > initialBitWidths[i] {
			widened = true
			t.Logf("Tensor %d auto-widened from %d bits -> %d bits", i, initialBitWidths[i], finalBitWidths[i])
		}
	}

	if !widened {
		t.Errorf("Expected Dynamic Bit Widening to expand bit allocation, but bit-widths remained unchanged: %v", finalBitWidths)
	}

	// 7. Verify Loss Decreased
	firstLoss := metrics[0].MeanLoss
	lastLoss := metrics[len(metrics)-1].MeanLoss
	t.Logf("Loss Convergence: Epoch 1 = %d -> Epoch %d = %d", firstLoss, config.Epochs, lastLoss)
	if lastLoss >= firstLoss {
		t.Errorf("Expected loss to decrease over epochs, but initial=%d, final=%d", firstLoss, lastLoss)
	}

	fmt.Println("\n>>> TEST PASSED: Vulkan portable training pipeline successfully trained custom model architecture with Dynamic Bit Widening! <<<")
}
