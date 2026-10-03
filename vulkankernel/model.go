package vulkankernel

import (
	"fmt"
	"math"
)

// Q16One represents 1.0 in Q16 fixed-point (2^16 = 65,536).
const (
	Q16FractionalBits uint  = 16
	Q16One            int32 = 1 << Q16FractionalBits
)

// Q10One represents 1.0 in Q10 fixed-point (2^10 = 1024).
const (
	Q10FractionalBits uint  = 10
	Q10One            int32 = 1 << Q10FractionalBits
)

// LossType defines the objective function.
type LossType int

const (
	LossMSE LossType = iota
	LossCrossEntropy
	LossBrierScore
)

// LayerConfig defines a customizable architectural layer.
type LayerConfig struct {
	Name            string
	InDim           int
	OutDim          int
	InitialBitWidth int // Minimum starting bit width for DBW (e.g. 2 for int2)
	UseBias         bool
}

// ModelArchitecture is the interface user models must implement.
// A developer only writes their architecture (layer definitions, forward pass logic, backward pass logic)
// and plugs it into the VulkanTrainingPipeline.
type ModelArchitecture interface {
	// Name returns the identifier of the architecture.
	Name() string

	// Layers returns all parameterized layers.
	Layers() []LayerConfig

	// Parameters returns pointers to all trainable DynamicBitTensors.
	// DBW monitoring and Vulkan memory sync operate transparently on these tensors.
	Parameters() []*DynamicBitTensor

	// Gradients returns pointers to all gradient DynamicBitTensors.
	Gradients() []*DynamicBitTensor

	// ZeroGrad resets all gradient buffers to zero.
	ZeroGrad()

	// Forward executes the forward evaluation graph on the input batch.
	// Returns output prediction logits / values.
	Forward(ctx *ComputeContext, inputs []int32) ([]int32, error)

	// Backward executes backpropagation given downstream loss gradients.
	Backward(ctx *ComputeContext, dLoss []int32) error
}

// ComputeContext provides access to GPU dispatch and acceleration helpers during Forward and Backward.
type ComputeContext struct {
	pipeline *Pipeline
	scratch  map[string]interface{}
}

// NewComputeContext creates a execution context.
func NewComputeContext(p *Pipeline) *ComputeContext {
	return &ComputeContext{
		pipeline: p,
		scratch:  make(map[string]interface{}),
	}
}

// SetScratch stores intermediate forward activations needed by backward.
func (c *ComputeContext) SetScratch(key string, val interface{}) {
	c.scratch[key] = val
}

// GetScratch retrieves stored activations.
func (c *ComputeContext) GetScratch(key string) (interface{}, bool) {
	v, ok := c.scratch[key]
	return v, ok
}

// Pipeline returns the underlying Vulkan compute pipeline.
func (c *ComputeContext) Pipeline() *Pipeline {
	return c.pipeline
}

// ComputeLoss calculates loss and loss gradient for predictions given targets.
func ComputeLoss(lossType LossType, preds, targets []int32, fractionalBits uint) (lossVal int32, gradLoss []int32, err error) {
	if len(preds) != len(targets) {
		return 0, nil, fmt.Errorf("predictions len %d != targets len %d", len(preds), len(targets))
	}
	n := len(preds)
	gradLoss = make([]int32, n)
	var totalLoss int64

	scale := int64(1 << fractionalBits)

	switch lossType {
	case LossMSE:
		for i := 0; i < n; i++ {
			diff := int64(preds[i]) - int64(targets[i])
			gradLoss[i] = int32(diff) // dL/dPred = 2*(pred - target) normalized
			totalLoss += (diff * diff) >> fractionalBits
		}
		lossVal = int32(totalLoss / int64(n))

	case LossBrierScore:
		// Brier Score = sum((p_i - y_i)^2)
		for i := 0; i < n; i++ {
			diff := int64(preds[i]) - int64(targets[i])
			gradLoss[i] = int32(diff)
			totalLoss += (diff * diff) >> fractionalBits
		}
		lossVal = int32(totalLoss)

	case LossCrossEntropy:
		// Integer Softmax Cross-Entropy loss approximation
		// grad = p_i - y_i
		for i := 0; i < n; i++ {
			diff := int64(preds[i]) - int64(targets[i])
			gradLoss[i] = int32(diff)
			if targets[i] > 0 {
				// -log(p_target) approximated in fixed point
				probClamped := int64(preds[i])
				if probClamped < 1 {
					probClamped = 1
				}
				negLog := int64(-math.Log(float64(probClamped)/float64(scale)) * float64(scale))
				totalLoss += negLog
			}
		}
		lossVal = int32(totalLoss)
	}

	return lossVal, gradLoss, nil
}
