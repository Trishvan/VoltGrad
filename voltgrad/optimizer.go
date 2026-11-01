package voltgrad

import (
	"fmt"
	"math"
)

// OptimizerType designates the parameter update strategy.
type OptimizerType int

const (
	OptimizerSGD OptimizerType = iota
	OptimizerSGDMomentum
	OptimizerAdamQ
)

// OptimizerConfig configures parameter optimization.
type OptimizerConfig struct {
	Type            OptimizerType
	LearningRateQ16 int32 // Learning rate in Q16 (e.g. 0.01 * 65536 = 655)
	MomentumBeta    int32 // Momentum factor in Q16 (e.g. 0.9 * 65536 = 58982)
	WeightDecayQ16  int32 // L2 regularizer in Q16 (optional)
	FractionalBits  uint  // Fixed-point scaling (usually 16 for Q16)
}

// DefaultOptimizerConfig returns recommended default hyperparameters.
func DefaultOptimizerConfig() OptimizerConfig {
	return OptimizerConfig{
		Type:            OptimizerSGDMomentum,
		LearningRateQ16: 655,   // ~0.01
		MomentumBeta:    58982, // 0.9
		WeightDecayQ16:  0,
		FractionalBits:  Q16FractionalBits,
	}
}

// DynamicBitOptimizer manages fixed-point integer parameter updates with mandatory Dynamic Bit Widening.
//
// Update Rule:
//
//	w_{i,j}(t+1) = w_{i,j}(t) - \lfloor \eta \cdot g_{i,j}(t) / 2^{Q_f} \rfloor
//
// If the updated weight w(t+1) exceeds the tensor's current bit-width, the DynamicBitTensor
// automatically triggers DBW expansion without precision loss.
type DynamicBitOptimizer struct {
	config    OptimizerConfig
	velocities [][]*DynamicBitTensor // Momentum velocity per parameter tensor
	steps      int64
}

// NewDynamicBitOptimizer constructs an optimizer for the target model architecture.
func NewDynamicBitOptimizer(config OptimizerConfig, model ModelArchitecture) *DynamicBitOptimizer {
	params := model.Parameters()
	velocities := make([][]*DynamicBitTensor, 1)
	velocities[0] = make([]*DynamicBitTensor, len(params))

	for i, p := range params {
		// Initialize velocity buffer with same shape, starting at minimal 2-bit
		velocities[0][i] = NewDynamicBitTensor(p.Shape(), 2)
	}

	return &DynamicBitOptimizer{
		config:     config,
		velocities: velocities,
		steps:      0,
	}
}

// Step performs one optimization step over all model parameters and gradients.
// Returns a summary of any DBW bit widening that occurred during the update.
func (opt *DynamicBitOptimizer) Step(model ModelArchitecture) ([]string, error) {
	params := model.Parameters()
	grads := model.Gradients()

	if len(params) != len(grads) {
		return nil, fmt.Errorf("param count %d != grad count %d", len(params), len(grads))
	}

	opt.steps++
	var wideningNotices []string
	lr := int64(opt.config.LearningRateQ16)
	beta := int64(opt.config.MomentumBeta)
	one := int64(1 << opt.config.FractionalBits)
	shift := opt.config.FractionalBits

	for pIdx := 0; pIdx < len(params); pIdx++ {
		p := params[pIdx]
		g := grads[pIdx]
		v := opt.velocities[0][pIdx]

		if p.Len() != g.Len() {
			return nil, fmt.Errorf("param tensor %d len %d != grad len %d", pIdx, p.Len(), g.Len())
		}

		oldBits := p.BitWidth()

		for i := 0; i < p.Len(); i++ {
			gradVal := g.Get(i)
			currW := p.Get(i)

			var updateStep int64

			switch opt.config.Type {
			case OptimizerSGD:
				// update = (lr * grad) >> shift
				updateStep = (lr * gradVal) >> shift

			case OptimizerSGDMomentum:
				// v(t+1) = (beta * v(t) + (one - beta) * grad) >> shift
				oldV := v.Get(i)
				newV := ((beta * oldV) + ((one - beta) * gradVal)) >> shift
				v.Set(i, newV)
				updateStep = (lr * newV) >> shift

			case OptimizerAdamQ:
				// Pure integer momentum approximation
				oldV := v.Get(i)
				newV := ((beta * oldV) + ((one - beta) * gradVal)) >> shift
				v.Set(i, newV)
				// Signed directional step
				sign := int64(0)
				if newV > 0 {
					sign = 1
				} else if newV < 0 {
					sign = -1
				}
				updateStep = (lr * sign) >> 1
			}

			// Weight decay (L2)
			if opt.config.WeightDecayQ16 > 0 {
				decay := (int64(opt.config.WeightDecayQ16) * currW) >> shift
				currW -= decay
			}

			// Parameter update with auto-widening on overflow
			newW := currW - updateStep
			p.Set(i, newW)
		}

		newBits := p.BitWidth()
		if newBits > oldBits {
			wideningNotices = append(wideningNotices, fmt.Sprintf("Param[%d]: DBW extended %d bits -> %d bits (Zero Clipping)", pIdx, oldBits, newBits))
		}
	}

	return wideningNotices, nil
}

// StepCtx performs an optimization step using GPU acceleration if available in ctx, or falls back to CPU.
func (opt *DynamicBitOptimizer) StepCtx(ctx *ComputeContext, model ModelArchitecture) ([]string, error) {
	if ctx != nil && ctx.IsGPU() {
		return opt.StepGPU(ctx.GpuEngine(), model)
	}
	return opt.Step(model)
}

// StepGPU executes parameter optimization directly on the physical GPU, including DBW expansion tracking.
func (opt *DynamicBitOptimizer) StepGPU(gpu *GpuEngine, model ModelArchitecture) ([]string, error) {
	params := model.Parameters()
	grads := model.Gradients()

	if len(params) != len(grads) {
		return nil, fmt.Errorf("param count %d != grad count %d", len(params), len(grads))
	}

	opt.steps++
	var wideningNotices []string

	for pIdx := 0; pIdx < len(params); pIdx++ {
		p := params[pIdx]
		g := grads[pIdx]
		v := opt.velocities[0][pIdx]

		if pt := p.PersistentGPU(); pt != nil && pt.IsAllocated() {
			oldBits := p.BitWidth()
			newBits, err := gpu.OptimizerStepResident(
				pt.BufferInt64(), pt.BufferGrad(), pt.BufferVel(), pt.BufferPacked(),
				p.Len(), oldBits, opt.config)
			if err != nil {
				return nil, fmt.Errorf("resident GPU optimizer step failed on param %d: %w", pIdx, err)
			}
			p.SetBitWidth(newBits)
			if newBits > oldBits {
				wideningNotices = append(wideningNotices, fmt.Sprintf("Param[%d]: DBW extended %d bits -> %d bits on GPU (Zero Clipping)", pIdx, oldBits, newBits))
			}
			continue
		}

		oldBits := p.BitWidth()
		newBits, err := gpu.OptimizerStep(p, g, v, opt.config)
		if err != nil {
			return nil, fmt.Errorf("GPU optimizer step failed on param %d: %w", pIdx, err)
		}

		if newBits > oldBits {
			wideningNotices = append(wideningNotices, fmt.Sprintf("Param[%d]: DBW extended %d bits -> %d bits on GPU (Zero Clipping)", pIdx, oldBits, newBits))
		}
	}

	return wideningNotices, nil
}

// LearningRateCosineDecay computes updated Q16 learning rate following a cosine schedule.
func LearningRateCosineDecay(baseLRQ16 int32, epoch, maxEpochs int) int32 {
	if epoch >= maxEpochs {
		epoch = maxEpochs - 1
	}
	cosVal := math.Cos(math.Pi * float64(epoch) / float64(maxEpochs))
	factor := 0.5 * (1.0 + cosVal)
	decayed := float64(baseLRQ16) * (0.05 + 0.95*factor)
	return int32(decayed)
}
