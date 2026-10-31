package voltgrad

import (
	"fmt"
	"time"
)

// TrainingConfig defines global hyperparameters for the training session.
type TrainingConfig struct {
	Epochs          int
	BatchSize       int
	OptimizerConfig OptimizerConfig
	LossType        LossType
	Verbose         bool
}

// DefaultTrainingConfig returns standard training defaults.
func DefaultTrainingConfig() TrainingConfig {
	return TrainingConfig{
		Epochs:          10,
		BatchSize:       64,
		OptimizerConfig: DefaultOptimizerConfig(),
		LossType:        LossMSE,
		Verbose:         true,
	}
}

// EpochMetrics captures training and optimization outcomes per epoch.
type EpochMetrics struct {
	Epoch           int
	MeanLoss        int32
	Duration        time.Duration
	BitWideningLogs []string
	ParamBitWidths  []int
}

// VulkanTrainingPipeline provides a model-agnostic GPU training runtime (like PyTorch CUDA).
// The developer only writes their architecture (implementing ModelArchitecture) and provides data.
// Dynamic Bit Widening (DBW) is monitored and applied automatically at every step.
type VulkanTrainingPipeline struct {
	hw        *HardwareContext
	pipeline  *Pipeline
	gpu       *GpuEngine
	optimizer *DynamicBitOptimizer
	config    TrainingConfig
}

// NewTrainingPipeline initializes the GPU runtime and connects the optimizer.
func NewTrainingPipeline(config TrainingConfig, model ModelArchitecture) (*VulkanTrainingPipeline, error) {
	hw, err := NewHardwareContext()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize Vulkan hardware: %w", err)
	}

	pipe, err := NewPipeline(hw, nil)
	if err != nil {
		hw.Close()
		return nil, fmt.Errorf("failed to create Vulkan compute pipeline: %w", err)
	}

	gpu, err := NewGpuEngine(hw)
	if err != nil {
		pipe.Close()
		hw.Close()
		return nil, fmt.Errorf("failed to create Vulkan neural network GPU engine: %w", err)
	}

	opt := NewDynamicBitOptimizer(config.OptimizerConfig, model)

	return &VulkanTrainingPipeline{
		hw:        hw,
		pipeline:  pipe,
		gpu:       gpu,
		optimizer: opt,
		config:    config,
	}, nil
}

// Train executes the complete training loop over the user's custom architecture.
func (p *VulkanTrainingPipeline) Train(model ModelArchitecture, inputs [][]int32, targets [][]int32) ([]EpochMetrics, error) {
	if len(inputs) != len(targets) {
		return nil, fmt.Errorf("inputs count %d != targets count %d", len(inputs), len(targets))
	}
	if len(inputs) == 0 {
		return nil, fmt.Errorf("training dataset is empty")
	}

	n := len(inputs)
	var history []EpochMetrics
	ctx := NewComputeContext(p.pipeline)
	ctx.SetGpuEngine(p.gpu)

	if p.config.Verbose {
		fmt.Printf("========================================================================\n")
		fmt.Printf("   VULKAN TRAINING PIPELINE (CUDA-Equivalent for Go) with DBW\n")
		fmt.Printf("========================================================================\n")
		fmt.Printf(" Model Architecture: %s\n", model.Name())
		fmt.Printf(" Samples: %d, Batch Size: %d, Epochs: %d\n", n, p.config.BatchSize, p.config.Epochs)
		fmt.Printf(" Mandatory Dynamic Bit Widening (DBW): ACTIVE\n")
		fmt.Printf(" Hardware Backend: Vulkan Host-Coherent Compute Engine\n")
		fmt.Printf("========================================================================\n\n")
	}

	for epoch := 1; epoch <= p.config.Epochs; epoch++ {
		start := time.Now()
		var totalLoss int64
		var epochWidenings []string

		// Batch processing
		for bStart := 0; bStart < n; bStart += p.config.BatchSize {
			bEnd := bStart + p.config.BatchSize
			if bEnd > n {
				bEnd = n
			}

			model.ZeroGrad()

			// Iterate samples in batch
			for idx := bStart; idx < bEnd; idx++ {
				x := inputs[idx]
				y := targets[idx]

				// 1. Forward Pass (User Architecture)
				preds, err := model.Forward(ctx, x)
				if err != nil {
					return nil, fmt.Errorf("forward pass failed at epoch %d sample %d: %w", epoch, idx, err)
				}

				// 2. Loss & Gradient
				lossVal, dLoss, err := ComputeLoss(p.config.LossType, preds, y, p.config.OptimizerConfig.FractionalBits)
				if err != nil {
					return nil, fmt.Errorf("loss computation failed: %w", err)
				}
				totalLoss += int64(lossVal)

				// 3. Backward Pass (User Architecture)
				if err := model.Backward(ctx, dLoss); err != nil {
					return nil, fmt.Errorf("backward pass failed: %w", err)
				}
			}

			// 4. Optimizer Step with Dynamic Bit Widening (DBW) on GPU
			widenings, err := p.optimizer.StepCtx(ctx, model)
			if err != nil {
				return nil, fmt.Errorf("optimizer step failed: %w", err)
			}
			if len(widenings) > 0 {
				epochWidenings = append(epochWidenings, widenings...)
			}
		}

		meanLoss := int32(totalLoss / int64(n))
		dur := time.Since(start)

		// Record parameter bit-widths
		bitWidths := make([]int, len(model.Parameters()))
		for i, param := range model.Parameters() {
			bitWidths[i] = param.BitWidth()
		}

		metrics := EpochMetrics{
			Epoch:           epoch,
			MeanLoss:        meanLoss,
			Duration:        dur,
			BitWideningLogs: epochWidenings,
			ParamBitWidths:  bitWidths,
		}
		history = append(history, metrics)

		if p.config.Verbose {
			fmt.Printf("Epoch %2d/%2d | Loss: %6d | Time: %6.2fms | Bit-Widths: %v",
				epoch, p.config.Epochs, meanLoss, float64(dur.Microseconds())/1000.0, bitWidths)
			if len(epochWidenings) > 0 {
				fmt.Printf("  <-- [DBW Event: %d expansions]", len(epochWidenings))
			}
			fmt.Println()
		}
	}

	return history, nil
}

// Close terminates GPU pipelines and device handles.
func (p *VulkanTrainingPipeline) Close() {
	if p.gpu != nil {
		p.gpu.Close()
		p.gpu = nil
	}
	if p.pipeline != nil {
		p.pipeline.Close()
		p.pipeline = nil
	}
	if p.hw != nil {
		p.hw.Close()
		p.hw = nil
	}
}
