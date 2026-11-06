// Package voltgrad provides fixed-point neural-network training in Go on Vulkan GPUs
// with Q16 fixed-point arithmetic and Dynamic Bit Widening (DBW).
//
// The core runtime engine components reside in subpackage "github.com/Trishvan/VoltGrad/voltgrad/core",
// and test suites reside in "github.com/Trishvan/VoltGrad/voltgrad/test".
package voltgrad

import (
	"github.com/Trishvan/VoltGrad/voltgrad/core"
)

// Re-exported Core Types
type (
	DynamicBitTensor         = core.DynamicBitTensor
	PersistentGpuTensor      = core.PersistentGpuTensor
	HardwareContext          = core.HardwareContext
	HardwareOptions          = core.HardwareOptions
	GpuBuffer                = core.GpuBuffer
	BufferPool               = core.BufferPool
	ComputeKernel            = core.ComputeKernel
	ExecutionToken           = core.ExecutionToken
	GpuEngine                = core.GpuEngine
	LinearLayer              = core.LinearLayer
	ModelArchitecture        = core.ModelArchitecture
	BatchedModelArchitecture = core.BatchedModelArchitecture
	GpuResidentModel         = core.GpuResidentModel
	ComputeContext           = core.ComputeContext
	LossType                 = core.LossType
	LayerConfig              = core.LayerConfig
	OptimizerType            = core.OptimizerType
	OptimizerConfig          = core.OptimizerConfig
	DynamicBitOptimizer      = core.DynamicBitOptimizer
	Pipeline                 = core.Pipeline
	TrainingConfig           = core.TrainingConfig
	EpochMetrics             = core.EpochMetrics
	VulkanTrainingPipeline   = core.VulkanTrainingPipeline
	DeviceReport             = core.DeviceReport
	MemoryHeapReport         = core.MemoryHeapReport
)

// Re-exported Constants
const (
	Q16FractionalBits = core.Q16FractionalBits
	Q16One            = core.Q16One
	Q10FractionalBits = core.Q10FractionalBits
	Q10One            = core.Q10One

	LossMSE          = core.LossMSE
	LossCrossEntropy = core.LossCrossEntropy
	LossBrierScore   = core.LossBrierScore

	OptimizerSGD         = core.OptimizerSGD
	OptimizerSGDMomentum = core.OptimizerSGDMomentum
	OptimizerAdamQ       = core.OptimizerAdamQ
)

// Re-exported Core Constructors & Functional APIs
var (
	NewDynamicBitTensor           = core.NewDynamicBitTensor
	NewPersistentGpuTensor        = core.NewPersistentGpuTensor
	NewHardwareContext            = core.NewHardwareContext
	NewHardwareContextWithOptions = core.NewHardwareContextWithOptions
	NewGpuEngine                  = core.NewGpuEngine
	NewComputeKernel              = core.NewComputeKernel
	NewBufferPool                 = core.NewBufferPool
	NewLinearLayer                = core.NewLinearLayer
	NewComputeContext             = core.NewComputeContext
	NewDynamicBitOptimizer        = core.NewDynamicBitOptimizer
	NewTrainingPipeline           = core.NewTrainingPipeline
	DefaultOptimizerConfig        = core.DefaultOptimizerConfig
	DefaultTrainingConfig         = core.DefaultTrainingConfig
	ComputeLoss                   = core.ComputeLoss
	MatMulVec64                   = core.MatMulVec64
	MatMulVecWithBias64           = core.MatMulVecWithBias64
	MatMulVecBackward             = core.MatMulVecBackward
	IntegerSoftmax                = core.IntegerSoftmax
	ReLU                          = core.ReLU
	ReLUBackward                  = core.ReLUBackward
	LearningRateCosineDecay       = core.LearningRateCosineDecay
	NewPipeline                   = core.NewPipeline
	BitsNeeded64                  = core.BitsNeeded64
	BitsNeeded32                  = core.BitsNeeded32
	MinMax64                      = core.MinMax64
	ReluForwardSPV                = core.ReluForwardSPV
)
