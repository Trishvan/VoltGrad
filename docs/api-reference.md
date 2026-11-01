# API Reference

Package `voltgrad` — import path `github.com/Trishvan/VoltGrad/voltgrad`.

Conventions used throughout the API:

- All user-visible arithmetic is in **fixed point**: activations, predictions, losses, and hyperparameters are integers scaled by the fractional bit count (`Q16` by default, scale 65536).
- `uint` parameters named `shift` or `fractionalBits` mean the same scale factor; pass `voltgrad.Q16FractionalBits` unless you intentionally choose another fixed-point format.
- Methods prefixed with `Ctx` dispatch on the GPU when the receiving `*ComputeContext` has a GPU engine attached, otherwise they run the CPU implementation. Both paths are bit-for-bit identical.
- Tensors (`*DynamicBitTensor`) widen their bit-width automatically; no allocation or precision policy needs manual management.

Table of contents:

1. [Constants](#1-constants)
2. [Mathematical helpers](#2-mathematical-helpers)
3. [Tensors](#3-tensors)
4. [Layers](#4-layers)
5. [Models and contexts](#5-models-and-contexts)
6. [Losses](#6-losses)
7. [Optimizers](#7-optimizers)
8. [Training](#8-training)
9. [Hardware and buffers](#9-hardware-and-buffers)
10. [Kernels and pipelines](#10-kernels-and-pipelines)
11. [GPU engine](#11-gpu-engine)
12. [Low-level bindings](#12-low-level-bindings)

## 1. Constants

```go
const (
    Q16FractionalBits uint  = 16
    Q16One            int32 = 1 << Q16FractionalBits // 65536
)

const (
    Q10FractionalBits uint  = 10
    Q10One            int32 = 1 << Q10FractionalBits // 1024
)
```

`Q16One` — value representing 1.0 in Q16. `Q10One` — the same for a compact Q10 format.

## 2. Mathematical helpers

```go
func BitsNeeded32(v int32) int
```
Minimum signed bit-width (2..32) required to represent `v` exactly.

```go
func BitsNeeded64(v int64) int
```
Minimum signed bit-width (2..64) required to represent `v` exactly.

```go
func MinMax64(width int) (minVal, maxVal int64)
```
Representable value range `[min, max]` for signed integers of `width` bits (2..64).

```go
func ReLU(x []int32) []int32
func ReLUBackward(dOut []int32, x []int32) []int32
```
Element-wise rectified linear unit and its gradient (`dX[i] = x[i] > 0 ? dOut[i] : 0`). CPU implementations; the GPU equivalents are on `GpuEngine`.

```go
func IntegerSoftmax(logits []int32) []int32
```
Numerically stable fixed-point softmax in Q16.

```go
func MatMulVec64(w *DynamicBitTensor, x []int32, shift uint) ([]int32, error)
func MatMulVecWithBias64(w *DynamicBitTensor, x []int32, bias *DynamicBitTensor, shift uint) ([]int32, error)
func MatMulVecBackward(dOut []int32, x []int32, w *DynamicBitTensor, dW *DynamicBitTensor, shift uint) ([]int32, error)
```
CPU matrix-vector kernels operating directly on packed tensors: forward (`out[i] = (sum_j W[i,j] * x[j]) >> shift`, optionally plus bias), and backward which accumulates `dW[i,j] += (dOut[i] * x[j]) >> shift` into `dW` and returns `dX`. `MatMulVecWithBias64` accumulates intermediates in 64 bits and clamps to int32, mirroring the GPU kernel exactly.

```go
func LearningRateCosineDecay(baseLRQ16 int32, epoch, maxEpochs int) int32
```
Cosine-decayed Q16 learning rate for `epoch` of `maxEpochs`; clamps to `[0.05, 1.05]` of the base rate.

## 3. Tensors

```go
type DynamicBitTensor struct { /* opaque */ }

func NewDynamicBitTensor(shape []int, initialBits int) *DynamicBitTensor
```
Creates a zeroed tensor of the given shape at `initialBits` width (2..64).

Element access, with automatic widening on every write:

```go
func (t *DynamicBitTensor) Get(i int) int64
func (t *DynamicBitTensor) GetInt32(i int) int32
func (t *DynamicBitTensor) Set(i int, v int64)
func (t *DynamicBitTensor) SetInt32(i int, v int32)
func (t *DynamicBitTensor) Add(i int, delta int64) int64 // accumulate, returns new value
```

Introspection:

```go
func (t *DynamicBitTensor) Shape() []int
func (t *DynamicBitTensor) Len() int                 // total scalar elements
func (t *DynamicBitTensor) BitWidth() int            // active bits per element
func (t *DynamicBitTensor) MemoryBytes() int         // packed stream bytes
func (t *DynamicBitTensor) UncompressedInt32Bytes() int
func (t *DynamicBitTensor) CompressionRatio() float64 // MemoryBytes / UncompressedInt32Bytes
```

Width management:

```go
func (t *DynamicBitTensor) Widen(newBitWidth int) // repack to a wider width, no precision loss
func (t *DynamicBitTensor) Clone() *DynamicBitTensor // deep copy
```

Bulk conversion (loads re-fit the width to the minimal fit of the provided values; loads only ever widen, never narrow):

```go
func (t *DynamicBitTensor) ToInt32Slice() []int32
func (t *DynamicBitTensor) FromInt32Slice(values []int32)
func (t *DynamicBitTensor) ToInt64Slice() []int64
func (t *DynamicBitTensor) FromInt64Slice(values []int64)
```
Bulk loaders require exactly `t.Len()` values.

Direct exchange of the packed bitstream with GPU code:

```go
func (t *DynamicBitTensor) Words() []uint32
func (t *DynamicBitTensor) SetWords(words []uint32, bitWidth int)
```
`Words()` returns the packed stream for buffer upload; `SetWords` adopts GPU-computed words at `bitWidth`. This is the exact interface the engine's repack kernel uses.

## 4. Layers

```go
type LayerConfig struct {
    Name            string
    InDim           int
    OutDim          int
    InitialBitWidth int // minimum starting bit width for DBW
    UseBias         bool
}

type LinearLayer struct {
    Name     string
    InDim    int
    OutDim   int
    Weight   *DynamicBitTensor // shape [OutDim, InDim]
    GradW    *DynamicBitTensor
    Bias     *DynamicBitTensor // shape [OutDim]; nil unless UseBias
    GradBias *DynamicBitTensor
    UseBias  bool
}

func NewLinearLayer(name string, inDim, outDim int, initialBits int, useBias bool) *LinearLayer
```

`NewLinearLayer` clamps `initialBits` to a minimum of 2 and allocates all parameter/gradient tensors.

Forward and backward, per sample:

```go
func (l *LinearLayer) Forward(x []int32, shift uint) ([]int32, error) // CPU
func (l *LinearLayer) ForwardCtx(ctx *ComputeContext, x []int32, shift uint) ([]int32, error)
func (l *LinearLayer) Backward(dOut []int32, x []int32, shift uint) ([]int32, error) // CPU
func (l *LinearLayer) BackwardCtx(ctx *ComputeContext, dOut []int32, x []int32, shift uint) ([]int32, error)
```

`Forward` computes `out = (W · x) >> shift + bias`; `Backward` accumulates `dW` (and `dBias`) and returns `dX`. The `Ctx` variants run the same math on the GPU engine when available.

Batched forms (flat row-major matrices, `[batchSize, dim]`):

```go
func (l *LinearLayer) ForwardBatched(x []int32, batchSize int, shift uint) ([]int32, error) // CPU
func (l *LinearLayer) ForwardBatchedCtx(ctx *ComputeContext, x []int32, batchSize int, shift uint) ([]int32, error)
func (l *LinearLayer) BackwardBatched(dOut []int32, x []int32, batchSize int, shift uint) ([]int32, error) // CPU
func (l *LinearLayer) BackwardBatchedCtx(ctx *ComputeContext, dOut []int32, x []int32, batchSize int, shift uint) ([]int32, error)
```

Maintenance:

```go
func (l *LinearLayer) ZeroGrad()
func (l *LinearLayer) InitWeightsIdentity(scale int32) // diagonal = scale, elsewhere 0 (Q16 units)
```

## 5. Models and contexts

```go
type ModelArchitecture interface {
    Name() string
    Layers() []LayerConfig
    Parameters() []*DynamicBitTensor
    Gradients() []*DynamicBitTensor
    ZeroGrad()
    Forward(ctx *ComputeContext, inputs []int32) ([]int32, error)
    Backward(ctx *ComputeContext, dLoss []int32) error
}

type BatchedModelArchitecture interface {
    ModelArchitecture
    ForwardBatched(ctx *ComputeContext, inputs []int32, batchSize int) ([]int32, error)
    BackwardBatched(ctx *ComputeContext, dLoss []int32, batchSize int) error
}
```

`ModelArchitecture` is the requirement for the training pipeline. `BatchedModelArchitecture` additionally enables whole-batch GEMM execution; flat layouts are row-major `b*dim + c`.

```go
type ComputeContext struct { /* opaque */ }

func NewComputeContext(p *Pipeline) *ComputeContext // standalone use; the pipeline builds its own

func (c *ComputeContext) IsGPU() bool
func (c *ComputeContext) GpuEngine() *GpuEngine
func (c *ComputeContext) SetGpuEngine(gpu *GpuEngine)
func (c *ComputeContext) Pipeline() *Pipeline

func (c *ComputeContext) SetScratch(key string, val any)
func (c *ComputeContext) GetScratch(key string) (any, bool)
```

`ComputeContext` carries GPU dispatch capability into your forward/backward methods plus an activation scratch map. `SetGpuEngine(nil)` clears GPU dispatch; `NewComputeContext(...)` without an engine is CPU-only.

## 6. Losses

```go
type LossType int

const (
    LossMSE LossType = iota
    LossCrossEntropy
    LossBrierScore
)

func ComputeLoss(lossType LossType, preds, targets []int32, fractionalBits uint) (lossVal int32, gradLoss []int32, err error)
```

Semantics: `LossMSE` — mean squared error (per-element gradient `pred - target`); `LossCrossEntropy` — integer softmax cross-entropy approximation (gradient `pred - target`, loss `-log(p_target)` with the probability clamped above zero); `LossBrierScore` — summed squared error with the same gradient. Length mismatch returns an error.

## 7. Optimizers

```go
type OptimizerType int

const (
    OptimizerSGD OptimizerType = iota
    OptimizerSGDMomentum
    OptimizerAdamQ
)

type OptimizerConfig struct {
    Type            OptimizerType
    LearningRateQ16 int32
    MomentumBeta    int32
    WeightDecayQ16  int32
    FractionalBits  uint
}

func DefaultOptimizerConfig() OptimizerConfig

type DynamicBitOptimizer struct { /* opaque */ }

func NewDynamicBitOptimizer(config OptimizerConfig, model ModelArchitecture) *DynamicBitOptimizer

func (opt *DynamicBitOptimizer) Step(model ModelArchitecture) ([]string, error)
func (opt *DynamicBitOptimizer) StepCtx(ctx *ComputeContext, model ModelArchitecture) ([]string, error)
func (opt *DynamicBitOptimizer) StepGPU(gpu *GpuEngine, model ModelArchitecture) ([]string, error)
```

Update rules (64-bit integer evaluation on both backends): SGD `w -= (lr*g) >> Qf`; SGD-Momentum maintains velocity `v' = ((β·v + (1−β)·g) >> Qf)` then `w -= (lr·v') >> Qf`; AdamQ uses the same velocity but a signed directional step `w -= (lr·sign(v')) >> 1`. Optional weight decay shrinks `w` before the update. Velocity tensors start at 2 bits and widen as needed.

Every call returns widening notices (for example `Param[0]: DBW extended 15 bits -> 16 bits on GPU (Zero Clipping)`), one per widened parameter tensor. Parameter/gradient count or shape mismatches return errors.

## 8. Training

```go
type TrainingConfig struct {
    Epochs          int
    BatchSize       int
    OptimizerConfig OptimizerConfig
    LossType        LossType
    Verbose         bool
}

func DefaultTrainingConfig() TrainingConfig
func (c *TrainingConfig) Validate() error

type EpochMetrics struct {
    Epoch           int
    MeanLoss        int32
    Duration        time.Duration
    BitWideningLogs []string
    ParamBitWidths  []int
}

type VulkanTrainingPipeline struct { /* opaque */ }

func NewTrainingPipeline(config TrainingConfig, model ModelArchitecture) (*VulkanTrainingPipeline, error)
func (p *VulkanTrainingPipeline) Train(model ModelArchitecture, inputs [][]int32, targets [][]int32) ([]EpochMetrics, error)
func (p *VulkanTrainingPipeline) Close()
```

`NewTrainingPipeline` opens the Vulkan hardware context, the compute pipeline, and the GPU engine; any failure returns an error with no retained resources. `Train` validates the configuration, requires `len(inputs) == len(targets) > 0`, runs the training loop (batched path when the model implements `BatchedModelArchitecture`), and returns per-epoch metrics. `Close` releases everything in reverse order and is safe to call once.

## 9. Hardware and buffers

```go
type HardwareContext struct { /* opaque */ }

func NewHardwareContext() (*HardwareContext, error)

func (c *HardwareContext) HasShaderInt64() bool // true after successful init
func (c *HardwareContext) DeviceReport() DeviceReport
func (c *HardwareContext) CreateHostCoherentBuffer(size uint64) (*GpuBuffer, error)
func (c *HardwareContext) GetPooledBuffer(size uint64) (*GpuBuffer, error)
func (c *HardwareContext) BufferPool() *BufferPool
func (c *HardwareContext) Close()
```

Device discovery enumerates all physical devices, requires a compute queue and the `shaderInt64` feature, and selects the best by type score (discrete > integrated > virtual > CPU). `CreateHostCoherentBuffer` maps host-visible + host-coherent memory (zero-copy CPU/GPU sharing); failures roll back all intermediate Vulkan resources.

```go
type DeviceReport struct {
    DeviceName       string
    DeviceType       string
    VendorID         uint32
    VendorName       string
    DeviceID         uint32
    APIVersion       string
    DriverVersion    string
    HasShaderInt64   bool
    MemoryHeaps      []MemoryHeapReport
    TotalDeviceLocal uint64
    TotalHostVisible uint64
}

func (r DeviceReport) String() string

type MemoryHeapReport struct {
    Index       int
    SizeBytes   uint64
    DeviceLocal bool
}
```

```go
type GpuBuffer struct { /* opaque */ }

func (b *GpuBuffer) Size() uint64     // active size in bytes
func (b *GpuBuffer) Capacity() uint64 // allocated capacity in bytes
func (b *GpuBuffer) Clear()
func (b *GpuBuffer) WriteInt32(data []int32)
func (b *GpuBuffer) WriteUint32(data []uint32)
func (b *GpuBuffer) WriteInt64(data []int64)
func (b *GpuBuffer) ReadInt32(count int) []int32
func (b *GpuBuffer) ReadUint32(count int) []uint32
func (b *GpuBuffer) ReadInt64(count int) []int64
func (b *GpuBuffer) Free()
func (b *GpuBuffer) Release()
```

Write methods panic if the slice exceeds the buffer capacity (a programming-contract check). Read methods always return copies in Go-owned memory, never references into mapped memory. `Free` releases unconditionally; `Release` returns a pooled buffer to its pool instead.

```go
type BufferPool struct { /* opaque */ }

func NewBufferPool(hw *HardwareContext) *BufferPool
func (p *BufferPool) Get(size uint64) (*GpuBuffer, error)
func (p *BufferPool) Put(buf *GpuBuffer)
func (p *BufferPool) Clear()
```

Buckets pool buffers by power-of-two capacity (minimum 256 bytes). `Get` returns any pooled buffer with `Capacity() >= size`; `Put` returns a buffer for reuse; `Clear` releases the whole pool. All pool operations are mutex-guarded.

## 10. Kernels and pipelines

```go
type Pipeline struct { /* opaque */ }

func NewPipeline(hw *HardwareContext, customShaderSPV []byte) (*Pipeline, error)
func (p *Pipeline) EnsureBuffers(count int) error
func (p *Pipeline) Dispatch(inputPairs []int32, pushParam int32) ([]int32, error)
func (p *Pipeline) Close()
```

`NewPipeline` compiles SPIR-V into a compute pipeline with a fixed layout: two storage buffers (binding 0 input, binding 1 output) and a 4-byte push constant. `nil` bytecode selects the embedded default kernel (`train.spv`). `Dispatch` uploads `[x0, y0, x1, y1, ...]` pairs, dispatches in workgroups of 64 with padding, waits on a fence (2-second timeout), and returns the first `count` outputs.

```go
type ComputeKernel struct { /* opaque */ }

func NewComputeKernel(hw *HardwareContext, spv []byte, numBindings int, pushSize uint32) (*ComputeKernel, error)
func (k *ComputeKernel) Dispatch(buffers []*GpuBuffer, pushData unsafe.Pointer, pushSize uint32, workgroups uint32) error
func (k *ComputeKernel) Close()
```

`ComputeKernel` is the general form: any number of storage-buffer bindings and a `pushSize`-byte push constant block, dispatched over an explicit workgroup count. `pushData` must exactly match the push-constant layout declared in the shader. Vulkan call results are checked at every dispatch step; the 2-second fence wait applies here as well.

## 11. GPU engine

```go
type GpuEngine struct { /* opaque */ }

func NewGpuEngine(hw *HardwareContext) (*GpuEngine, error) // compiles all bundled NN kernels
func (e *GpuEngine) Close()
func (e *GpuEngine) DeviceReport() DeviceReport
```

Per-sample matmul:

```go
func (e *GpuEngine) MatMulForward(w *DynamicBitTensor, x []int32, bias *DynamicBitTensor, shift uint) ([]int32, error)
func (e *GpuEngine) MatMulBackward(dOut []int32, x []int32, w *DynamicBitTensor, gradW *DynamicBitTensor, gradBias *DynamicBitTensor, shift uint) ([]int32, error)
```

Batched matmul:

```go
func (e *GpuEngine) MatMulBatchedForward(w *DynamicBitTensor, x []int32, batchSize int, bias *DynamicBitTensor, shift uint) ([]int32, error)
func (e *GpuEngine) MatMulBatchedBackward(dOut []int32, x []int32, batchSize int, w *DynamicBitTensor, gradW *DynamicBitTensor, gradBias *DynamicBitTensor, shift uint) ([]int32, error)
```

Activations and loss:

```go
func (e *GpuEngine) ReLUForward(x []int32) ([]int32, error)
func (e *GpuEngine) ReLUBackward(dOut []int32, x []int32) ([]int32, error)
func (e *GpuEngine) MSELoss(preds, targets []int32, shift uint) (int32, []int32, error)
```

Optimizer (single parameter tensor per call):

```go
func (e *GpuEngine) OptimizerStep(p *DynamicBitTensor, g *DynamicBitTensor, v *DynamicBitTensor, config OptimizerConfig) (expandedBitWidth int, err error)
```

Forward calls return fresh result slices. Backward calls accumulate into the gradient tensors supplied (`gradW`, `gradBias` — pass `nil` where the layer has no bias) and return the input gradient. `OptimizerStep` returns the parameter tensor's bit-width after the update; comparing it against the pre-call width detects widening. All engine methods are CPU/GPU-bit-equal by design and by test.

## 12. Low-level bindings

The package also exports the raw Vulkan structures used by the device chain (`VkInstanceCreateInfo`, `VkDeviceCreateInfo`, `VkBufferCreateInfo`, `VkShaderModuleCreateInfo`, `VkComputePipelineCreateInfo`, `VkDescriptorSetLayoutCreateInfo`, `VkWriteDescriptorSet`, `VkPhysicalDeviceFeatures`, and related enums and types). They exist to model the Vulkan C ABI with `unsafe` layout fidelity and are not intended for direct use by application code; treat them as internal implementation detail subject to change.

Note: the engine communicates weights, biases, gradients, and velocities with the device as 8-byte `int64` words and activations as 4-byte `int32` words; kernels compute in 64-bit and clamp results to the int32 range per each kernel's contract ([Shader Reference](../voltgrad/shaders/README.md)).
