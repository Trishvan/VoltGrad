# Training Guide

This guide covers the knobs that shape a training run: configuration validation, data layout, loss functions, optimizers, and the monitoring output. Interfaces are documented in the [API Reference](api-reference.md); a first walkthrough is in the [Tutorial](tutorial.md).

## 1. Training configuration

```go
type TrainingConfig struct {
    Epochs          int
    BatchSize       int
    OptimizerConfig OptimizerConfig
    LossType        LossType
    Verbose         bool
}
```

Defaults (`DefaultTrainingConfig`):

| Field | Default |
| :--- | :--- |
| `Epochs` | 10 |
| `BatchSize` | 64 |
| `OptimizerConfig` | `DefaultOptimizerConfig()` — SGD with momentum (see section 4). |
| `LossType` | `LossMSE` |
| `Verbose` | `true` |

`Validate()` enforces:

- `Epochs > 0` and `BatchSize > 0`
- `OptimizerConfig.LearningRateQ16 > 0`
- `1 <= FractionalBits <= 32`
- `MomentumBeta >= 0` and `WeightDecayQ16 >= 0`

Validation runs in both `NewTrainingPipeline` and `Train`; an invalid configuration never starts training.

The pipeline copies its configuration at construction time. Changing a `TrainingConfig` value after `NewTrainingPipeline` has no effect on a running pipeline — create a new pipeline to change epochs, batch size, or optimizer settings.

## 2. Data preparation

`Train(model, inputs, targets)` requires:

- `inputs` and `targets` are `[][]int32` in Q16 fixed point (65536 == 1.0).
- `len(inputs) == len(targets)`, both greater than zero.
- Every sample may have its own length; batched models additionally assume all samples share `len(inputs[0])` as the row width.

Conversion helpers are straightforward:

```go
func toQ16(f float64) int32 { return int32(math.Round(f * 65536.0)) }
func fromQ16(q int32) float64 { return float64(q) / 65536.0 }
```

Keep magnitudes in a sane Q16 range for your use case. The parameter tensors themselves tolerate the full 64-bit range, but activations are `int32` and matmul outputs are clamped to the int32 range on both CPU and GPU.

## 3. Loss functions

`LossType` selects the objective; gradients are returned per prediction element so `Backward` can consume them directly.

| Loss | Value | Gradient (`dL/dPred`) |
| :--- | :--- | :--- |
| `LossMSE` | Mean of `(pred - target)^2` over all elements (shifted back into Q16). | `pred - target` |
| `LossCrossEntropy` | Integer softmax cross-entropy approximation: `-log(p_target)` for each positive target, with `p` clamped at the smallest representable Q16 quantum. | `pred - target` |
| `LossBrierScore` | Sum of `(pred - target)^2` over all elements (not normalized). | `pred - target` |

Implementations live in `ComputeLoss` (`voltgrad/model.go`). On the GPU path for `LossMSE`, the loss and gradient are computed by the `loss_mse` kernel directly in device memory.

Classification workflows typically pair `LossCrossEntropy` with an `IntegerSoftmax` in the model's forward pass; `LossMSE` covers regression and general fixed-point function approximation.

## 4. Optimizers

```go
type OptimizerConfig struct {
    Type            OptimizerType
    LearningRateQ16 int32 // Q16, e.g. 0.01 -> 655
    MomentumBeta    int32 // Q16, e.g. 0.9  -> 58982
    WeightDecayQ16  int32 // Q16, optional
    FractionalBits  uint  // usually Q16FractionalBits
}
```

Available types and their per-element update rules (all evaluated in 64-bit integers):

| Type | Rule |
| :--- | :--- |
| `OptimizerSGD` | `w -= (lr * g) >> Qf` |
| `OptimizerSGDMomentum` | `v' = ((beta * v) + ((1 - beta) * g)) >> Qf`, then `w -= (lr * v') >> Qf` |
| `OptimizerAdamQ` | Same velocity update as momentum, then a signed step: `w -= (lr * sign(v')) >> 1` |

Weight decay (`WeightDecayQ16 > 0`) shrinks the current weight before the update step on every optimizer type, on both CPU and GPU. Velocity tensors are created at 2 bits and widen with the same DBW rules as parameters.

The three execution tiers (`Step` = CPU, `StepCtx` = GPU when available, `StepGPU` = explicit GPU) are functionally identical — the parity tests pin CPU and GPU to bit-for-bit equal outcomes. The `DynamicBitOptimizer` returned by `NewDynamicBitOptimizer` owns the velocity buffers, so do not create a fresh optimizer per batch; construct once and reuse.

`DefaultOptimizerConfig` is SGD with momentum at lr ≈ 0.01, beta = 0.9, no decay, Q16 — sensible starting point for small nets.

## 5. Learning-rate schedules

The exported helper computes cosine-decayed rates:

```go
lr := voltgrad.LearningRateCosineDecay(baseLRQ16, epoch, maxEpochs)
```

The training pipeline applies a single constant learning rate for a whole `Train` run. To schedule, either run multiple pipelines between schedule points, or drive the optimizer yourself (its `Step`/`StepCtx` methods work with any loop); the helper is provided for that purpose.

When rolling your own loop, remember the responsibilities that `VulkanTrainingPipeline.Train` normally takes: batch iteration, `ZeroGrad` per batch, loss and backward per sample or per batch, and the optimizer step with widening-log capture.

## 6. Monitoring training

`Train` returns `[]EpochMetrics`:

```go
type EpochMetrics struct {
    Epoch           int
    MeanLoss        int32
    Duration        time.Duration
    BitWideningLogs []string
    ParamBitWidths  []int
}
```

- `MeanLoss` — the epoch-level mean loss as the pipeline accounts it across batches and samples.
- `Duration` — wall time including all GPU dispatches for that epoch.
- `BitWideningLogs` — every DBW expansion that fired during the epoch, in order.
- `ParamBitWidths` — the bit-width of each parameter tensor at epoch end (same indexing as `model.Parameters()`).

With `Verbose = true` you additionally get a device banner (GPU name, vendor, driver, Vulkan API version, VRAM) before epoch 1 and one progress line per epoch. Set `Verbose = false` for benchmarking or silent runs — the banner value can also be fetched at any time from `pipeline`'s hardware context or from `GpuEngine.DeviceReport()`.

Reading the widening stream is the primary diagnostic: a training run whose tensors never widen typically indicates dead gradients (check `ZeroGrad` and loss scaling), while endless widening typically indicates an oversized learning rate pushing values across width boundaries every step.

## 7. Batch size and the two training paths

The pipeline selects the path per model type:

- **Batched path** (model implements `BatchedModelArchitecture`): each batch is flattened into a `[batchSize * inDim]` matrix, forwarded through batched GEMM, loss computed once over the flattened batch, then batched backward, then the optimizer step. This maximizes GPU occupancy and is the recommended path for anything beyond toy datasets.
- **Per-sample path** (model implements only `ModelArchitecture`): each sample runs forward/loss/backward independently within the batch, then the optimizer steps once for the batch.

Both paths share identical optimizer semantics and DBW behavior; parameter values converge to the same trajectory given the same hyperparameters. Partial final batches are handled (the last batch may be smaller than `BatchSize`).

## 8. Practical recommendations

- Start tensors at 2 bits for maximum compression; if convergence speed matters more, start wider (for example 8 or 16) — DBW never narrows, so starting wide only trades memory for speed.
- Make `BatchSize` a multiple of your hardware's likely compute-lane count (64 lanes is the shader dispatch quantum used by the bundled kernels) to minimize padding waste in batched GEMM.
- Keep `Verbose = false` inside tight benchmark loops; per-epoch printing is not free at toy scales.
- Weight decay is applied in fixed point; values around `65` (≈ 0.001) are a common Q16 starting point.
- For deterministic experiments, keep in mind the training loop is sequential and single-queue: the same inputs, config, and seed produce the same trajectory.
