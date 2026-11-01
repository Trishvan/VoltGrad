# VoltGrad — Adaptive, Vendor-Neutral GPU Computing for Go

A fully self-contained, model-agnostic Vulkan GPU training pipeline for Go with **Mandatory Dynamic Bit Widening (DBW)**.

VoltGrad brings **adaptive, vendor-neutral GPU computing** to Go without proprietary vendor lock-in or heavyweight runtime dependencies. Rather than claiming immediate parity with mature single-vendor ecosystems, VoltGrad prioritizes portability, hardware neutrality across all Vulkan-compliant devices (Intel, AMD, NVIDIA, Apple Silicon via MoltenVK, and mobile platforms), and arbitrary-precision integer representations with zero clipping. Reproducible benchmarks will empirically establish how close execution performance gets to CUDA as kernel optimizations progress.

- The **developer only writes their model architecture** (`Forward` and `Backward` passes, layer definitions).
- The pipeline handles **GPU hardware compute discovery, host-coherent zero-copy memory mapping, batching, loss computation, fixed-point optimizers (SGD, Momentum, Adam-Q)**, and **automatic Dynamic Bit Widening (DBW)**.
- The `*Ctx` layer calls ride the **GPU** when the pipeline is open, with a CPU fallback that yields **bit-for-bit identical** integer results.

---

## 📁 Architecture Overview

All code lives in this standalone folder `voltgrad/` with **zero external dependencies** (pure Go standard library + Vulkan C-ABI):

| File | Purpose |
| :--- | :--- |
| [`tensor.go`](voltgrad/tensor.go) | `DynamicBitTensor` with mandatory Dynamic Bit Widening ($W \in [2, 64]$ bits), bit-packing, zero quantization clipping. |
| [`model.go`](voltgrad/model.go) | `ModelArchitecture` interface (`Forward`, `Backward`, `Parameters`, `Gradients`) & `ComputeLoss` (MSE, Cross-Entropy, Brier). |
| [`hardware.go`](voltgrad/hardware.go) | Direct Vulkan GPU discovery, logical device initialization, and host-coherent zero-copy memory mapping. |
| [`kernel.go`](voltgrad/kernel.go) | Generic SPIR-V compute kernel manager with arbitrary storage buffer descriptor sets and push constants. |
| [`gpu_engine.go`](voltgrad/gpu_engine.go) | Coordinates GPU MatMul Forward, MatMul Backward ($dX, dW$), GPU Optimizer, and GPU DBW repacking. |
| [`pipeline.go`](voltgrad/pipeline.go) | Vulkan compute shader compilation, pipeline layouts, push constants, and dispatch execution. |
| [`optimizer.go`](voltgrad/optimizer.go) | `DynamicBitOptimizer` implementing $w(t+1) = w(t) - \lfloor \eta \cdot g / 2^{Q_f} \rfloor$ on GPU/CPU with DBW auto-widening. |
| [`layers.go`](voltgrad/layers.go) | Reusable building blocks (`LinearLayer`, MatMul, ReLU) with GPU and CPU forward/backward differentiation. |
| [`ops.go`](voltgrad/ops.go) | Fixed-point integer operations (MatMulVec64, Softmax, ReLU, activation backward passes). |
| [`trainer.go`](voltgrad/trainer.go) | `VulkanTrainingPipeline` coordinating the training loop, epochs, batching, GPU dispatch, and DBW logging. |
| [`pipeline_test.go`](voltgrad/pipeline_test.go) | Complete working example showing custom 2-layer MLP training with live DBW bit-depth expansion. |
| [`gpu_features_test.go`](voltgrad/gpu_features_test.go) | Unit tests explicitly verifying GPU matrix multiplication and GPU optimizer with DBW bit-for-bit. |
| [`correctness_test.go`](voltgrad/correctness_test.go) | Test suite covering `shaderInt64` feature detection, true 64-bit DBW values (`1<<35`, `1<<45`, `math.MaxInt64`), boundaries, and randomized GPU/CPU parity. |
| [`shaders/`](voltgrad/shaders/) | GLSL compute shader sources (`.comp`), compiled SPIR-V binaries (`.spv`), compilation script, and documentation. |

---

## 🚀 Quickstart: Developer Writing a Custom Model

Developers only define their struct implementing [`ModelArchitecture`](voltgrad/model.go):

```go
package main

import (
    "github.com/Trishvan/VoltGrad/voltgrad"
)

type MyCustomModel struct {
    fc1 *voltgrad.LinearLayer
    fc2 *voltgrad.LinearLayer
}

func NewMyCustomModel(inDim, hiddenDim, outDim int) *MyCustomModel {
    return &MyCustomModel{
        // Start weights at 2-bit! DBW will automatically expand bit-depth as gradients flow
        fc1: voltgrad.NewLinearLayer("fc1", inDim, hiddenDim, 2, true),
        fc2: voltgrad.NewLinearLayer("fc2", hiddenDim, outDim, 2, true),
    }
}

func (m *MyCustomModel) Name() string { return "MyCustomModel" }

func (m *MyCustomModel) Parameters() []*voltgrad.DynamicBitTensor {
    return []*voltgrad.DynamicBitTensor{m.fc1.Weight, m.fc1.Bias, m.fc2.Weight, m.fc2.Bias}
}

func (m *MyCustomModel) Gradients() []*voltgrad.DynamicBitTensor {
    return []*voltgrad.DynamicBitTensor{m.fc1.GradW, m.fc1.GradBias, m.fc2.GradW, m.fc2.GradBias}
}

func (m *MyCustomModel) ZeroGrad() {
    m.fc1.ZeroGrad()
    m.fc2.ZeroGrad()
}

func (m *MyCustomModel) Forward(ctx *voltgrad.ComputeContext, x []int32) ([]int32, error) {
    h1, _ := m.fc1.ForwardCtx(ctx, x, voltgrad.Q16FractionalBits)
    h1Act := voltgrad.ReLU(h1)

    ctx.SetScratch("x", x)
    ctx.SetScratch("h1", h1)
    ctx.SetScratch("h1Act", h1Act)

    return m.fc2.ForwardCtx(ctx, h1Act, voltgrad.Q16FractionalBits)
}

func (m *MyCustomModel) Backward(ctx *voltgrad.ComputeContext, dLoss []int32) error {
    x, _ := ctx.GetScratch("x")
    h1, _ := ctx.GetScratch("h1")
    h1Act, _ := ctx.GetScratch("h1Act")

    dH1Act, _ := m.fc2.BackwardCtx(ctx, dLoss, h1Act.([]int32), voltgrad.Q16FractionalBits)
    dH1 := voltgrad.ReLUBackward(dH1Act, h1.([]int32))
    m.fc1.BackwardCtx(ctx, dH1, x.([]int32), voltgrad.Q16FractionalBits)
    return nil
}
```

When the training pipeline runs, every `ComputeContext` it hands your model is wired to a live `GpuEngine` — the `*Ctx` calls dispatch on it directly. Plain `Forward`/`Backward` calls stay CPU, and both paths produce identical packed integers.

### Launch Training on Vulkan GPU

```go
func main() {
    model := NewMyCustomModel(4, 8, 2)

    config := voltgrad.DefaultTrainingConfig()
    config.Epochs = 10
    config.BatchSize = 16

    pipeline, err := voltgrad.NewTrainingPipeline(config, model)
    if err != nil {
        panic(err)
    }
    defer pipeline.Close()

    // inputs and targets in fixed-point Q16 (65536 = 1.0)
    metrics, err := pipeline.Train(model, inputs, targets)
}
```

---

## 🔬 Live Verification Test Output

```
=== RUN   TestVulkanTrainingPipelineWithDBW
    pipeline_test.go:137: Initial Parameter Bit-Widths: [18 2 18 2]
========================================================================
   VULKAN TRAINING PIPELINE — Adaptive, Vendor-Neutral GPU Computing for Go
========================================================================
 Model Architecture: Example2LayerMLP_4x8x2
 Samples: 64, Batch Size: 16, Epochs: 5
 Mandatory Dynamic Bit Widening (DBW): ACTIVE
 Hardware Backend: Vulkan Host-Coherent Compute Engine
========================================================================

Epoch  1/ 5 | Loss:  21446 | Time:   0.50ms | Bit-Widths: [18 16 18 16]  <-- [DBW Event: 6 expansions]
Epoch  2/ 5 | Loss:  12046 | Time:   1.11ms | Bit-Widths: [18 16 18 16]
Epoch  3/ 5 | Loss:   7427 | Time:   1.04ms | Bit-Widths: [18 16 18 17]  <-- [DBW Event: 1 expansions]
Epoch  4/ 5 | Loss:   4377 | Time:   0.51ms | Bit-Widths: [18 16 18 17]
Epoch  5/ 5 | Loss:   2800 | Time:   0.62ms | Bit-Widths: [18 16 18 17]
    pipeline_test.go:151: Final Parameter Bit-Widths after DBW: [18 16 18 17]
    pipeline_test.go:157: Tensor 1 auto-widened from 2 bits -> 16 bits
    pipeline_test.go:157: Tensor 3 auto-widened from 2 bits -> 17 bits
    pipeline_test.go:168: Loss Convergence: Epoch 1 = 21446 -> Epoch 5 = 2800

>>> TEST PASSED: Vulkan portable training pipeline successfully trained custom model architecture with Dynamic Bit Widening! <<<
--- PASS: TestVulkanTrainingPipelineWithDBW (0.15s)
PASS
```

### GPU Bit-for-Bit Checks

The engine layer has its own unit tests pinning GPU parity against CPU on the packed integers:

```
=== RUN   TestGPUMatMulLinearLayer
    gpu_features_test.go:56: Forward Pass: CPU = [178 -119 75], GPU = [178 -119 75]
    gpu_features_test.go:83: Backward dX: CPU = [2 -19 16 -8], GPU = [2 -19 16 -8]
    gpu_features_test.go:96: >>> GPU Matrix Multiplication for LinearLayer VERIFIED 100% BIT-FOR-BIT! <<<
--- PASS: TestGPUMatMulLinearLayer (0.12s)
=== RUN   TestGPUOptimizerWithDBW
    gpu_features_test.go:158: Initial bit width: 2, New bit width after GPU DBW: 13
    gpu_features_test.go:166: Weights: CPU = [329 -656 1311 -2622], GPU = [329 -656 1311 -2622]
    gpu_features_test.go:178: >>> GPU Optimizer with Dynamic Bit Widening (DBW) VERIFIED 100% BIT-FOR-BIT! <<<
--- PASS: TestGPUOptimizerWithDBW (0.09s)
PASS
```

---

## 🗺️ Roadmap

### VoltGrad 0.2 — GPU execution and numerical correctness (Priority 1)
- [x] Enable and validate shaderInt64 (queried via `vkGetPhysicalDeviceFeatures` and enabled in `VkDeviceCreateInfo`).
- [x] Fix GPU/CPU optimizer discrepancies (64-bit precision MAC, weight decay, Adam-Q signed directional steps).
- [x] Add 64-bit DBW correctness tests (validating values through 64 bits and `math.MaxInt64`).
- [x] Add overflow and boundary tests (validating range limits and byte-boundary transitions).
- [x] Fix GPU resource error handling (reusable cleanup rollback, `vkDeviceWaitIdle` on shutdown).
- [x] Update README examples.

### VoltGrad 0.3 — GPU performance (Priority 2)
- [ ] Batched GEMM.
- [ ] Persistent GPU allocations.
- [ ] Buffer pooling.
- [ ] Asynchronous command execution.
- [ ] GPU activation and reduction kernels.
- [ ] GPU profiling and timestamps.

### VoltGrad 0.4 — Adaptive computing
- [ ] Block-wise DBW.
- [ ] Mixed-precision execution.
- [ ] Adaptive precision policies.
- [ ] Fused neural-network operations.
- [ ] Execution graph.

### VoltGrad 1.0 — Production runtime
- [ ] Cross-platform Vulkan loader.
- [ ] CPU/Vulkan backend selection.
- [ ] Device selection and capability reporting.
- [ ] Model serialization.
- [ ] C ABI and language bindings.
- [ ] Benchmark and compatibility suite.