# VoltGrad — Adaptive, Vendor-Neutral GPU Computing for Go

A fully self-contained, model-agnostic Vulkan GPU training pipeline for Go with **Mandatory Dynamic Bit Widening (DBW)**.

VoltGrad brings **adaptive, vendor-neutral GPU computing** to Go without proprietary vendor lock-in or heavyweight runtime dependencies. Rather than claiming immediate parity with mature single-vendor ecosystems, VoltGrad prioritizes portability, hardware neutrality across all Vulkan-compliant devices (Intel, AMD, NVIDIA, Apple Silicon via MoltenVK, and mobile platforms), and arbitrary-precision integer representations with zero clipping. Reproducible benchmarks will empirically establish how close execution performance gets to CUDA as kernel optimizations progress.

- The **developer only writes their model architecture** (`Forward` and `Backward` passes, layer definitions).
- The pipeline handles **GPU hardware compute discovery, host-coherent zero-copy memory mapping, batching, loss computation, fixed-point optimizers (SGD, Momentum, Adam-Q)**, and **automatic Dynamic Bit Widening (DBW)**.
- The `*Ctx` layer calls ride the **GPU** when the pipeline is open, with a CPU fallback that yields **bit-for-bit identical** integer results.

---

## 📁 Architecture Overview

All code lives in this standalone folder `voltgrad/` with **zero CGO** (pure Go standard library + Vulkan C-ABI dynamic loading):

| File | Purpose |
| :--- | :--- |
| [`tensor.go`](voltgrad/tensor.go) | `DynamicBitTensor` with mandatory Dynamic Bit Widening ($W \in [2, 64]$ bits), bit-packing, zero quantization clipping. |
| [`persistent_tensor.go`](voltgrad/persistent_tensor.go) | `PersistentGpuTensor` maintaining resident weights, bias, and velocity states in GPU memory across iterations. |
| [`model.go`](voltgrad/model.go) | `ModelArchitecture` & `BatchedModelArchitecture` interfaces, `ComputeLoss` (MSE, Cross-Entropy, Brier). |
| [`hardware.go`](voltgrad/hardware.go) | Direct Vulkan GPU discovery, logical device initialization, host-coherent memory mapping, bounds-checked `GpuBuffer`, and `DeviceReport`. |
| [`buffer_pool.go`](voltgrad/buffer_pool.go) | Zero-allocation power-of-two capacity bucketing for Vulkan host-coherent storage buffers. |
| [`kernel.go`](voltgrad/kernel.go) | Generic SPIR-V compute kernel manager with asynchronous dispatch (`DispatchAsync`), `ExecutionToken`, and GPU timestamp profiling (`DispatchProfiled`). |
| [`gpu_engine.go`](voltgrad/gpu_engine.go) | Coordinates GPU Batched MatMul, Backpropagation, GPU Activations (ReLU), GPU Loss (MSE), and DBW Optimizer. |
| [`pipeline.go`](voltgrad/pipeline.go) | Vulkan compute shader compilation, pipeline layouts, push constants, and dispatch execution. |
| [`optimizer.go`](voltgrad/optimizer.go) | `DynamicBitOptimizer` implementing $w(t+1) = w(t) - \lfloor \eta \cdot g / 2^{Q_f} \rfloor$ on GPU/CPU with DBW auto-widening. |
| [`layers.go`](voltgrad/layers.go) | Reusable building blocks (`LinearLayer`, MatMul, Batched GEMM) with persistent GPU binding (`BindGPU`, `SyncToHost`). |
| [`ops.go`](voltgrad/ops.go) | Fixed-point integer operations (MatMulVec64, Softmax, ReLU, activation backward passes). |
| [`trainer.go`](voltgrad/trainer.go) | `VulkanTrainingPipeline` coordinating training epochs, batched GEMM dispatch, and DBW logging. |
| [`vulkan_loader_windows.go`](voltgrad/vulkan_loader_windows.go) / [`_unix.go`](voltgrad/vulkan_loader_unix.go) | Zero-CGO cross-platform dynamic Vulkan loader for Windows (`vulkan-1.dll`), Linux (`libvulkan.so`), and macOS (`libvulkan.dylib`). |
| [`benchmark_test.go`](voltgrad/benchmark_test.go) | Standard Go benchmarks measuring CPU MatMul, GPU MatMul, Batched GEMM, and Persistent Tensor synchronization. |
| [`gpu_performance_test.go`](voltgrad/gpu_performance_test.go) | Performance and parity suite for Batched GEMM, GPU Activations, MSE Loss, Asynchronous Dispatch, and Profiling. |
| [`shaders/`](voltgrad/shaders/) | GLSL compute shader sources (GEMM forward/backward, batched GEMM, ReLU, Loss, DBW repack, Optimizer), compiled SPIR-V bytecode, and build scripts. |
| [`docs/adr/`](docs/adr/) | Architecture Decision Records, including [ADR-0001: Tensor-Wise DBW](docs/adr/0001-tensor-wise-dynamic-bit-widening.md). |

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
=== RUN   TestTrainingPipelinePersistentWeightsParity
    pipeline_test.go:358: >>> Persistent GPU Training Pipeline & CPU Reference: 100% BIT-FOR-BIT IDENTICAL! <<<
--- PASS: TestTrainingPipelinePersistentWeightsParity (0.14s)
PASS
```

---

## 📊 Reproducible Benchmarks (Physical GPU vs. CPU)

Benchmarked on physical hardware (**Intel(R) Iris(R) Xe Graphics**, Vulkan 1.4.323, Driver 0.405.2989, 8074 MB VRAM, Intel Core i5-1135G7 @ 2.40GHz):

| Benchmark Operation | Mode | Iterations | Time / Op | Memory / Op | Allocs / Op | Allocation Delta |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MatMul Forward** ($64 \times 64$) | Pooled Transient GPU | 4,347 | 278.5 µs | 4,744 B | 6 allocs | Baseline |
| **MatMul Forward** ($64 \times 64$) | **Persistent GPU Resident** | **4,684** | **258.1 µs** | **632 B** | **4 allocs** | **-86.7% memory** |
| **Batched GEMM** ($B=32, 64 \times 64$) | Pooled Transient GPU | 3,744 | 290.5 µs | 12,680 B | 6 allocs | Baseline |
| **Batched GEMM** ($B=32, 64 \times 64$) | **Persistent GPU Resident** | **3,861** | **318.8 µs** | **8,568 B** | **4 allocs** | **-32.4% memory** |
| **DBW Optimizer Step** ($N=1024$) | Pooled Transient GPU | 3,229 | 376.5 µs | 41,340 B | 9 allocs | Baseline |
| **DBW Optimizer Step** ($N=1024$) | **Persistent GPU Resident** | **2,724** | **430.1 µs** | **580 B** | **7 allocs** | **-98.6% memory** |
| **Training Step** (Batch 32) | CPU Reference | 3,975 | 337.6 µs | 10,122 B | 268 allocs | Baseline |
| **Training Step** (Batch 32) | **Persistent GPU Pipeline** | **397** | **2.67 ms** | **11,061 B** | **68 allocs** | **-74.6% allocs** |
| **Persistent Sync** (Host $\leftrightarrow$ GPU) | Bidirectional Sync | 14,814 | 78.3 µs | 20,480 B | 6 allocs | Zero copy |

### How to Reproduce Locally:
```powershell
go test -bench="Benchmark" -run="TestNone" github.com/Trishvan/VoltGrad/voltgrad -benchmem
```

---

## 🏛️ Core Design Principles & Architecture Decisions

VoltGrad adheres to strict open-source design principles formalized in our Architecture Decision Records:

- **[ADR-0001: Tensor-Wise Dynamic Bit Widening](docs/adr/0001-tensor-wise-dynamic-bit-widening.md)**: Standardizes on **tensor-wise Dynamic Bit Widening** rather than block-wise DBW.
  - **One dynamic bit-width per tensor**: Guarantees simple, deterministic memory layouts without irregular tiling or inner-loop thread divergence.
  - **Predictable widening & repacking**: Definite rules for when a tensor expands bit-width and repacks in parallel on GPU.
  - **Strict CPU/GPU parity**: Both backends yield bit-for-bit identical integers across all widths ($W \in [2, 64]$).
  - **Correctness and maintainability over theoretical micro-compression**: Optimizes for open-source community maintainability, reproducible debugging, and numerical rock-solid stability.

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
- [x] Batched GEMM (`MatMulBatchedForward`, `MatMulBatchedBackward`, `BatchedModelArchitecture`).
- [x] Persistent GPU allocations (`PersistentGpuTensor` maintaining resident weights & velocities across epochs).
- [x] Buffer pooling (`BufferPool` with power-of-two capacity bucketing for temporary IO buffers).
- [x] Asynchronous command execution (`DispatchAsync` returning `ExecutionToken` with non-blocking submission).
- [x] GPU activation and reduction kernels (`ReLUForward`, `ReLUBackward`, `MSELoss`).
- [x] GPU profiling and timestamps (`DispatchProfiled` with Vulkan GPU timestamp queries).

### VoltGrad 0.4 — Adaptive computing
- [x] Tensor-wise DBW architectural standardization ([ADR-0001](docs/adr/0001-tensor-wise-dynamic-bit-widening.md)).
- [ ] Mixed-precision execution.
- [ ] Adaptive precision policies.
- [ ] Fused neural-network operations.
- [ ] Execution graph.
*(Note: Block-wise DBW is documented in ADR-0001 as deferred to future research exploration).*

### VoltGrad 1.0 — Production runtime
- [x] Cross-platform Vulkan loader (Windows DLL, Linux `.so`, macOS `.dylib` / MoltenVK with zero CGO).
- [ ] CPU/Vulkan backend selection.
- [x] Device selection and capability reporting (`DeviceReport` with vendor, VRAM, and API introspection).
- [ ] Model serialization.
- [ ] C ABI and language bindings.
- [x] Benchmark and compatibility suite.