# Contributing to VoltGrad

Thank you for your interest in contributing to VoltGrad! VoltGrad is an open-source, zero-CGO, adaptive GPU training pipeline for Go with Dynamic Bit Widening (DBW) and direct Vulkan compute execution.

Whether you are optimizing compute shaders, validating Vulkan drivers across diverse GPUs, adding neural network operations, or improving documentation, this guide will help you get started.

---

## 🏛️ Core Design Invariants

All contributions must uphold VoltGrad's five foundational design principles:

1. **Zero CGO & Vendor-Neutrality**:
   VoltGrad links directly against the native Vulkan C-ABI at runtime (`syscall` on Windows, `purego` on Linux and macOS). We do not require a C compiler, native build toolchains, or vendor-locked ML runtimes (no CUDA, ROCm, or OneAPI dependencies).

2. **Tensor-Wise Dynamic Bit Widening ([ADR-0001](docs/adr/0001-tensor-wise-dynamic-bit-widening.md))**:
   VoltGrad strictly enforces **tensor-wise DBW** ($W \in [2, 64]$ bits). Bit-width selection is uniform across each tensor. Widening occurs dynamically and monotonically whenever a value overflows the current bit-width range $[-2^{W-1}, 2^{W-1}-1]$. **Narrowing is explicitly prohibited during training** to preserve training stability.

3. **Bit-for-Bit Mathematical Parity**:
   The GPU compute shaders and CPU reference implementations must produce **100% numerically identical integer results** across all supported bit-widths ($2 \le W \le 64$), rounding modes, and fixed-point scale factors ($Q_f$).

4. **Explicit GPU Memory Lifecycle & Buffer Safety**:
   - `PersistentGpuTensor` preallocates physical headroom (4 buffers at 8 bytes/element each, ~32 bytes/element) to allow DBW widening up to 64-bit without reallocations.
   - Memory allocated via `CreateHostCoherentBuffer` requests host-visible, host-coherent memory (accessible by both GPU compute units and CPU without explicit flush/invalidate).
   - Temporary buffers must use `BufferPool` with power-of-two capacity bucketing.
   - Buffers must never be accessed after `Release()` or `Put()`, and concurrency safety must be maintained under lock.

5. **Strict Benchmark & Test Assertions**:
   Benchmarks must never silently swallow errors (`b.Fatalf` on any failure). Integration and parity tests must run cleanly on both CPU and Vulkan-capable devices (or Lavapipe).

---

## 🛠️ Development Environment Setup

### Prerequisites

- **Go**: Version 1.22 or newer ([golang.org](https://go.dev/dl/)).
- **Vulkan SDK**: Version 1.3 or newer ([lunarg.com/vulkan-sdk](https://vulkan.lunarg.com/)).
  - Ensure `glslc` or `glslangValidator` is available on your system `PATH`.
- **GPU or Software Driver**:
  - Any Vulkan 1.1+ capable physical GPU (NVIDIA, AMD, Intel, Apple Silicon via MoltenVK).
  - For headless servers or Linux containers without a physical GPU: install Lavapipe (`mesa-vulkan-drivers`).

### Verifying the Toolchain

Check your Go compiler and Vulkan shader compiler:

```bash
go version
glslc --version
```

---

## 🔨 Build & Test Workflow

### 1. Compiling Shaders

VoltGrad includes a pure Go shader compilation CLI that compiles GLSL compute shaders (`voltgrad/shaders/*.comp`) into SPIR-V bytecode (`*.spv`):

```bash
go run ./voltgrad/cmd/compile-shaders
```

If you modify any shader in `voltgrad/shaders/`, you **must** run this tool and commit both the `.comp` and updated `.spv` binaries.

### 2. Running Tests

Run the full unit and parity test suite:

```bash
go test -v ./...
```

To test with the Go race detector (on supported platforms):

```bash
go test -race ./voltgrad
```

*Note: On systems without a Vulkan driver, hardware tests will automatically skip cleanly (`Vulkan not available, skipping`).*

### 3. Running Benchmarks

Verify performance and memory allocations across CPU and persistent GPU paths:

```bash
go test -bench="Benchmark" -run="TestNone" github.com/Trishvan/VoltGrad/voltgrad -benchmem
```

Ensure all benchmarks complete with `PASS` and zero runtime panics.

---

## 🎯 Open Contributor Priority Areas

We actively invite community contributions in the following areas:

### 1. Vulkan Hardware Validation & Driver Compatibility
- **Cross-Vendor Testing**: Test VoltGrad on diverse hardware:
  - NVIDIA (GeForce RTX 20/30/40/50 series, Quadro, Tesla/A100/H100).
  - AMD (Radeon RX 6000/7000 series, Instinct MI200/MI300 via RADV / AMDVLK).
  - Intel (Iris Xe integrated, Arc A-series discrete).
  - Apple Silicon (M1/M2/M3/M4 using MoltenVK on macOS).
  - Cloud / Headless (AWS, GCP, Azure instances with Lavapipe).
- Report any Vulkan feature flags or device limits (`maxComputeWorkGroupSize`, `shaderInt64`, timestamp periods) that require fallbacks.

### 2. Compute Shader Optimization
- **Shared Memory Tiling**: Implement local workgroup shared memory (`shared int32/int64 tile[...]`) in `matmul_forward.comp`, `matmul_batched_forward.comp`, and backpropagation shaders to improve cache locality and memory bandwidth saturation on discrete GPUs.
- **Subgroup Operations**: Utilize Vulkan subgroup extensions (`GL_KHR_shader_subgroup_basic`, `GL_KHR_shader_subgroup_arithmetic`) for fast intra-warp reduction during MSE loss calculation and vector dot products.
- **Workgroup Size Tuning**: Benchmark workgroup dimensions (e.g. $8 \times 8$, $16 \times 16$, $32 \times 8$) across different GPU architectures.

### 3. Full Model Execution Graph & Activation Residency
- Currently, inputs and intermediate layer activations pass between Go slices and temporary GPU buffers between forward and backward steps.
- **Goal**: Implement a fused execution graph where intermediate layer outputs remain in GPU-resident memory across all hidden layers, eliminating intermediate host-device synchronization.

### 4. Fused Neural Network Kernels
- Fused MatMul + ReLU forward pass.
- Fused Bias Addition + Activation.
- Fused Optimizer Step + Weight Decay + DBW Repack in a single dispatch pass.

### 5. Numerical Edge Cases & Dynamic Precision Policies
- Fixed-point quantization analysis and precision loss bounds.
- Adaptive heuristic policies for determining when to widen bit-widths based on gradient magnitude trends.

---

## 📋 Pull Request Guidelines

1. **Keep Commits Focused**: Structure your PR into logical, well-described commits.
2. **Preserve Zero CGO**: Do not introduce native CGO bindings or dependencies on external native libraries outside of the dynamic Vulkan C-ABI loader.
3. **Ensure Bit-for-Bit Parity**: If you add or modify a mathematical operation or shader, provide matching CPU and GPU implementations and write a parity test in `gpu_performance_test.go` or `pipeline_test.go`.
4. **Format Your Code**: Ensure all Go code is formatted with `gofmt -s -w .`.
5. **Update Documentation**: If your changes impact memory management, API contracts, or architecture, update the relevant docs in `docs/` or `README.md`.
