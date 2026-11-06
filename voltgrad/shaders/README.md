# VoltGrad Vulkan Compute Shaders

This directory contains the GLSL compute shaders and precompiled SPIR-V binaries (`.spv`) powering VoltGrad's dual-path GPU execution engine:

- **64-bit Native Execution Path**: Utilizes `shaderInt64` for full 64-bit integer MACs and Dynamic Bit Widening up to 64 bits ($W \le 64$).
- **32-bit Core Execution Path**: Pure GLSL 4.50 with zero hardware extension dependencies, computing exact 64-bit signed products and accumulations via `imulExtended` and `uaddCarry`, with 50% lower VRAM footprint ($W \le 32$).

See [ADR-0002](../docs/adr/0002-dual-path-execution-and-optional-shaderint64.md) for full architectural details.

## Shaders Overview

### 64-Bit Native Path (Vulkan 1.0 with `shaderInt64`)

1. **`matmul_forward.comp`** (`matmul_forward.spv`)
   - Computes: \( Y = (W \cdot x) \gg \text{shift} + \text{bias} \)
   - Unpacks dynamically-packed weights (2 to 64 bits) directly in GPU registers.
   - Clamps intermediate 64-bit accumulations to 32-bit fixed point range.

2. **`matmul_backward_dx.comp`** (`matmul_backward_dx.spv`)
   - Computes: \( dX = (W^T \cdot dOut) \gg \text{shift} \)
   - Dequantizes DBW weights on-the-fly and accumulates input error gradients.

3. **`matmul_backward_dw.comp`** (`matmul_backward_dw.spv`)
   - Computes: \( dW_{i,j} += (dOut_i \cdot x_j) \gg \text{shift} \)
   - Accumulates full 64-bit weight and bias gradients directly into storage buffers.

4. **`matmul_batched_forward.comp`** (`matmul_batched_forward.spv`)
   - Computes: \( Y[B, \text{outDim}] = (X[B, \text{inDim}] \cdot W^T) \gg \text{shift} + \text{bias} \)
   - Parallelized across batch size \( B \) and output features without thread divergence.

5. **`matmul_batched_backward_dx.comp`** (`matmul_batched_backward_dx.spv`)
   - Computes input batch gradients: \( dX[B, \text{inDim}] = (dOut[B, \text{outDim}] \cdot W) \gg \text{shift} \).

6. **`matmul_batched_backward_dw.comp`** (`matmul_batched_backward_dw.spv`)
   - Accumulates weight and bias gradients across the entire batch in parallel on GPU.

7. **`loss_mse.comp`** (`loss_mse.spv`)
   - Computes squared error loss and error gradients directly in GPU VRAM with 64-bit sum reduction.

8. **`optimizer_step.comp`** (`optimizer_step.spv`)
   - Supports SGD, SGD with Momentum, and AdamQ on 64-bit parameters (`int64_t`).
   - Evaluates dynamic bit-width requirement (`bits_needed_64`) per element and updates target bit-width via `atomicMax(meta[0], needed)`.

9. **`repack_dbw.comp`** (`repack_dbw.spv`)
   - Repacks 64-bit parameter values into contiguous bitstreams in VRAM ($W \le 64$).
   - Handles multi-word bitstream spans across up to three 32-bit storage words using lockless `atomicOr`.

### 32-Bit Core Path (Universal Vulkan 1.0, Pure GLSL 4.50)

10. **`matmul_forward_i32.comp`** (`matmul_forward_i32.spv`)
    - Computes \( Y = (W \cdot x) \gg \text{shift} + \text{bias} \) using 32-bit parameters and registers.
    - Uses `imulExtended` and `uaddCarry` for exact 64-bit intermediate signed MACs without `shaderInt64`.

11. **`matmul_backward_dx_i32.comp`** (`matmul_backward_dx_i32.spv`)
    - Computes input gradient \( dX = (W^T \cdot dOut) \gg \text{shift} \) using 32-bit unpacked weights.

12. **`matmul_backward_dw_i32.comp`** (`matmul_backward_dw_i32.spv`)
    - Accumulates 32-bit weight and bias gradients directly into storage buffers.

13. **`matmul_batched_forward_i32.comp`** (`matmul_batched_forward_i32.spv`)
    - Batched matrix-matrix multiplication on 32-bit GPU path.

14. **`matmul_batched_backward_dx_i32.comp`** (`matmul_batched_backward_dx_i32.spv`)
    - Computes input batch gradients across the batch in 32-bit mode.

15. **`matmul_batched_backward_dw_i32.comp`** (`matmul_batched_backward_dw_i32.spv`)
    - Accumulates weight and bias gradients across batches in 32-bit mode.

16. **`loss_mse_i32.comp`** (`loss_mse_i32.spv`)
    - Computes MSE loss and gradients using 32-bit storage buffers.

17. **`optimizer_step_i32.comp`** (`optimizer_step_i32.spv`)
    - In-place parameter updates, momentum velocity updates, and `bits_needed_32` tracking on 32-bit buffers.

18. **`repack_dbw_i32.comp`** (`repack_dbw_i32.spv`)
    - Direct GPU bitstream repacking for $W \le 32$ bits with 16 bytes/element buffer allocation.

### Common Activation Shaders

19. **`relu_forward.comp`** (`relu_forward.spv`)
    - Element-wise rectified linear unit: \( Y_i = \max(0, X_i) \).

20. **`relu_backward.comp`** (`relu_backward.spv`)
    - Element-wise activation backward: \( dX_i = (X_i > 0 \ ?\ dOut_i : 0) \).

## Compiling Shaders (Pure Go & Native Tools)

Precompiled SPIR-V binaries (`.spv`) are already committed into the repository and embedded directly into Go binaries via `//go:embed`. Recompilation is **only necessary when modifying `.comp` source files**.

### Option A: Using the VoltGrad Native Go Tool (Recommended)
VoltGrad includes a pure Go compilation tool that automatically locates `glslc` or `glslangValidator` from your system PATH or Vulkan SDK:

```bash
go run ./cmd/compileshaders
```

### Option B: Using `glslc` (Google Shaderc / Vulkan SDK)
```bash
glslc -fshader-stage=compute matmul_forward.comp -o matmul_forward.spv
glslc -fshader-stage=compute matmul_batched_forward.comp -o matmul_batched_forward.spv
```

### Option C: Using `glslangValidator` (Khronos Reference Compiler)
```bash
glslangValidator -V matmul_forward.comp -o matmul_forward.spv
```
