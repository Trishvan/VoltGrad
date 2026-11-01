# VoltGrad Vulkan Compute Shaders

This directory contains the GLSL compute shaders and precompiled SPIR-V binaries (`.spv`) powering VoltGrad's GPU execution engine with 64-bit integer arithmetic and Dynamic Bit Widening (DBW).

## Shaders Overview

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

7. **`relu_forward.comp`** (`relu_forward.spv`)
   - Element-wise rectified linear unit: \( Y_i = \max(0, X_i) \).

8. **`relu_backward.comp`** (`relu_backward.spv`)
   - Element-wise activation backward: \( dX_i = (X_i > 0 \ ?\ dOut_i : 0) \).

9. **`loss_mse.comp`** (`loss_mse.spv`)
   - Computes squared error loss and error gradients directly in GPU VRAM.

10. **`optimizer_step.comp`** (`optimizer_step.spv`)
    - Supports SGD, SGD with Momentum, and AdamQ.
    - Operates on full 64-bit parameters (`int64_t`), gradients, and momentum velocity state.
    - Evaluates dynamic bit-width requirement (`bits_needed_64`) per element and updates target bit-width via `atomicMax(meta[0], needed)`.

11. **`repack_dbw.comp`** (`repack_dbw.spv`)
    - Repacks 64-bit parameter values into contiguous bitstreams in VRAM.
    - Handles multi-word bitstream spans across up to three 32-bit storage words using lockless `atomicOr`.

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
