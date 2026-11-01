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

4. **`optimizer_step.comp`** (`optimizer_step.spv`)
   - Supports SGD, SGD with Momentum, and AdamQ.
   - Operates on full 64-bit parameters (`int64_t`), gradients, and momentum velocity state.
   - Evaluates dynamic bit-width requirement (`bits_needed_64`) per element and updates target bit-width via `atomicMax(meta[0], needed)`.

5. **`repack_dbw.comp`** (`repack_dbw.spv`)
   - Repacks 64-bit parameter values into contiguous bitstreams in VRAM.
   - Handles multi-word bitstream spans across up to three 32-bit storage words using lockless `atomicOr`.

## Compiling Shaders

To recompile the GLSL `.comp` files to `.spv` SPIR-V binaries:

### Option A: Using Node.js with `@webgpu/glslang` (cross-platform, zero C++ toolchain needed)
```bash
npm install @webgpu/glslang
node compile.js
```

### Option B: Using the Vulkan SDK `glslangValidator`
```bash
glslangValidator -V matmul_forward.comp -o matmul_forward.spv
glslangValidator -V matmul_backward_dx.comp -o matmul_backward_dx.spv
glslangValidator -V matmul_backward_dw.comp -o matmul_backward_dw.spv
glslangValidator -V optimizer_step.comp -o optimizer_step.spv
glslangValidator -V repack_dbw.comp -o repack_dbw.spv
```
