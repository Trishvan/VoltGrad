# ADR 0002: Dual-Path 32-bit and 64-bit Execution with Optional shaderInt64

- **Status**: Accepted
- **Date**: 2026-11-06
- **Deciders**: Core Engineering / Maintainers
- **Context**: Vulkan Hardware Compatibility & Adaptive Dynamic Precision Execution

---

## 1. Context & Problem Statement

Prior to this decision, VoltGrad required Vulkan physical devices to provide hardware `shaderInt64` (64-bit integer compute shader operations) as a mandatory requirement during device enumeration. Any GPU or platform lacking `shaderInt64` was rejected during initialization.

While `shaderInt64` enables straightforward 64-bit signed integer accumulation and wide dynamic bit widening ($W \le 64$), this mandatory requirement created significant limitations:

1. **Hardware Incompatibility**: Integrated GPUs (e.g. many Intel HD/UHD/Iris graphics, AMD APUs), mobile GPUs, virtual GPUs, and older discrete GPUs do not universally support `shaderInt64` or disable it in drivers.
2. **Memory Footprint**: Persistent GPU parameter, gradient, and velocity buffers preallocated 32 bytes per element (8 bytes $\times$ 4 buffers) even when the active model parameters only required 2 to 16 bits.
3. **Hardware Proposition**: VoltGrad's core mission is *Adaptive, Vendor-Neutral GPU Computing for Go*. Requiring workstation-class 64-bit integer instruction sets universally contradicted this principle.

We needed an architecture that allows VoltGrad to run on any compliant Vulkan 1.0 GPU—from low-power integrated graphics to high-end workstation accelerators—while preserving 100% bit-for-bit mathematical parity and ADR-0001 (Tensor-Wise DBW).

---

## 2. Decision

**VoltGrad makes `shaderInt64` an optional device capability and implements two distinct, parallel execution paths selected at device initialization:**

1. **64-bit Native Execution Path**:
   - Selected automatically when the physical GPU supports `shaderInt64` and 32-bit mode is not forced.
   - Uses native 64-bit GLSL compute shaders (`matmul_forward.comp`, `optimizer_step.comp`, etc.) with `GL_ARB_gpu_shader_int64` / `shaderInt64`.
   - Supports direct GPU acceleration for tensor bit-widths up to 64 bits ($W \le 64$).
   - Preallocates 8 bytes per element per persistent buffer (approx. 32 bytes/element total).

2. **32-bit Core Execution Path**:
   - Selected automatically when the physical GPU does not support `shaderInt64`, or explicitly via `HardwareOptions{Force32Bit: true}` / `VOLTGRAD_FORCE_32BIT=1`.
   - Uses 9 dedicated pure 32-bit compute shaders (`*_i32.comp`) written in standard GLSL 4.50 with zero hardware extension dependencies.
   - Computes exact 64-bit signed product accumulation in registers using standard GLSL built-ins `imulExtended(int, int, out int hi, out int lo)` and `uaddCarry(uint, uint, out uint carry)`.
   - Preallocates 4 bytes per element per persistent buffer (approx. 16 bytes/element total)—**cutting VRAM footprint in half (50% reduction)**.
   - Accelerates tensor bit-widths up to 32 bits ($W \le 32$) directly on the GPU.

3. **Dynamic Precision Selection & Graceful CPU Fallback (ADR-0001 integration)**:
   - Dynamic Bit Widening (DBW) remains strictly tensor-wise.
   - If an operation or parameter on a 32-bit device requires $> 32$ bits of precision, the runtime automatically and transparently routes that operation through the exact 64-bit CPU reference implementation without runtime errors, clipping, or loss of precision.

---

## 3. Mathematical Parity via Pure Core GLSL 4.50

The 32-bit compute pipeline does not truncate intermediate calculations to 32 bits. Instead, it utilizes hardware 32-bit multiplier units to maintain exact 64-bit precision in registers:

```glsl
// Core GLSL 4.50: Exact 64-bit signed product of two 32-bit integers
int hi, lo;
imulExtended(w, x, hi, lo);

// Core GLSL 4.50: Exact 64-bit accumulation with carry
uint carry;
accum_lo = uaddCarry(accum_lo, uint(lo), carry);
accum_hi += hi + int(carry);

// Bit-for-bit shift and clamp identical to 64-bit CPU signed shift
int result = shift_and_clamp(accum_hi, accum_lo, shift);
```

Extensive testing confirms 100% bit-for-bit numerical equivalence between 32-bit GPU execution, 64-bit GPU execution, and CPU reference evaluation across all operations.

---

## 4. Architectural Comparison

| Architectural Dimension | 64-bit Native Path | 32-bit Core Path |
| :--- | :--- | :--- |
| **Vulkan Hardware Requirement** | Vulkan 1.0 + `shaderInt64` | Baseline Vulkan 1.0 (Core GLSL 4.50) |
| **Hardware Compatibility** | Workstation & high-end discrete GPUs | All GPUs (Integrated, Discrete, APU, Virtual) |
| **Native GPU Bit-Width Range** | $W \in [2, 64]$ bits | $W \in [2, 32]$ bits |
| **Fallback for $W > 32$** | None needed (native GPU $W \le 64$) | Automatic, transparent 64-bit CPU reference |
| **Persistent VRAM Footprint** | ~32 bytes / element (4 $\times$ 8 bytes) | ~16 bytes / element (4 $\times$ 4 bytes) — **50% savings** |
| **Intermediate Accumulation** | Native 64-bit registers (`int64_t`) | Split `imulExtended` / `uaddCarry` 64-bit registers |
| **Bitstream Repacking on GPU** | `repack_dbw.comp` ($W \le 64$) | `repack_dbw_i32.comp` ($W \le 32$) |

---

## 5. Consequences & Impact

### Positive
- **Universal Hardware Support**: VoltGrad now executes GPU training on integrated Intel/AMD GPUs, virtual machines, and low-cost development environments without requiring `shaderInt64`.
- **50% Memory Reduction**: Devices in 32-bit mode consume half the VRAM for persistent tensors, allowing larger batch sizes and models.
- **Zero Numerical Divergence**: Exact bit-for-bit parity is maintained across both GPU paths and the CPU reference.
- **Configurable Control**: Developers can benchmark both paths on the same physical hardware via `HardwareOptions.Force32Bit` or `VOLTGRAD_FORCE_32BIT=1`.

### Negative / Trade-Offs
- Maintenance of 9 additional GLSL compute shaders (`*_i32.comp`). This is mitigated by automated compilation in `cmd/compileshaders` and comprehensive CI test coverage.
