# ADR 0001: Tensor-Wise Dynamic Bit Widening (DBW) Architecture Decision

- **Status**: Accepted
- **Date**: 2026-11-01
- **Deciders**: Core Engineering / Maintainers
- **Context**: VoltGrad Runtime Architecture & Dynamic Bit Widening (DBW)

---

## 1. Context & Problem Statement

VoltGrad is a vendor-neutral, self-contained GPU computing runtime for Go designed to train neural networks using arbitrary-precision fixed-point integer arithmetic and **Dynamic Bit Widening (DBW)** ($W \in [2, 64]$ bits). Under DBW, parameter tensors initialize at minimal bit-depths (e.g. 2 bits) and expand dynamically as gradients and optimizer momentum steps demand, avoiding quantization clipping while maximizing memory efficiency and arithmetic throughput.

During the architecture design of the runtime, two structural granularities for DBW were evaluated:

1. **Tensor-Wise DBW**: Each tensor carries a single uniform dynamic bit-width across all its elements. When an element update exceeds the representable range, the entire tensor widens, and its contiguous bitstream is repacked.
2. **Block-Wise DBW**: Tensors are subdivided into fixed-size blocks (e.g., $16 \times 16$ or 32 elements), each carrying its own independent dynamic bit-width and scaling metadata.

While block-wise DBW can theoretically achieve higher bit utilization in cases where tensor sub-regions exhibit heterogeneous gradient magnitudes, it introduces substantial structural and numerical complexity. We needed to establish a definitive architectural principle governing DBW granularity for VoltGrad.

---

## 2. Decision

**VoltGrad will standardize on Tensor-Wise Dynamic Bit Widening as its core architectural foundation.**

Block-wise DBW is excluded from the core production roadmap and preserved strictly as an exploratory research topic for future study. VoltGrad will concentrate all engineering efforts on making tensor-wise DBW exceptionally dependable, numerically verified, and performant.

---

## 3. Core Design Principles

This decision establishes four foundational design principles for VoltGrad:

1. **One Dynamic Bit-Width Per Tensor**: Keep width selection, metadata, and memory layout completely uniform across all elements in a tensor.
2. **Predictable Widening**: Define deterministic, reproducible rules for when a tensor expands bit-width and how values are repacked into contiguous host/device storage.
3. **Strict CPU/GPU Parity as a Requirement**: The CPU reference backend and Vulkan GPU compute shaders must follow identical widening triggers, clamping behavior, and overflow semantics down to the single-bit level.
4. **Correctness & Community Maintainability Before Theoretical Compression**: In an open-source project, dependable execution, verifiable correctness, and accessible code structure take strict priority over marginal compression gains that complicate the codebase.

---

## 4. Trade-Off Analysis

| Architectural Dimension | Tensor-Wise DBW (Chosen) | Block-Wise DBW (Rejected for Core) |
| :--- | :--- | :--- |
| **Bit-Width Selection** | **One width per tensor**; trivial state tracking. | **Per-block width tracking**; requires complex spatial coordination. |
| **Metadata Overhead** | **Minimal** (single uint32 for bit-width, element count). | **High** (metadata arrays per block, tiling strides, alignment padding). |
| **Compute Kernel Complexity** | **Low to Moderate**; straightforward strided memory loops in GLSL. | **High**; non-uniform inner loop dispatch, thread divergence within warps/subgroups. |
| **Memory Utilization** | Uniform width determined by tensor peak value. | Marginally better for tensors with extreme localized variance. |
| **Repacking Mechanics** | Clean, parallel linear bitstream repacking across contiguous words. | Complex multi-block scatter/gather repacking with fragmentation risk. |
| **CPU/GPU Parity** | **Easily validated & maintained bit-for-bit** across all hardware. | Prone to subtle boundary discrepancies, edge-case rounding, and race conditions. |
| **Debugging & Profiling** | **Straightforward**; inspectable bit-depths per layer. | **Opaque**; requires specialized block-level visualization tools. |
| **Maintenance Burden** | **Sustainable for open-source contributors**; clear mental model. | Requires dedicated commercial engineering team for continuous kernel tuning. |

---

## 5. Rationale & Commercial vs. Open-Source Realities

Block-wise DBW is not inherently numerically unstable, but its complexity creates exponential surface area for implementation bugs:
- Irregular memory strides cause thread divergence in GPU compute kernels.
- Repacking partial blocks introduces fine-grained synchronization hazards across thread blocks.
- Maintaining identical cross-platform CPU and Vulkan GPU implementations across edge cases requires disproportionate verification effort.

In a commercial setting with a dedicated compiler and kernel optimization team, the maintenance overhead of block-wise representation might be justified to eke out marginal memory bandwidth gains.

For an open-source runtime like VoltGrad, the primary imperative is **trust, maintainability, and reproducibility**. External contributors must be able to understand the execution model, run tests with confidence, and add custom layers without navigating multi-tiered block metadata. Tensor-wise DBW provides a rock-solid, predictable model that is verifiable, robust, and fast.

---

## 6. Consequences & Implementation Impact

- **Roadmap Update**: Replaced "Block-wise DBW" on the VoltGrad 0.4 roadmap with focused milestones on **Adaptive Precision Policies**, **Mixed-Precision Execution**, **Fused Operations**, and **Execution Graphs** atop the tensor-wise engine.
- **Verification Focus**: Engineering effort remains focused on rigorous randomized parity testing, boundary validation, and zero-allocation buffer management for tensor-wise DBW.
- **Future Research**: Block-wise DBW will only be reconsidered if empirical, reproducible benchmarks in real-world workloads demonstrate a compelling efficiency gain that decisively justifies the maintenance cost.
