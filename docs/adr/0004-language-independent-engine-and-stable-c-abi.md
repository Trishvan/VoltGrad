# ADR 0004: Language-Independent Engine Core with Stable C-Compatible ABI & Pluggable Backends (Go-First, Not Go-Only)

- **Status**: Accepted
- **Date**: 2026-11-06
- **Deciders**: Core Engineering / Maintainers
- **Context**: Engine Architecture, Public ABI, Multi-Language Strategy & Backend Abstraction

---

## 1. Context & Problem Statement

VoltGrad began as a pure Go library designed to train neural networks on Vulkan GPUs using fixed-point arithmetic and Dynamic Bit Widening (DBW). In [ADR-0001](0001-tensor-wise-dynamic-bit-widening.md), we standardized on tensor-wise DBW. In [ADR-0002](0002-dual-path-execution-and-optional-shaderint64.md), we introduced dual-path 32-bit and 64-bit execution for universal hardware compatibility. In [ADR-0003](0003-polyglot-go-rust-execution-runtime.md), we adopted a polyglot architecture utilizing Rust for low-level host runtime management via zero-CGO `purego` dynamic linking.

As VoltGrad evolves from a specialized prototype into a production-grade machine learning computing engine, an essential architectural question arises regarding its scope and long-term ecosystem integration:

1. **The Risk of Language Lock-In**:
   If the compute engine is tightly coupled to Go-specific types (e.g. Go channels, interface values, garbage-collected slice headers, or Go runtime structures), no other programming language can consume VoltGrad's capabilities without either completely reimplementing the runtime or embedding a full Go runtime environment.
2. **The Risk of Backend Coupling**:
   If the engine's public interface leaks Vulkan-specific concepts (such as `VkDevice`, `VkBuffer`, `VkCommandBuffer`, or pipeline layouts), it becomes impossible to introduce alternate or future execution backends (such as WebGPU, DirectCompute, Metal, or high-performance CPU SIMD) without breaking caller code.
3. **The Core Realization**:
   **VoltGrad is Go-first, not Go-only.**
   Go is the primary, first-class development experience for orchestrating models, configuring pipelines, and developer workflows. However, the underlying compute engine—responsible for tensor memory management, Dynamic Bit Widening repacking, execution graph scheduling, and hardware acceleration—is an independent computational asset that should be accessible to any modern programming language.

VoltGrad requires a foundational architectural commitment to a **language-independent engine core**, exposed through a **stable, minimal C-compatible ABI**, capable of powering multiple first-party and community language bindings and driving pluggable hardware backends.

---

## 2. Core Architectural Principles

This decision establishes three guiding principles for VoltGrad's system architecture:

```
                          +-------------------------+
                          |     VoltGrad Engine     |
                          | (Core Runtime & State)  |
                          +-------------------------+
                                       |
                       Stable C-Compatible ABI (C-FFI)
                                       |
            +--------------------------+--------------------------+
            |                          |                          |
    +---------------+          +---------------+          +---------------+
    |  Go Binding   |          | Rust Binding  |          | Other Binding |
    | (First-Class) |          |   (Future)    |          | (e.g. Python) |
    +---------------+          +---------------+          +---------------+
            |                          |                          |
       Go ML Users               Rust Systems              Data Scientists
            |                          |                          |
            +--------------------------+--------------------------+
                                       |
                       Internal Backend Abstraction
                                       |
            +--------------------------+--------------------------+
            |                          |                          |
    +---------------+          +---------------+          +---------------+
    | Vulkan Driver |          | WebGPU Driver |          | CPU Reference |
    | (32 & 64-bit) |          |   (Future)    |          | (Bit-Exact)   |
    +---------------+          +---------------+          +---------------+
            |                          |                          |
       Physical GPU              Web/Embedded                Host CPU
```

### Principle 1: One Engine, Multiple Language Bindings
- A single canonical engine core implements tensor storage, Dynamic Bit Widening repacking algorithms, execution scheduling, and memory pooling.
- **Go is the flagship, first-class binding**: Go developers enjoy an ergonomic, native API with full documentation, tutorials, and idiomatic patterns.
- Additional language bindings (Rust, Python, C/C++, Zig) interact with the exact same engine core through the stable ABI without reimplementing kernel dispatch, memory management, or DBW logic.

### Principle 2: Language-Independent Interfaces
- The engine's public ABI exposes **zero language-specific types**: no Go slices, no Go channels, no Rust lifetimes, and no C++ templates.
- The interface adheres strictly to standard C calling conventions (`extern "C"`, standard C layout), using explicit primitive types (`int32_t`, `int64_t`, `uint32_t`, `size_t`) and fixed-layout structures.
- All internal states, device contexts, memory allocations, and execution graphs are encapsulated behind **opaque pointer handles**.

### Principle 3: Language-Specific Developer Experiences
- While the ABI is strictly language-agnostic, each language binding must feel completely natural, idiomatic, and native to its respective ecosystem:
  - **Go Binding**: Idiomatic Go structs, method receivers, multiple return values `(*Tensor, error)`, standard Go slices for host I/O, and zero-CGO dynamic linking via `purego`.
  - **Rust Binding**: Strong typing, RAII `Drop` implementations for automatic handle destruction, `Result<T, VoltGradError>`, and zero-cost traits.
  - **Python Binding**: Object-oriented models (`nn.Module` style), NumPy array buffer interoperability, context managers (`with voltgrad.cuda_stream():`), and vectorized tensor indexing.

---

## 3. C-Compatible ABI Design Specifications

To ensure maximum performance, zero memory churn, and long-term ABI stability, the C interface is designed around four foundational patterns:

### 1. Opaque Handles
Internal engine resources are never exposed as concrete structs across the boundary. Instead, callers hold opaque pointers that represent engine entities:

```c
typedef struct VoltGradEngine*   VoltGradEngineHandle;
typedef struct VoltGradTensor*   VoltGradTensorHandle;
typedef struct VoltGradGraph*    VoltGradGraphHandle;
typedef struct VoltGradToken*    VoltGradTokenHandle;
```

Opaque handles guarantee that the internal implementation, layout, and memory representations of tensors and engines can evolve without breaking ABI compatibility with compiled client bindings.

### 2. Explicit Tensor Descriptors (`VoltGradTensorDesc`)
Tensors are defined using explicit, C-compatible descriptor structs specifying rank, dimensions, bit-width, scaling, and memory residency:

```c
typedef enum {
    VOLTGRAD_RESIDENCY_HOST             = 0,
    VOLTGRAD_RESIDENCY_DEVICE_PERSISTENT = 1,
    VOLTGRAD_RESIDENCY_SHARED           = 2
} VoltGradResidency;

typedef struct {
    uint32_t           rank;
    const int64_t*     shape;             /* Array of dimension lengths */
    int32_t            bit_width;         /* Current DBW bit-width (2..64) */
    int32_t            fixed_point_scale; /* Fixed-point fractional bits (e.g. Q16) */
    VoltGradResidency  residency;          /* Host, persistent GPU, or shared */
} VoltGradTensorDesc;
```

### 3. Explicit Ownership & Lifecycles
To eliminate memory leaks, use-after-free, and cross-runtime garbage collection conflicts:
- Resources allocated by the engine are destroyed exclusively by the engine:
  ```c
  VoltGradStatus voltgrad_tensor_create(VoltGradEngineHandle engine, const VoltGradTensorDesc* desc, VoltGradTensorHandle* out_tensor);
  VoltGradStatus voltgrad_tensor_destroy(VoltGradTensorHandle tensor);
  ```
- Host buffers passed into the engine for read/write are described by contiguous pointer-and-length views (`const void* data, size_t byte_len`), with deterministic synchronous copy guarantees.

### 4. Batched Operations & Graph Execution (Zero Boundary Churn)
Calling across the FFI boundary for every individual scalar or tensor element causes intolerable overhead. Furthermore, copying large tensors across the host-device boundary destroys training performance.

The ABI is designed for **coarse-grained, resident execution**:
- **Resident Parameter Preservation**: Persistent GPU tensors (weights, gradients, momentum velocities) remain in device memory across forward, backward, and optimizer steps. The binding passes opaque handles (`VoltGradTensorHandle`) without transferring buffer bytes across the FFI.
- **Batched Dispatch & Graph Capture**:
  Callers can record a sequence of operations into a `VoltGradGraphHandle` and submit the entire graph with a single ABI call:
  ```c
  VoltGradStatus voltgrad_graph_begin(VoltGradEngineHandle engine, VoltGradGraphHandle* out_graph);
  VoltGradStatus voltgrad_graph_add_matmul(VoltGradGraphHandle graph, VoltGradTensorHandle a, VoltGradTensorHandle b, VoltGradTensorHandle out);
  VoltGradStatus voltgrad_graph_add_relu(VoltGradGraphHandle graph, VoltGradTensorHandle x, VoltGradTensorHandle out);
  VoltGradStatus voltgrad_graph_submit(VoltGradGraphHandle graph, VoltGradTokenHandle* out_token);
  VoltGradStatus voltgrad_token_wait(VoltGradTokenHandle token, uint64_t timeout_ns);
  ```

---

## 4. Pluggable Execution Backend Abstraction

The public ABI deliberately abstracts away all backend-specific details. Callers interact with generalized computational primitives (`matmul`, `relu`, `repack_dbw`, `optimizer_step`), not Vulkan pipelines or GPU queues.

Underneath the stable ABI, the engine core defines an internal driver interface:

```
+--------------------------------------------------------------+
|                   VoltGrad Engine Core                       |
|        (DBW Policies, Tensor Graph, Memory Allocator)        |
+--------------------------------------------------------------+
                               |
               Internal Backend Driver Interface
                               |
       +-----------------------+-----------------------+
       |                                               |
       v                                               v
+-----------------------------+                 +-----------------------------+
|    Vulkan Compute Driver    |                 |     CPU Reference Driver    |
| (64-bit & 32-bit Pipelines) |                 |   (Bit-Exact Verification)  |
+-----------------------------+                 +-----------------------------+
```

### Supported & Future Backends
1. **Vulkan Backend (Primary)**:
   - Full support for 64-bit native (`shaderInt64`) and 32-bit core pipelines as formalized in [ADR-0002](0002-dual-path-execution-and-optional-shaderint64.md).
   - High-throughput asynchronous compute queue dispatch.
2. **CPU Reference Backend (Canonical Parity)**:
   - Bit-exact, platform-independent reference implementation.
   - Provides verification baselines and automatic fallback when DBW widens past GPU capabilities or when running in headless environments.
3. **Future Hardware Backends**:
   - **WebGPU Driver**: For browser execution and lightweight embedded client devices.
   - **Metal Driver**: For optimized execution on Apple Silicon macOS/iOS hardware without MoltenVK translation overhead.

---

## 5. Architectural Comparison

| Architectural Dimension | Go-Centric Framework (Prior Model) | Language-Independent Engine (ADR-0004) |
| :--- | :--- | :--- |
| **Language Target** | Exclusively Go applications. | Any language with C FFI support (Go, Rust, Python, etc.). |
| **Public Engine Interface** | Go structs (`*GpuEngine`, `*Tensor`). | Stable C ABI with opaque handles (`VoltGradEngineHandle`). |
| **Go Integration Model** | Direct module dependency. | First-class Go binding leveraging zero-CGO `purego`. |
| **Hardware Coupling** | Direct Vulkan structs in host code. | Pluggable backend driver interface (Vulkan, CPU, Future WebGPU). |
| **Memory Boundary Overhead** | Frequent slice allocations and copying. | Resident device handles with zero-copy batched execution. |
| **Ecosystem Portability** | Limited to the Go ecosystem. | Universal; embeddable into games, servers, and Python pipelines. |

---

## 6. Implementation & Roadmap Alignment

The implementation of ADR-0004 integrates cleanly with the polyglot roadmap established in ADR-0003:

1. **Step 1: ABI Specification Header (`include/voltgrad.h`)**:
   Formalize the complete C ABI header file defining all types, enums, descriptors, and function signatures.
2. **Step 2: Engine C-ABI Layer**:
   Implement the C-ABI export layer in the execution engine, ensuring all error codes, opaque handle lookups, and memory management adhere to strict C calling conventions.
3. **Step 3: Go First-Class Binding via Purego**:
   Update VoltGrad's Go package (`voltgrad/`) to act as the flagship language binding, dynamically loading the engine library via `purego` while preserving the existing public Go API (`ModelArchitecture`, `LinearLayer`, `Optimizer`).
4. **Step 4: Rust Language Binding**:
   Create a safe, idiomatic Rust binding crate (`voltgrad-rs`) wrapping the C ABI with RAII types and typed traits.
5. **Step 5: Python Interoperability (Long-Term)**:
   Provide a lightweight Python binding (`voltgrad-py`) enabling data scientists to train with fixed-point DBW from Python scripts and Jupyter notebooks.

---

## 7. Consequences & Invariants

### Positive
- **Future-Proof Extensibility**: VoltGrad can serve Go backends, Rust microservices, or Python experimentation without duplicating the core compute kernels or DBW logic.
- **Zero-CGO Preserved for Go**: Go applications continue compiling with `CGO_ENABLED=0` by dynamically binding the engine shared library via `purego`.
- **Backend Portability**: Future compute backends (WebGPU, Metal) can be introduced without altering a single line of client application code.
- **Zero-Copy Performance**: Opaque handles and graph execution eliminate unnecessary buffer transfers between host bindings and device memory.

### Negative / Trade-Offs
- **ABI Stability Commitment**: Any change to the public C ABI requires strict semantic versioning and backward compatibility discipline.
- **Header Maintenance**: Requires maintaining `include/voltgrad.h` alongside binding updates.

---

## 8. References

- [ADR-0001: Tensor-Wise Dynamic Bit Widening](0001-tensor-wise-dynamic-bit-widening.md)
- [ADR-0002: Dual-Path 32-bit and 64-bit Execution with Optional shaderInt64](0002-dual-path-execution-and-optional-shaderint64.md)
- [ADR-0003: Polyglot Architecture — Go-First, Rust-Powered Execution Runtime via Purego FFI](0003-polyglot-go-rust-execution-runtime.md)
- [Purego: Call C functions from Go without Cgo](https://github.com/ebitengine/purego)
