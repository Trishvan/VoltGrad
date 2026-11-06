# ADR 0003: Polyglot Architecture — Go-First, Rust-Powered Execution Runtime via Purego FFI

- **Status**: Accepted
- **Date**: 2026-11-06
- **Deciders**: Core Engineering / Maintainers
- **Context**: Runtime Architecture, Memory Management & Execution Engine Evolution

---

## 1. Context & Problem Statement

VoltGrad is a vendor-neutral, self-contained GPU computing framework designed to train neural networks using arbitrary-precision fixed-point integer arithmetic and **Dynamic Bit Widening (DBW)** ([ADR-0001](0001-tensor-wise-dynamic-bit-widening.md)). In [ADR-0002](0002-dual-path-execution-and-optional-shaderint64.md), VoltGrad established a dual-path execution architecture that enables seamless GPU training across both 64-bit and 32-bit Vulkan hardware, with graceful fallback to an exact 64-bit CPU reference.

To date, VoltGrad has operated as a pure Go host library driving Vulkan compute shaders via dynamic platform loading (`purego` on Unix/macOS, `syscall` on Windows). While this architecture successfully proved zero-CGO Vulkan execution in Go, the increasing scope of host-side execution engine responsibilities introduces architectural tension:

1. **Host-Side Execution Engine Responsibilities**:
   - Vulkan physical device enumeration, capability probing, and logical device creation.
   - Device memory allocation, custom sub-allocation, and buffer pooling.
   - Command buffer pool management, command recording, and multi-stream submission.
   - Fine-grained hardware synchronization (fences, timeline semaphores, pipeline barriers).
   - Asynchronous submission tracking, execution tokens, and timestamp query normalization.
   - Performance-critical CPU fallback kernels (GEMM, activations, losses for $W > 32$ on 32-bit GPUs).
   - CPU-side DBW bitstream repacking and packing/unpacking routines.

2. **The Language Dilemma**:
   - **Why not rewrite entirely in Rust?** Go provides exceptional developer ergonomics, lightning-fast compilation times, a simple concurrency model, trivial cross-compilation, and an approachable user API. Forcing machine-learning practitioners to write Rust would compromise VoltGrad's accessibility and developer velocity.
   - **Why not keep everything in Go?** In Go, low-level Vulkan memory management, manual pointer manipulation, and device lifecycle tracking bypass the type safety of Go's garbage collector without gaining Rust's compile-time RAII (Resource Acquisition Is Initialization), borrow checker guarantees, and zero-cost abstractions. Furthermore, managing native off-heap memory, asynchronous hardware completion, and memory fences in Go requires defensive boilerplate, manual mutex tracking, and custom lifecycle counters that remain vulnerable to synchronization races.
   - **The GPU Compute Reality**: Actual mathematical acceleration (GEMM, DBW repacking, ReLU, MSE, AdamQ) executes exclusively on GPU hardware via GLSL/SPIR-V compute shaders. Neither Go nor Rust executes on the GPU. Rust's role is to optimize the **host-side execution runtime, memory management, and dispatch orchestration**, not to replace the GPU shaders.

VoltGrad requires a clear, principled division of responsibilities that preserves Go's developer experience while granting the execution engine the rigorous safety, zero-overhead memory control, and concurrency guarantees of Rust.

---

## 2. Decision

**VoltGrad adopts a three-tier polyglot runtime architecture: Go-First Frontend, Rust-Powered Execution Engine, and GLSL/SPIR-V Compute Shaders.**

```
+--------------------------------------------------------------------------+
|                       Go — Frontend & Orchestration                      |
|  Public API · Model Architecture · Training Loop · Config · Datasets     |
+--------------------------------------------------------------------------+
                                     |
                       C-Compatible ABI (via purego)
                       Zero-CGO Dynamic Linking
                                     |
                                     v
+--------------------------------------------------------------------------+
|                         Rust — Execution Runtime                         |
|  Vulkan Device Lifecycle · Memory Allocator · Buffers · Synchronization   |
|  Command Recording · Execution Graphs · CPU Fallback & SIMD DBW Repack   |
+--------------------------------------------------------------------------+
                                     |
                         Vulkan Compute Interface
                                     |
                                     v
+--------------------------------------------------------------------------+
|                           GPU — Compute Hardware                         |
|  GLSL / SPIR-V Compute Shaders (64-bit Native & 32-bit Core Pipelines)   |
|  GEMM · DBW Arithmetic · Activations · Losses · Optimizers · Reductions  |
+--------------------------------------------------------------------------+
```

### Component Responsibility & Ownership Matrix

| Subsystem / Component | Assigned Language | Architectural Rationale |
| :--- | :--- | :--- |
| **ModelArchitecture & Layer API** | **Go** | Keep public API idiomatic, composable, and accessible for Go developers. |
| **Training Pipeline & Orchestration** | **Go** | Training loops, dataset iterators, learning rate schedules, and metrics logging belong in Go. |
| **Tensor Public Interface** | **Go** | Go callers interact with high-level `Tensor` objects without unsafe pointer exposure. |
| **Vulkan Hardware & Device Management**| **Rust** | Rust exclusively owns Vulkan instance, physical device, logical device, and queue lifecycles. |
| **GPU Memory & Buffer Pool** | **Rust** | Rust RAII ensures deterministic allocation, pooling, and deallocation with zero GC interference. |
| **Command Buffers & Synchronization** | **Rust** | Fences, semaphores, barriers, and command pools are managed safely under Rust's ownership model. |
| **Execution Graphs & Dispatch** | **Rust** | Low-overhead command recording, multi-pass graph capture, and pipeline binding in native code. |
| **CPU Fallback Arithmetic ($W > 32$)** | **Rust** | Vectorized SIMD implementations for high-precision CPU fallback on 32-bit GPUs. |
| **CPU DBW Bitstream Packing** | **Go / Rust** | Maintained in Go; ported to Rust selectively if empirical profiling demonstrates a bottleneck. |
| **GPU Compute Shaders** | **GLSL / SPIR-V** | Shaders remain independent GLSL compute sources compiled directly to SPIR-V. |

### Strict Vulkan Device Ownership Principle

To eliminate synchronization bugs and dual-ownership leaks, **Rust exclusively owns the Vulkan device lifecycle**:
- Go and Rust will **not** independently manage the same Vulkan device, queues, command pools, or resource lifetimes.
- Rust initializes the Vulkan instance, selects the physical device (evaluating `shaderInt64` support per [ADR-0002](0002-dual-path-execution-and-optional-shaderint64.md)), creates the logical device, and manages all queue families.
- Rust exposes coarse-grained, opaque resource handles (`*BackendHandle`, `*BufferHandle`, `*TokenHandle`) to Go. Go treats these handles as opaque tokens and cannot directly mutate underlying native Vulkan objects.

---

## 3. Communication Boundary: C ABI via Purego (Preserving Zero-CGO)

VoltGrad evaluates two primary integration mechanisms for Go-Rust interoperation:

### Option A: Standard CGO
- Rust compiles as a shared library (`cdylib`) and exposes `extern "C"` functions. Go links through CGO directives (`import "C"`).
- *Downsides*: Breaks VoltGrad's strict zero-CGO design invariant. Requires a native C toolchain (gcc/clang/MinGW) on every developer machine, complicates cross-compilation, degrades Go build times, and incurs CGO stack-switching runtime overhead.

### Option B: C-Compatible ABI via Purego (Chosen)
- Rust compiles as a self-contained shared library (`cdylib`): `.dll` on Windows, `.so` on Linux, and `.dylib` on macOS.
- Rust functions are exported using standard C ABI (`extern "C"`).
- Go dynamically loads the shared library at runtime using `github.com/ebitengine/purego` (or `syscall.NewLazyDLL` on Windows).
- *Advantages*:
  - **Zero CGO Invariant Maintained**: Go projects compile with `CGO_ENABLED=0` without requiring any C toolchain.
  - **Uniform Loading Model**: VoltGrad already leverages `purego` for its Vulkan loader; the same mechanism cleanly binds the Rust runtime library.
  - **Dynamic Backend Swapping**: If the compiled Rust runtime is absent, VoltGrad can transparently fall back to its pure Go reference backend.

### Boundary Design Principles: Coarse-Grained FFI

To avoid FFI overhead and memory safety hazards across the language boundary, the interface adheres to three strict rules:

1. **No Fine-Grained Calls**: The boundary is never crossed for individual tensor elements, scalars, or inner-loop iterations. Calls are made exclusively at the operational level (e.g. `voltgrad_backend_matmul_forward`, `voltgrad_backend_step_optimizer`).
2. **Explicit Memory Ownership Protocols**:
   - Memory allocated by Rust is freed exclusively by Rust (`voltgrad_buffer_destroy`).
   - Memory allocated by Go is passed to Rust as read-only or write-only slices with explicit pointer and length descriptors (`VoltGradSlice { ptr: *mut c_void, len: usize, cap: usize }`).
   - Rust never retains or caches host pointers provided by Go across asynchronous dispatch boundaries unless explicitly registered in a pinned persistent buffer.
3. **Robust Error Propagation**:
   - Rust functions never unwind or panic across the FFI boundary (`catch_unwind` is mandatory).
   - Functions return integer error status codes (`VoltGradStatus`), with detailed thread-local error messages retrievable via `voltgrad_last_error_message()`.

---

## 4. Implementation & Migration Strategy

The migration to the polyglot runtime will proceed in seven controlled, incremental phases:

```
[Phase 1: Go Reference Engine] ---> [Phase 2: ComputeBackend Abstraction]
                                                  |
                                                  v
[Phase 4: Device & Buffers] <------- [Phase 3: Rust Crate & Purego FFI]
         |
         v
[Phase 5: GPU Dispatch & Graphs] --> [Phase 6: Empirical Benchmarking]
                                                  |
                                                  v
                                     [Phase 7: Selective SIMD CPU Port]
```

### Phase 1: Preserve Existing Go Implementation
The current pure-Go Vulkan engine and CPU reference implementation remain fully preserved and operational. The Go implementation serves as the canonical behavioral reference and ensures that VoltGrad remains 100% functional even in environments where the native Rust shared library is omitted.

### Phase 2: Introduce `ComputeBackend` Abstraction in Go
Define a clean internal Go interface that decouples model layers, optimizers, and tensors from the underlying execution implementation:

```go
type ComputeBackend interface {
    Name() string
    HardwareReport() DeviceReport
    CreateBuffer(size uint64, usage BufferUsage) (BufferHandle, error)
    MatMulForward(desc MatMulDescriptor) (ExecutionToken, error)
    MatMulBackward(desc MatMulBackwardDescriptor) (ExecutionToken, error)
    OptimizerStep(desc OptimizerDescriptor) (ExecutionToken, error)
    Synchronize(token ExecutionToken) error
    Close() error
}
```

### Phase 3: Build the Rust Runtime Crate (`voltgrad-runtime`)
Initialize a standalone Rust crate (`voltgrad-runtime`) configured as a `cdylib` library. Utilize `ash` for type-safe Vulkan 1.0 bindings, exposing a minimal `extern "C"` ABI covering library initialization, version negotiation, and capability querying.

### Phase 4: Port Device & Buffer Management to Rust
Implement Vulkan physical device enumeration, capability scoring (including [ADR-0002](0002-dual-path-execution-and-optional-shaderint64.md) 32-bit/64-bit detection), logical device creation, and GPU memory allocation in Rust. Implement a lock-free, RAII-governed buffer pool in Rust that eliminates data races during concurrent buffer release and pool clearing.

### Phase 5: Port GPU Dispatch, Resident Tensors & Execution Graphs
Port pipeline creation, descriptor set binding, compute shader dispatch, and synchronization primitives to Rust. Integrate pre-recorded execution graphs to submit multi-layer forward and backward passes within a single command buffer submission, eliminating per-operation dispatch overhead.

### Phase 6: Empirical Benchmarking & Verification
Run the comprehensive VoltGrad verification test suite comparing the Rust runtime against the Go runtime:
- Validate 100% bit-for-bit numerical parity across all DBW bit-widths ($W \in [2, 64]$).
- Profile host-side dispatch latency, memory allocation counts, and CPU utilization under sustained training.
- Verify zero memory leaks and clean device shutdown across abnormal termination conditions.

### Phase 7: Selective CPU DBW & Fallback Porting
Benchmark CPU-side tensor operations under real training workloads. Port CPU fallback arithmetic ($W > 32$) and DBW bitstream repacking to Rust with explicit AVX2/AVX-512/NEON SIMD vectorization **only if** profiling confirms they represent a measurable bottleneck in the end-to-end pipeline.

---

## 5. Trade-Off Analysis

| Architectural Dimension | Pure Go Runtime (Current) | Polyglot Go + Rust + GLSL (Chosen) |
| :--- | :--- | :--- |
| **Developer API & UX** | Go: Simple, idiomatic, fast iteration. | **Go**: Unchanged; public API remains 100% Go. |
| **Vulkan Safety & RAII** | Manual cleanup, vulnerable to lifecycle bugs. | **Rust RAII**: Resource cleanup guaranteed by compiler. |
| **Memory Allocation** | Relies on Go heap + manual off-heap pointers. | **Rust Allocator**: Zero GC pressure; custom pool allocators. |
| **Build & Toolchain** | Single toolchain (`go build`). | Two toolchains (`cargo` for runtime, `go` for app). |
| **CGO Dependency** | **Zero CGO** (`purego`/`syscall`). | **Zero CGO Preserved** (dynamic `purego` loading of `cdylib`). |
| **Distribution** | Pure Go module source. | Go module with precompiled `cdylib` binaries per OS/arch. |
| **Dispatch Overhead** | Modest Go runtime overhead per dispatch. | Minimal; native execution graphs and queue submissions. |
| **CPU SIMD Optimization**| Restricted to Go compiler autovectorization. | Native SIMD intrinsics (AVX2, AVX-512, NEON) in Rust. |

---

## 6. Consequences & System Invariants

### Positive Consequences
- **Elimination of Host Runtime Concurrency Hazards**: Vulkan resource lifecycles, command fences, and memory allocations are statically checked by Rust's ownership model, preventing device-in-use destruction and double-release bugs.
- **Zero-CGO Invariant Intact**: By utilizing `purego` dynamic loading rather than CGO, VoltGrad retains its zero-CGO compilation model and cross-compilation ergonomics.
- **Lower Latency & CPU Overhead**: Native command buffer recording, queue submission, and execution graph capture drastically reduce host overhead between GPU dispatches.
- **Preserved Go Developer Experience**: Users of VoltGrad continue writing standard, idiomatic Go code without needing to know or write Rust.

### Neutral & Negative Consequences
- **Build Pipeline Complexity**: Building the native runtime from source requires a Rust toolchain (`cargo`). *Mitigation*: Distribute precompiled runtime shared libraries (`voltgrad_runtime.dll`, `libvoltgrad_runtime.so`, `libvoltgrad_runtime.dylib`) alongside GitHub releases, and maintain the pure Go backend as an automatic fallback when the native binary is not installed.
- **FFI Boundary Discipline**: Requires rigorous testing of ABI data structures, alignment, and pointer safety to guarantee cross-language stability.
- **Roadmap Realignment**: Development proceeds through the phased `ComputeBackend` abstraction before fully replacing Go host internals.

---

## 7. References

- [ADR-0001: Tensor-Wise Dynamic Bit Widening](0001-tensor-wise-dynamic-bit-widening.md)
- [ADR-0002: Dual-Path 32-bit and 64-bit Execution with Optional shaderInt64](0002-dual-path-execution-and-optional-shaderint64.md)
- [Ebitengine Purego Documentation](https://github.com/ebitengine/purego)
- [Ash: Lightweight Vulkan Bindings for Rust](https://docs.rs/ash/)
