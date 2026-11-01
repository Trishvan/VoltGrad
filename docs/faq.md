# FAQ

## Hardware and platform

**Which GPUs are supported?**
Any device with a Vulkan 1.1+ driver that reports the `shaderInt64` capability. Discovery enumerates every Vulkan device present and requires this feature — Intel, AMD, NVIDIA discrete and integrated GPUs are typical; Apple Silicon works through MoltenVK.

**If I have two GPUs, which one is used?**
The best-scoring device: discrete GPUs rank highest, then integrated, then virtual GPUs, then CPU rasterizers, and among equals the first enumerated wins. Multi-device training is not supported yet; exactly one device is opened per pipeline.

**Do I need the Vulkan SDK installed to use VoltGrad?**
No. The library loads the Vulkan runtime that ships with your GPU drivers (or MoltenVK), and all compute kernels ship precompiled inside the package. The SDK (or `glslc`/`glslangValidator`) is only needed to recompile shaders when you modify the GLSL sources — see [Shader Reference](../voltgrad/shaders/README.md).

**Does it work without cgo?**
Yes. Windows uses the Go standard library's syscall loader; Linux and macOS use `github.com/ebitengine/purego` for dynamic loading. No C compiler or CGO is required on any platform.

**Why is `shaderInt64` mandatory?**
Parameter updates, gradient accumulation, and the optimizer run in 64-bit integer arithmetic so that large-magnitude values never truncate — that is the foundation of the zero-clipping guarantee. The feature gate keeps GPU and CPU bit-for-bit identical by contract instead of by luck.

## Arithmetic and DBW

**Is floating point used anywhere in training?**
No. Every value in the training path — parameters, gradients, activations, losses, hyperparameters — is a fixed-point integer. The only conversion to and from floats is at your data boundary; on the device side, training math stays entirely in integers (the only non-integer touchpoint inside the library is a constant Vulkan queue-priority value used at device initialization).

**What is Dynamic Bit Widening (DBW)?**
Every tensor stores its elements packed at a current bit-width in [2, 64]. Writing a value that does not fit triggers an automatic, deterministic repack into a wider width — no clipping, no loss. See [ADR-0001](adr/0001-tensor-wise-dynamic-bit-widening.md) for the design rationale (one width per tensor, not per-block).

**What happens at 64 bits?**
64 is the ceiling; the representable range is the full signed 64-bit integer range, which is also the widest the CPU path computes. Boundary behavior at every width, including the extremes, is covered by `TestOverflowAndBoundaryConditions` and `TestFull64BitGPUDBWOptimizer`.

**Can a tensor narrow back down?**
No. Bulk loaders (`FromInt32Slice` / `FromInt64Slice`) only ever widen. This keeps widening monotone and predictable; if you need a fresh minimum-width tensor, create a new one.

**How do I check memory savings?**
`MemoryBytes()` gives the packed size, `UncompressedInt32Bytes()` the plain-int32 equivalent, and `CompressionRatio()` their ratio.

## Using the library

**When does the CPU fallback path run?**
Whenever you call the plain (non-`Ctx`) methods — `Forward`, `Backward`, `Step` — or when a `*Ctx` method's context has no GPU engine attached. Both paths produce identical values; the fallback exists so models can run on machines without GPUs.

**Is concurrency supported?**
VoltGrad targets a single-threaded training loop: one device, one compute queue, fence-serialized dispatches. The `BufferPool` is safe for concurrent use (mutex-guarded), but the pipeline, engine, and tensor operations are not — coordinate external access if you touch them from multiple goroutines.

**How can I avoid per-iteration host-to-device transfer overhead?**
Bind layer parameters once with `LinearLayer.BindGPU` (or manage `NewPersistentGpuTensor` directly): weights and bias stay resident in GPU memory across iterations and synchronize only on demand (`SyncToGPU` / `SyncToHost`), instead of being re-uploaded on every forward call. `BenchmarkPersistentGpuTensorSync` measures the round-trip cost.

**When would I use `DispatchAsync`, `DispatchProfiled`, or `ExecutionToken`?**
`Dispatch` blocks the CPU thread until the GPU finishes. `DispatchAsync` instead returns an `ExecutionToken` whose `Wait(timeoutNs)` you call when ready, so host-side work can overlap GPU execution. `DispatchProfiled` returns the measured execution duration, from GPU timestamp queries when the device exposes them and from wall-clock timing otherwise.

**How can I measure and compare CPU versus GPU performance?**
Run the bundled benchmark suite: `go test -bench=. -run '^$' ./voltgrad/`. It includes `BenchmarkCPUMatMul`, `BenchmarkGPUMatMul`, `BenchmarkBatchedGEMM`, `BenchmarkDBWOptimizerStep`, and `BenchmarkPersistentGpuTensorSync`. The GPU benchmarks skip automatically when no `shaderInt64` device is available, so the same command runs on any machine.

**Where is checkpointing / model serialization?**
Not yet — it is on the 1.0 roadmap. State is accessible for manual serialization today: parameters are `*DynamicBitTensor` objects whose contents are readable through `ToInt64Slice()`, and reconstructing a fresh model via `FromInt64Slice` restores weights (bit-widths are re-derived as the minimal fit).

**How much GPU memory does the engine use?**
The visible pieces are host-coherent mapped buffers created per dispatch plus the pooled buffers. The `DeviceReport` lists total device-local and host-visible heaps so you can size workloads; the bundled kernels do not allocate device-local VRAM themselves.

**Why is my first epoch slower than the rest?**
Device and kernel initialization happen at pipeline construction, and the first runs also have widening events that repack tensors; later epochs reuse pooled buffers. Duration comes through on `EpochMetrics.Duration`.

## Errors and troubleshooting

**`no compatible Vulkan compute device with shaderInt64 found`**
No GPU passed the compute-queue plus `shaderInt64` gate. Update drivers, verify with `vulkaninfo`, or run on different hardware.

**`vulkan-1.dll dynamic load failure` / `failed loading Vulkan shared library (...)`**
The Vulkan loader is missing or not on the search path (Windows driver repair; install `libvulkan1` on Linux; MoltenVK on macOS). This is an environment issue, not a library defect.

**`buffer overflow: ... exceeds buffer capacity` / `buffer read bounds error: ... exceeds buffer capacity`**
`GpuBuffer` memory I/O is bounds-checked in both directions: writing a slice that exceeds the buffer's capacity, or reading more elements than the buffer holds, returns an error instead of corrupting mapped memory. Size buffers to `len(data) * elementSize` before writing, and size reads to the number of elements actually present.

**Training runs but loss does not fall**
Check conversion to Q16 (learning-rate 655 ≈ 0.01), that `ZeroGrad()` zeroes every gradient tensor, that `Gradients()` matches `Parameters()` pairwise, and that scratch values are stored before they are needed in `Backward`. Nothing widens and nothing falls typically means dead gradients.

**GPU tests fail on machines without a discrete GPU**
They are hardware tests by design: the parity and pipeline tests require a `shaderInt64` device. Compile-only validation (`go build ./...`) works everywhere.

## Project

**How are the bundled shaders maintained?**
GLSL sources live in `voltgrad/shaders/*.comp` with their SPIR-V binaries committed alongside. Recompile with `go run ./cmd/compileshaders` after editing sources, and commit both together — the CI verifies the `.comp` files keep compiling.

**Where do design decisions get recorded?**
Architecture Decision Records live in `docs/adr/`. [ADR-0001](adr/0001-tensor-wise-dynamic-bit-widening.md) covers the tensor-vs-block DBW decision and the parity-first principle that shapes all numeric code.
