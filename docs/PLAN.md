# VoltGrad - plan notes

## Goals
- Vulkan via vulkan-1.dll, pure Go stdlib, no bindings, no CGO.
- Fixed-point Q16 pipeline end to end (weights, grads, loss, optimizer).
- Dynamic Bit Widening: never clip; widen storage instead.
- User writes only the model (Forward/Backward); runtime owns the rest.

## Phases
- [x] pick module layout (standalone vulkankernel folder)
- [ ] C-ABI struct mirrors (instance first, device, then pipeline)
- [ ] packed tensor: get/set, bit packing, widen()
- [ ] device + compute queue discovery
- [ ] host-coherent buffers (zero-copy mapping)
- [ ] SPIR-V kernel + compute pipeline plumbing
- [ ] math: integer GEMV, softmax, losses
- [ ] optimizer: SGD, momentum, Adam-like variant in fixed point
- [ ] trainer loop with epoch metrics + DBW event log
- [ ] integration test (2-bit MLP -> verify expansion + loss drop)

## Session log
- 2026-09-13: drafted this plan; agreed widening strategy (repack over in-place,
  cache predictability beats transient byte savings).
- 2026-09-16: skeleton tensor, chose little-endian bitstream packing.
- 2026-09-18: widen() walk-through, settled on overflow-triggered widening only.
- 2026-09-21: queue-family bitfield review; memory heuristics vs exact table walk.
- 2026-09-28: imported first compiled kernel; counted push-constant size (4B).
- 2026-10-02: loss forms (MSE/CE/Brier) cross-checked on paper against float refs.

## Risks / open questions
- Q16 shifts near INT32 limits: clamp GEMV accumulators.
- Cross-entropy in pure integers is an approximation; document it.
