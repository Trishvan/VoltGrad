---
description: VoltGrad is a self-contained Go machine-learning training library running on Vulkan GPUs with Q16 fixed-point arithmetic and Dynamic Bit Widening.
---

# VoltGrad Documentation

VoltGrad is a self-contained Go library for training neural networks on Vulkan-capable GPUs using fixed-point integer arithmetic and Dynamic Bit Widening (DBW). No CGO, no dependencies on native ML runtimes.

## Documentation Index

| Document | Contents |
| :--- | :--- |
| [Getting Started](getting-started.md) | Requirements, installation, environment verification, and a minimal working program. |
| [Tutorial](tutorial.md) | Building, training, and evaluating a first model end to end. |
| [Training Guide](training-guide.md) | Training configuration, data preparation, loss functions, optimizers, and monitoring. |
| [API Reference](api-reference.md) | Complete reference for every exported type and function. |
| [FAQ](faq.md) | Frequently asked questions about hardware, behavior, and limits. |

## Related Materials

- [Project README](../README.md) — project overview, architecture table, roadmap.
- [Contributing Guide](../CONTRIBUTING.md) — development environment, invariants, and open community tasks.
- [ADR-0001: Tensor-Wise DBW](adr/0001-tensor-wise-dynamic-bit-widening.md) — the architectural decision record for the widening model.
- [ADR-0002: Dual-Path Execution](adr/0002-dual-path-execution-and-optional-shaderint64.md) — the architectural decision record for optional shaderInt64 and dual 32-bit/64-bit GPU paths.
- [Shader Reference](../voltgrad/shaders/README.md) — the Vulkan compute kernels bundled with the library, and how to recompile them.
