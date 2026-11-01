# Getting Started

## 1. Overview

VoltGrad runs neural-network training on the GPU through the Vulkan compute API. All arithmetic is done in fixed-point integers; parameters are stored in variable-width packed tensors that expand their bit-width automatically as values grow (Dynamic Bit Widening). Your program defines a model (an interface with two methods: `Forward` and `Backward`, plus parameter bookkeeping); VoltGrad handles device discovery, memory, batching, losses, optimizers, and kernel dispatch.

The import path for the library is:

```go
import "github.com/Trishvan/VoltGrad/voltgrad"
```

## 2. Requirements

| Component | Requirement |
| :--- | :--- |
| Go toolchain | The module requires Go 1.27.1 (`go.mod`). Builds with older toolchains that follow the Go toolchain directive will download it automatically. |
| GPU | A device with a Vulkan 1.1+ driver that reports the `shaderInt64` feature. This feature is mandatory; devices without it are skipped during discovery. |
| GPU drivers | Windows: any recent driver installs `vulkan-1.dll` (loaded via `syscall`, no CGO). Linux: `libvulkan.so.1` or `libvulkan.so` (for example, the `libvulkan1` package plus a Vulkan-capable Mesa/vendor driver). macOS: MoltenVK — the loader probes `libvulkan.1.dylib`, `libMoltenVK.dylib`, `/usr/local/lib/libvulkan.dylib`, and `/opt/homebrew/lib/libvulkan.dylib`. |

Dependency footprint: Windows builds use only the Go standard library. Linux and macOS builds additionally use `github.com/ebitengine/purego` for dynamic loading. All shader binaries (SPIR-V) are embedded in the package, so nothing needs to be installed at runtime beyond GPU drivers.

## 3. Installation

Add the library to your module:

```bash
go get github.com/Trishvan/VoltGrad
```

To pin a known-good release instead of the latest commit:

```bash
go get github.com/Trishvan/VoltGrad@v0.1.1
```

## 4. Verifying the installation

The fastest check is to enumerate the GPU through the library:

```go
package main

import (
    "fmt"

    "github.com/Trishvan/VoltGrad/voltgrad"
)

func main() {
    hw, err := voltgrad.NewHardwareContext()
    if err != nil {
        panic(err) // typically: no Vulkan driver, or no shaderInt64 device
    }
    defer hw.Close()

    fmt.Println(hw.DeviceReport())
}
```

On a healthy system this prints device name, vendor, driver and Vulkan API versions, and memory heaps. If the program panics, see the requirements table above — the most common cause is a driver without Vulkan support, or a GPU without 64-bit integer shader support.

## 5. Quick start

The following program trains a small fully connected network for 50 epochs on a synthetic dataset. It is deliberately short; each part is explained in the [Tutorial](tutorial.md).

```go
package main

import (
    "fmt"
    "math"

    "github.com/Trishvan/VoltGrad/voltgrad"
)

func toQ16(f float64) int32 { return int32(math.Round(f * 65536.0)) }

type Net struct {
    fc1, fc2 *voltgrad.LinearLayer
}

func NewNet(in, hidden, out int) *Net {
    return &Net{
        fc1: voltgrad.NewLinearLayer("fc1", in, hidden, 2, true),
        fc2: voltgrad.NewLinearLayer("fc2", hidden, out, 2, true),
    }
}

func (n *Net) Name() string { return "Net" }

func (n *Net) Layers() []voltgrad.LayerConfig {
    return []voltgrad.LayerConfig{
        {Name: "fc1", InDim: n.fc1.InDim, OutDim: n.fc1.OutDim, InitialBitWidth: 2, UseBias: true},
        {Name: "fc2", InDim: n.fc2.InDim, OutDim: n.fc2.OutDim, InitialBitWidth: 2, UseBias: true},
    }
}

func (n *Net) Parameters() []*voltgrad.DynamicBitTensor {
    return []*voltgrad.DynamicBitTensor{n.fc1.Weight, n.fc1.Bias, n.fc2.Weight, n.fc2.Bias}
}

func (n *Net) Gradients() []*voltgrad.DynamicBitTensor {
    return []*voltgrad.DynamicBitTensor{n.fc1.GradW, n.fc1.GradBias, n.fc2.GradW, n.fc2.GradBias}
}

func (n *Net) ZeroGrad() { n.fc1.ZeroGrad(); n.fc2.ZeroGrad() }

func (n *Net) Forward(ctx *voltgrad.ComputeContext, x []int32) ([]int32, error) {
    h1, _ := n.fc1.ForwardCtx(ctx, x, voltgrad.Q16FractionalBits)
    act := voltgrad.ReLU(h1)
    ctx.SetScratch("x", x)
    ctx.SetScratch("h1pre", h1)
    ctx.SetScratch("h1act", act)
    return n.fc2.ForwardCtx(ctx, act, voltgrad.Q16FractionalBits)
}

func (n *Net) Backward(ctx *voltgrad.ComputeContext, dLoss []int32) error {
    x, _ := ctx.GetScratch("x")
    h1pre, _ := ctx.GetScratch("h1pre")
    h1act, _ := ctx.GetScratch("h1act")

    dH, _ := n.fc2.BackwardCtx(ctx, dLoss, h1act.([]int32), voltgrad.Q16FractionalBits)
    dX := voltgrad.ReLUBackward(dH, h1pre.([]int32))
    _, _ = n.fc1.BackwardCtx(ctx, dX, x.([]int32), voltgrad.Q16FractionalBits)
    return nil
}

func main() {
    // Dataset: y = 3x + 1, in fixed-point Q16 (65536 == 1.0).
    var inputs, targets [][]int32
    for i := 0; i < 16; i++ {
        x := toQ16(float64(i)/16*2 - 1) // -1.0 .. 0.875
        inputs = append(inputs, []int32{x})
        targets = append(targets, []int32{toQ16(3*float64(x)/65536 + 1)})
    }

    model := NewNet(1, 8, 1)

    config := voltgrad.DefaultTrainingConfig()
    config.Epochs = 50
    config.BatchSize = 8

    pipeline, err := voltgrad.NewTrainingPipeline(config, model)
    if err != nil {
        panic(err)
    }
    defer pipeline.Close()

    metrics, err := pipeline.Train(model, inputs, targets)
    if err != nil {
        panic(err)
    }

    last := metrics[len(metrics)-1]
    fmt.Printf("final loss: %d, parameter bit-widths: %v\n", last.MeanLoss, last.ParamBitWidths)
}
```

With `Verbose` enabled (the default) the training loop prints a device banner and one progress line per epoch, including loss, duration, and current parameter bit-widths.

## 6. Next steps

- [Tutorial](tutorial.md) — build the model step by step, inspect DBW behavior, and integrate the batched interface.
- [Training Guide](training-guide.md) — configuration, optimizer selection, loss functions, and monitoring.
- [API Reference](api-reference.md) — signatures and semantics for every exported symbol.
