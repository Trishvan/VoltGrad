# Tutorial: Training Your First Model

This tutorial builds, trains, and inspects a small network from scratch. It assumes the [Getting Started](getting-started.md) requirements are met (Go, a Vulkan device with `shaderInt64`, GPU drivers).

What you will do:

1. Implement the `ModelArchitecture` interface for a 1-8-1 network with ReLU.
2. Prepare a fixed-point dataset.
3. Train it with the default SGD-with-Momentum optimizer.
4. Read the returned metrics and observe Dynamic Bit Widening.
5. Extend the model to the batched training path.

## 1. The model interface

A model is any Go type implementing `ModelArchitecture`:

| Method | Meaning |
| :--- | :--- |
| `Name() string` | Identifier printed in the training banner. |
| `Layers() []LayerConfig` | Descriptors of your parameterized layers. |
| `Parameters() []*DynamicBitTensor` | All trainable tensors, in a fixed order. |
| `Gradients() []*DynamicBitTensor` | Their gradient tensors, pairwise matching `Parameters()`. |
| `ZeroGrad()` | Zero every gradient buffer (called once per batch, before backward). |
| `Forward(ctx, inputs)` | Compute predictions for one sample in Q16. |
| `Backward(ctx, dLoss)` | Backpropagate the loss gradient and accumulate into gradients. |

Use `voltgrad.LinearLayer` for fully connected layers — it owns weight, bias, and gradient tensors and provides forward/backward in both sample-wise and batched form.

## 2. Defining the network

```go
type Net struct {
    fc1, fc2 *voltgrad.LinearLayer
}

func NewNet(in, hidden, out int) *Net {
    // Start all tensors at 2 bits. DBW widens automatically as learning proceeds.
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
```

Two rules to remember:

- Every tensor returned in `Parameters()` must have a gradient counterpart in `Gradients()` at the same index. The optimizer walks the two lists in parallel.
- Gradients accumulate across samples within a batch; the trainer calls `ZeroGrad()` before each batch's backward passes and steps the optimizer after the batch.

## 3. Forward and backward passes

```go
func (n *Net) Forward(ctx *voltgrad.ComputeContext, x []int32) ([]int32, error) {
    h1, _ := n.fc1.ForwardCtx(ctx, x, voltgrad.Q16FractionalBits)
    act := voltgrad.ReLU(h1)

    ctx.SetScratch("x", x)       // inputs needed by backward
    ctx.SetScratch("h1pre", h1)  // pre-activation, needed by ReLUBackward
    ctx.SetScratch("h1act", act) // activation, needed by fc2 backward

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
```

Notes on the `*Ctx` calls:

- They check the context: if the training pipeline has a live GPU engine, the call runs on the GPU; otherwise it falls back to the CPU implementation. The two paths produce identical values, so you can write one forward/backward body for both environments.
- The `shift` argument is the fixed-point scale (`voltgrad.Q16FractionalBits` is standard).
- `ComputeContext.SetScratch` / `GetScratch` store forward values for backward; keys are free-form and values are `interface{}`, so type-assert them back.

## 4. Preparing data

Data is `[]int32` in Q16 fixed point, where 65536 represents 1.0:

```go
func toQ16(f float64) int32 { return int32(math.Round(f * 65536.0)) }
```

Build a small regression dataset (targets for `y = 3x + 1`):

```go
var inputs, targets [][]int32
for i := 0; i < 16; i++ {
    f := float64(i)/15*2 - 1 // x in [-1.0, 1.0]
    inputs = append(inputs, []int32{toQ16(f)})
    targets = append(targets, []int32{toQ16(3*f + 1)})
}
```

Set `len(inputs) == len(targets)`, both non-empty; sample dimension is `len(inputs[0])` (any size).

## 5. Training

```go
model := NewNet(1, 8, 1)

config := voltgrad.DefaultTrainingConfig() // 10 epochs, batch 64, MSE, SGD-momentum 0.01/0.9
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
```

During training (Verbose on by default) you see, per epoch, the mean loss, wall time, and the current bit-width of each parameter tensor — plus a `[DBW Event: N expansions]` marker when any tensor widened during that epoch.

## 6. Reading the results

`Train` returns `[]EpochMetrics`, one entry per epoch:

```go
first := metrics[0]
last := metrics[len(metrics)-1]

fmt.Printf("loss: %d -> %d over %d epochs\n", first.MeanLoss, last.MeanLoss, config.Epochs)
fmt.Printf("final bit-widths: %v\n", last.ParamBitWidths)
for _, w := range last.BitWideningLogs {
    fmt.Println(w) // e.g. "Param[0]: DBW extended 15 bits -> 16 bits (Zero Clipping)"
}
```

What to expect:

- `MeanLoss` starts high (random 2-bit initialization) and falls as training progresses; by construction the values are integers, so small residual jitter between epochs is normal rounding, not a bug.
- `ParamBitWidths` moves from the initial 2 upward until values fit comfortably. Standard-width runs settle far below 32 bits because Q16 learning keeps magnitudes bounded.
- `BitWideningLogs` lists every widening; a run with many early events and none late shows the widening phase completing.

To quantify memory savings:

```go
for i, p := range model.Parameters() {
    fmt.Printf("param %d: %d bytes (%d%% of int32 storage)\n",
        i, p.MemoryBytes(), int(100*p.CompressionRatio()))
}
```

## 7. Going batched

The sample-wise path above dispatches each sample separately. For larger work, implement `BatchedModelArchitecture` so the whole batch flows through one GEMM:

```go
func (n *Net) ForwardBatched(ctx *voltgrad.ComputeContext, x []int32, batchSize int) ([]int32, error) {
    h1, _ := n.fc1.ForwardBatchedCtx(ctx, x, batchSize, voltgrad.Q16FractionalBits)
    act := voltgrad.ReLU(h1)

    ctx.SetScratch("x", x)
    ctx.SetScratch("h1pre", h1)
    ctx.SetScratch("h1act", act)
    return n.fc2.ForwardBatchedCtx(ctx, act, batchSize, voltgrad.Q16FractionalBits)
}

func (n *Net) BackwardBatched(ctx *voltgrad.ComputeContext, dLoss []int32, batchSize int) error {
    x, _ := ctx.GetScratch("x")
    h1pre, _ := ctx.GetScratch("h1pre")
    h1act, _ := ctx.GetScratch("h1act")

    dH, _ := n.fc2.BackwardBatchedCtx(ctx, dLoss, h1act.([]int32), batchSize, voltgrad.Q16FractionalBits)
    dX := voltgrad.ReLUBackward(dH, h1pre.([]int32))
    _, _ = n.fc1.BackwardBatchedCtx(ctx, dX, x.([]int32), batchSize, voltgrad.Q16FractionalBits)
    return nil
}
```

Conventions for the batched interface:

- Inputs and outputs are flat, row-major slices: element `(b, c)` of a `[batchSize, dim]` matrix sits at index `b*dim + c`.
- `BackwardBatched` receives the loss gradient for the whole batch in the same layout; it must accumulate into the layer gradient tensors, which the optimizer consumes after the batch.
- When the model is batched, the trainer computes the loss over the flattened batch in a single call — on the GPU directly for `LossMSE` (`GpuEngine.MSELoss`), otherwise on the CPU — and then runs `BackwardBatched`.

With this interface implemented, the same `pipeline.Train` call trains through the batched GEMM path; nothing else in the setup changes. The complete assembled model (`ExampleMLPArchitecture` in `voltgrad/pipeline_test.go`) is the reference implementation used by the integration test.

## 8. Where to go next

- [Training Guide](training-guide.md) — optimizer selection, loss functions, schedules, and monitoring details.
- [API Reference](api-reference.md) — full signatures for every type used above.
- [FAQ](faq.md) — common questions about hardware support and runtime behavior.
