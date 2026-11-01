package voltgrad

// LinearLayer provides a standard parameterized fully-connected layer with weights and optional bias.
type LinearLayer struct {
	Name      string
	InDim     int
	OutDim    int
	Weight    *DynamicBitTensor // Shape: [OutDim, InDim]
	GradW     *DynamicBitTensor
	Bias      *DynamicBitTensor // Shape: [OutDim]
	GradBias  *DynamicBitTensor
	UseBias   bool
	GpuWeight *PersistentGpuTensor
	GpuBias   *PersistentGpuTensor
}

// NewLinearLayer creates a linear layer starting with initialBits (e.g. 2 bits).
func NewLinearLayer(name string, inDim, outDim int, initialBits int, useBias bool) *LinearLayer {
	if initialBits < 2 {
		initialBits = 2
	}

	l := &LinearLayer{
		Name:    name,
		InDim:   inDim,
		OutDim:  outDim,
		Weight:  NewDynamicBitTensor([]int{outDim, inDim}, initialBits),
		GradW:   NewDynamicBitTensor([]int{outDim, inDim}, initialBits),
		UseBias: useBias,
	}

	if useBias {
		l.Bias = NewDynamicBitTensor([]int{outDim}, initialBits)
		l.GradBias = NewDynamicBitTensor([]int{outDim}, initialBits)
	}

	return l
}

// Forward computes output = (W * x) >> shift + bias.
func (l *LinearLayer) Forward(x []int32, shift uint) ([]int32, error) {
	var bias *DynamicBitTensor
	if l.UseBias {
		bias = l.Bias
	}
	return MatMulVecWithBias64(l.Weight, x, bias, shift)
}

// ForwardCtx computes output using physical GPU acceleration when available, with automatic CPU fallback.
func (l *LinearLayer) ForwardCtx(ctx *ComputeContext, x []int32, shift uint) ([]int32, error) {
	if ctx != nil && ctx.IsGPU() {
		return ctx.GpuEngine().MatMulForward(l.Weight, x, l.Bias, shift)
	}
	return l.Forward(x, shift)
}

// ForwardBatched computes output for a batch: Y[B, OutDim] = (X[B, InDim] * W^T) >> shift + bias.
func (l *LinearLayer) ForwardBatched(x []int32, batchSize int, shift uint) ([]int32, error) {
	out := make([]int32, batchSize*l.OutDim)
	for b := 0; b < batchSize; b++ {
		sampleX := x[b*l.InDim : (b+1)*l.InDim]
		sampleY, err := l.Forward(sampleX, shift)
		if err != nil {
			return nil, err
		}
		copy(out[b*l.OutDim:(b+1)*l.OutDim], sampleY)
	}
	return out, nil
}

// ForwardBatchedCtx computes batched output using GPU acceleration when available, with automatic CPU fallback.
func (l *LinearLayer) ForwardBatchedCtx(ctx *ComputeContext, x []int32, batchSize int, shift uint) ([]int32, error) {
	if ctx != nil && ctx.IsGPU() {
		var bias *DynamicBitTensor
		if l.UseBias {
			bias = l.Bias
		}
		return ctx.GpuEngine().MatMulBatchedForward(l.Weight, x, batchSize, bias, shift)
	}
	return l.ForwardBatched(x, batchSize, shift)
}

// Backward calculates gradients dW and dX.
func (l *LinearLayer) Backward(dOut []int32, x []int32, shift uint) ([]int32, error) {
	if l.UseBias {
		for i := 0; i < l.OutDim; i++ {
			l.GradBias.Add(i, int64(dOut[i]))
		}
	}
	return MatMulVecBackward(dOut, x, l.Weight, l.GradW, shift)
}

// BackwardCtx calculates gradients using physical GPU acceleration when available, with automatic CPU fallback.
func (l *LinearLayer) BackwardCtx(ctx *ComputeContext, dOut []int32, x []int32, shift uint) ([]int32, error) {
	if ctx != nil && ctx.IsGPU() {
		return ctx.GpuEngine().MatMulBackward(dOut, x, l.Weight, l.GradW, l.GradBias, shift)
	}
	return l.Backward(dOut, x, shift)
}

// BackwardBatched calculates gradients across an entire batch on the CPU.
func (l *LinearLayer) BackwardBatched(dOut []int32, x []int32, batchSize int, shift uint) ([]int32, error) {
	dX := make([]int32, batchSize*l.InDim)
	for b := 0; b < batchSize; b++ {
		sampleDOut := dOut[b*l.OutDim : (b+1)*l.OutDim]
		sampleX := x[b*l.InDim : (b+1)*l.InDim]
		sampleDX, err := l.Backward(sampleDOut, sampleX, shift)
		if err != nil {
			return nil, err
		}
		copy(dX[b*l.InDim:(b+1)*l.InDim], sampleDX)
	}
	return dX, nil
}

// BackwardBatchedCtx calculates gradients using physical GPU batched acceleration when available.
func (l *LinearLayer) BackwardBatchedCtx(ctx *ComputeContext, dOut []int32, x []int32, batchSize int, shift uint) ([]int32, error) {
	if ctx != nil && ctx.IsGPU() {
		var gradBias *DynamicBitTensor
		if l.UseBias {
			gradBias = l.GradBias
		}
		return ctx.GpuEngine().MatMulBatchedBackward(dOut, x, batchSize, l.Weight, l.GradW, gradBias, shift)
	}
	return l.BackwardBatched(dOut, x, batchSize, shift)
}

// ZeroGrad zeroes out gradient tensors.
func (l *LinearLayer) ZeroGrad() {
	for i := 0; i < l.GradW.Len(); i++ {
		l.GradW.Set(i, 0)
	}
	if l.UseBias {
		for i := 0; i < l.GradBias.Len(); i++ {
			l.GradBias.Set(i, 0)
		}
	}
}

// InitWeightsIdentity initializes diagonal elements to 1.0 (Q16) and others to 0.
func (l *LinearLayer) InitWeightsIdentity(scale int32) {
	for r := 0; r < l.OutDim; r++ {
		for c := 0; c < l.InDim; c++ {
			idx := r*l.InDim + c
			if r == c {
				l.Weight.SetInt32(idx, scale)
			} else {
				l.Weight.SetInt32(idx, 0)
			}
		}
	}
}

// BindGPU allocates and binds persistent GPU buffers for the layer's weights and biases.
func (l *LinearLayer) BindGPU(hw *HardwareContext) error {
	pw, err := NewPersistentGpuTensor(hw, l.Weight)
	if err != nil {
		return err
	}
	l.GpuWeight = pw

	if l.UseBias && l.Bias != nil {
		pb, err := NewPersistentGpuTensor(hw, l.Bias)
		if err != nil {
			pw.Free()
			l.GpuWeight = nil
			return err
		}
		l.GpuBias = pb
	}
	return nil
}

// SyncToHost synchronizes resident GPU weights and biases back into CPU memory.
func (l *LinearLayer) SyncToHost() error {
	if l.GpuWeight != nil {
		if err := l.GpuWeight.SyncToHost(); err != nil {
			return err
		}
	}
	if l.GpuBias != nil {
		if err := l.GpuBias.SyncToHost(); err != nil {
			return err
		}
	}
	return nil
}

// FreeGPU releases all persistent GPU tensor allocations for this layer.
func (l *LinearLayer) FreeGPU() {
	if l.GpuWeight != nil {
		l.GpuWeight.Free()
		l.GpuWeight = nil
	}
	if l.GpuBias != nil {
		l.GpuBias.Free()
		l.GpuBias = nil
	}
}
