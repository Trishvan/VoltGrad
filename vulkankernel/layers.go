package vulkankernel

// LinearLayer provides a standard parameterized fully-connected layer with weights and optional bias.
type LinearLayer struct {
	Name    string
	InDim   int
	OutDim  int
	Weight  *DynamicBitTensor // Shape: [OutDim, InDim]
	GradW   *DynamicBitTensor
	Bias    *DynamicBitTensor // Shape: [OutDim]
	GradBias *DynamicBitTensor
	UseBias bool
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
	out, err := MatMulVec64(l.Weight, x, shift)
	if err != nil {
		return nil, err
	}
	if l.UseBias {
		for i := 0; i < l.OutDim; i++ {
			out[i] += l.Bias.GetInt32(i)
		}
	}
	return out, nil
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
