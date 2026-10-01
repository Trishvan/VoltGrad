package vulkankernel

import (
	"fmt"
	"math"
)

// MatMulVec64: integer GEMV under Q-shift scaling.
func MatMulVec64(w *DynamicBitTensor, x []int32, shift uint) ([]int32, error) {
	shape := w.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("weight tensor must be 2D [rows, cols], got %v", shape)
	}
	rows, cols := shape[0], shape[1]
	if len(x) != cols {
		return nil, fmt.Errorf("vector length %d != weight cols %d", len(x), cols)
	}

	out := make([]int32, rows)
	for r := 0; r < rows; r++ {
		var accum int64
		rowOffset := r * cols
		for c := 0; c < cols; c++ {
			wVal := w.Get(rowOffset + c)
			accum += wVal * int64(x[c])
		}
		res := accum >> shift
		if res > math.MaxInt32 {
			res = math.MaxInt32
		} else if res < math.MinInt32 {
			res = math.MinInt32
		}
		out[r] = int32(res)
	}
	return out, nil
}

// MatMulVecBackward computes gradients for weights and input vector:
//
//	dW[i,j] += (dOut[i] * x[j]) >> shift
//	dX[j]   += (sum_i dOut[i] * W[i,j]) >> shift
func MatMulVecBackward(dOut []int32, x []int32, w *DynamicBitTensor, dW *DynamicBitTensor, shift uint) ([]int32, error) {
	shape := w.Shape()
	rows, cols := shape[0], shape[1]

	if len(dOut) != rows {
		return nil, fmt.Errorf("dOut len %d != rows %d", len(dOut), rows)
	}
	if len(x) != cols {
		return nil, fmt.Errorf("x len %d != cols %d", len(x), cols)
	}

	dX := make([]int32, cols)
	for c := 0; c < cols; c++ {
		var accum int64
		for r := 0; r < rows; r++ {
			wVal := w.Get(r*cols + c)
			accum += int64(dOut[r]) * wVal
		}
		res := accum >> shift
		dX[c] = int32(res)
	}

	// Accumulate into dW
	for r := 0; r < rows; r++ {
		dVal := int64(dOut[r])
		if dVal == 0 {
			continue
		}
		rowOffset := r * cols
		for c := 0; c < cols; c++ {
			xVal := int64(x[c])
			grad := (dVal * xVal) >> shift
			dW.Add(rowOffset+c, grad)
		}
	}

	return dX, nil
}

// IntegerSoftmax computes numerically stable fixed-point softmax in Q16.
func IntegerSoftmax(logits []int32) []int32 {
	if len(logits) == 0 {
		return nil
	}
	maxL := logits[0]
	for _, l := range logits {
		if l > maxL {
			maxL = l
		}
	}

	// Compute exp(x - max) with float fallback or integer polynomial
	probs := make([]int32, len(logits))
	var sumExp float64
	expVals := make([]float64, len(logits))

	for i, l := range logits {
		diff := float64(l-maxL) / float64(Q16One)
		e := math.Exp(diff)
		expVals[i] = e
		sumExp += e
	}

	if sumExp == 0 {
		sumExp = 1e-9
	}

	for i := range probs {
		p := expVals[i] / sumExp
		probs[i] = int32(p * float64(Q16One))
	}
	return probs
}

// ReLU applies integer rectified linear unit: out = max(0, x).
func ReLU(x []int32) []int32 {
	out := make([]int32, len(x))
	for i, v := range x {
		if v > 0 {
			out[i] = v
		}
	}
	return out
}

// ReLUBackward passes gradient through ReLU: dX = (x > 0 ? dOut : 0).
func ReLUBackward(dOut []int32, x []int32) []int32 {
	dX := make([]int32, len(x))
	for i := range x {
		if x[i] > 0 {
			dX[i] = dOut[i]
		}
	}
	return dX
}
