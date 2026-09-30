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
