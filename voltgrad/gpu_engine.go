package voltgrad

import (
	"fmt"
	"unsafe"
)

// GpuEngine coordinates GPU compute kernels for matrix multiplication, backpropagation, and DBW optimization.
type GpuEngine struct {
	hw                       *HardwareContext
	kernelMatMulFwd          *ComputeKernel
	kernelMatMulBwdDx        *ComputeKernel
	kernelMatMulBwdDw        *ComputeKernel
	kernelMatMulBatchedFwd   *ComputeKernel
	kernelMatMulBatchedBwdDx *ComputeKernel
	kernelMatMulBatchedBwdDw *ComputeKernel
	kernelReLUFwd            *ComputeKernel
	kernelReLUBwd            *ComputeKernel
	kernelLossMSE            *ComputeKernel
	kernelOptimizer          *ComputeKernel
	kernelRepack             *ComputeKernel
}

// NewGpuEngine compiles all neural-network SPIR-V kernels onto the physical GPU.
func NewGpuEngine(hw *HardwareContext) (*GpuEngine, error) {
	var created []*ComputeKernel
	cleanup := func() {
		for _, k := range created {
			if k != nil {
				k.Close()
			}
		}
	}

	create := func(name string, spv []byte, bindings int, pushSize uint32) (*ComputeKernel, error) {
		k, err := NewComputeKernel(hw, spv, bindings, pushSize)
		if err != nil {
			cleanup()
			return nil, fmt.Errorf("failed creating %s kernel: %w", name, err)
		}
		created = append(created, k)
		return k, nil
	}

	kFwd, err := create("matmul_forward", matmulForwardSPV, 4, 20)
	if err != nil {
		return nil, err
	}
	kBwdDx, err := create("matmul_backward_dx", matmulBackwardDxSPV, 3, 16)
	if err != nil {
		return nil, err
	}
	kBwdDw, err := create("matmul_backward_dw", matmulBackwardDwSPV, 4, 16)
	if err != nil {
		return nil, err
	}
	kBatchedFwd, err := create("matmul_batched_forward", matmulBatchedForwardSPV, 4, 24)
	if err != nil {
		return nil, err
	}
	kBatchedBwdDx, err := create("matmul_batched_backward_dx", matmulBatchedBackwardDxSPV, 3, 20)
	if err != nil {
		return nil, err
	}
	kBatchedBwdDw, err := create("matmul_batched_backward_dw", matmulBatchedBackwardDwSPV, 4, 20)
	if err != nil {
		return nil, err
	}
	kReLUFwd, err := create("relu_forward", reluForwardSPV, 2, 4)
	if err != nil {
		return nil, err
	}
	kReLUBwd, err := create("relu_backward", reluBackwardSPV, 3, 4)
	if err != nil {
		return nil, err
	}
	kLossMSE, err := create("loss_mse", lossMseSPV, 4, 8)
	if err != nil {
		return nil, err
	}
	kOpt, err := create("optimizer_step", optimizerStepSPV, 4, 24)
	if err != nil {
		return nil, err
	}
	kRepack, err := create("repack_dbw", repackDbwSPV, 2, 8)
	if err != nil {
		return nil, err
	}

	return &GpuEngine{
		hw:                       hw,
		kernelMatMulFwd:          kFwd,
		kernelMatMulBwdDx:        kBwdDx,
		kernelMatMulBwdDw:        kBwdDw,
		kernelMatMulBatchedFwd:   kBatchedFwd,
		kernelMatMulBatchedBwdDx: kBatchedBwdDx,
		kernelMatMulBatchedBwdDw: kBatchedBwdDw,
		kernelReLUFwd:            kReLUFwd,
		kernelReLUBwd:            kReLUBwd,
		kernelLossMSE:            kLossMSE,
		kernelOptimizer:          kOpt,
		kernelRepack:             kRepack,
	}, nil
}

// MatMulForward executes Y = (W * x) >> shift + bias on the GPU with DBW dequantization in registers.
func (e *GpuEngine) MatMulForward(w *DynamicBitTensor, x []int32, bias *DynamicBitTensor, shift uint) ([]int32, error) {
	if w == nil {
		return nil, fmt.Errorf("weight tensor is nil")
	}
	shape := w.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("weight tensor must be 2D, got %v", shape)
	}
	outDim, inDim := shape[0], shape[1]
	if outDim <= 0 || inDim <= 0 {
		return nil, fmt.Errorf("invalid weight tensor dimensions: %dx%d", outDim, inDim)
	}
	if len(x) != inDim {
		return nil, fmt.Errorf("input vector len %d != inDim %d", len(x), inDim)
	}
	if bias != nil && bias.Len() != outDim {
		return nil, fmt.Errorf("bias len %d != outDim %d", bias.Len(), outDim)
	}

	bufX, err := e.hw.GetPooledBuffer(uint64(len(x) * 4))
	if err != nil {
		return nil, err
	}
	defer bufX.Release()
	bufX.WriteInt32(x)

	words := w.Words()
	bufW, err := e.hw.GetPooledBuffer(uint64(len(words) * 4))
	if err != nil {
		return nil, err
	}
	defer bufW.Release()
	bufW.WriteUint32(words)

	useBias := uint32(0)
	var bufBias *GpuBuffer
	if bias != nil {
		useBias = 1
		biasSlice := bias.ToInt64Slice()
		bufBias, err = e.hw.GetPooledBuffer(uint64(len(biasSlice) * 8))
		if err != nil {
			return nil, err
		}
		defer bufBias.Release()
		bufBias.WriteInt64(biasSlice)
	} else {
		bufBias, err = e.hw.GetPooledBuffer(8)
		if err != nil {
			return nil, err
		}
		defer bufBias.Release()
	}

	bufY, err := e.hw.GetPooledBuffer(uint64(outDim * 4))
	if err != nil {
		return nil, err
	}
	defer bufY.Release()

	push := [5]uint32{
		uint32(inDim),
		uint32(outDim),
		uint32(w.BitWidth()),
		uint32(shift),
		useBias,
	}

	workgroups := uint32((outDim + 63) / 64)
	buffers := []*GpuBuffer{bufX, bufW, bufBias, bufY}
	if err := e.kernelMatMulFwd.Dispatch(buffers, unsafe.Pointer(&push[0]), 20, workgroups); err != nil {
		return nil, fmt.Errorf("GPU MatMulForward dispatch failed: %w", err)
	}

	return bufY.ReadInt32(outDim)
}

// MatMulBatchedForward executes Y[B, outDim] = (X[B, inDim] * W^T) >> shift + bias on the GPU.
func (e *GpuEngine) MatMulBatchedForward(w *DynamicBitTensor, x []int32, batchSize int, bias *DynamicBitTensor, shift uint) ([]int32, error) {
	if w == nil {
		return nil, fmt.Errorf("weight tensor is nil")
	}
	shape := w.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("weight tensor must be 2D, got %v", shape)
	}
	outDim, inDim := shape[0], shape[1]
	if outDim <= 0 || inDim <= 0 {
		return nil, fmt.Errorf("invalid weight tensor dimensions: %dx%d", outDim, inDim)
	}
	if batchSize <= 0 {
		return nil, fmt.Errorf("batchSize must be > 0, got %d", batchSize)
	}
	expectedXLen := batchSize * inDim
	if len(x) != expectedXLen {
		return nil, fmt.Errorf("input matrix len %d != batchSize*inDim %d", len(x), expectedXLen)
	}
	if bias != nil && bias.Len() != outDim {
		return nil, fmt.Errorf("bias len %d != outDim %d", bias.Len(), outDim)
	}

	bufX, err := e.hw.GetPooledBuffer(uint64(len(x) * 4))
	if err != nil {
		return nil, err
	}
	defer bufX.Release()
	bufX.WriteInt32(x)

	words := w.Words()
	bufW, err := e.hw.GetPooledBuffer(uint64(len(words) * 4))
	if err != nil {
		return nil, err
	}
	defer bufW.Release()
	bufW.WriteUint32(words)

	useBias := uint32(0)
	var bufBias *GpuBuffer
	if bias != nil {
		useBias = 1
		biasSlice := bias.ToInt64Slice()
		bufBias, err = e.hw.GetPooledBuffer(uint64(len(biasSlice) * 8))
		if err != nil {
			return nil, err
		}
		defer bufBias.Release()
		bufBias.WriteInt64(biasSlice)
	} else {
		bufBias, err = e.hw.GetPooledBuffer(8)
		if err != nil {
			return nil, err
		}
		defer bufBias.Release()
	}

	totalOut := batchSize * outDim
	bufY, err := e.hw.GetPooledBuffer(uint64(totalOut * 4))
	if err != nil {
		return nil, err
	}
	defer bufY.Release()

	push := [6]uint32{
		uint32(inDim),
		uint32(outDim),
		uint32(batchSize),
		uint32(w.BitWidth()),
		uint32(shift),
		useBias,
	}

	workgroups := uint32((totalOut + 63) / 64)
	buffers := []*GpuBuffer{bufX, bufW, bufBias, bufY}
	if err := e.kernelMatMulBatchedFwd.Dispatch(buffers, unsafe.Pointer(&push[0]), 24, workgroups); err != nil {
		return nil, fmt.Errorf("GPU MatMulBatchedForward dispatch failed: %w", err)
	}

	return bufY.ReadInt32(totalOut)
}

// MatMulBackward executes dX = (W^T * dOut) >> shift and accumulates dW on the GPU.
func (e *GpuEngine) MatMulBackward(dOut []int32, x []int32, w *DynamicBitTensor, gradW *DynamicBitTensor, gradBias *DynamicBitTensor, shift uint) ([]int32, error) {
	if w == nil {
		return nil, fmt.Errorf("weight tensor is nil")
	}
	shape := w.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("weight tensor must be 2D, got %v", shape)
	}
	outDim, inDim := shape[0], shape[1]
	if outDim <= 0 || inDim <= 0 {
		return nil, fmt.Errorf("invalid weight tensor dimensions: %dx%d", outDim, inDim)
	}
	if len(dOut) != outDim {
		return nil, fmt.Errorf("dOut length %d != outDim %d", len(dOut), outDim)
	}
	if len(x) != inDim {
		return nil, fmt.Errorf("x length %d != inDim %d", len(x), inDim)
	}
	if gradW == nil {
		return nil, fmt.Errorf("gradW tensor is nil")
	}
	if gradW.Len() != outDim*inDim {
		return nil, fmt.Errorf("gradW length %d != outDim*inDim %d", gradW.Len(), outDim*inDim)
	}
	if gradBias != nil && gradBias.Len() != outDim {
		return nil, fmt.Errorf("gradBias length %d != outDim %d", gradBias.Len(), outDim)
	}

	// 1. Compute dX on GPU
	bufDOut, err := e.hw.GetPooledBuffer(uint64(len(dOut) * 4))
	if err != nil {
		return nil, err
	}
	defer bufDOut.Release()
	bufDOut.WriteInt32(dOut)

	words := w.Words()
	bufW, err := e.hw.GetPooledBuffer(uint64(len(words) * 4))
	if err != nil {
		return nil, err
	}
	defer bufW.Release()
	bufW.WriteUint32(words)

	bufDX, err := e.hw.GetPooledBuffer(uint64(inDim * 4))
	if err != nil {
		return nil, err
	}
	defer bufDX.Release()

	pushDx := [4]uint32{
		uint32(inDim),
		uint32(outDim),
		uint32(w.BitWidth()),
		uint32(shift),
	}
	wgDx := uint32((inDim + 63) / 64)
	buffersDx := []*GpuBuffer{bufDOut, bufW, bufDX}
	if err := e.kernelMatMulBwdDx.Dispatch(buffersDx, unsafe.Pointer(&pushDx[0]), 16, wgDx); err != nil {
		return nil, fmt.Errorf("GPU MatMulBackward dX dispatch failed: %w", err)
	}
	dX, err := bufDX.ReadInt32(inDim)
	if err != nil {
		return nil, err
	}

	// 2. Accumulate dW and dBias on GPU
	bufX, err := e.hw.GetPooledBuffer(uint64(len(x) * 4))
	if err != nil {
		return nil, err
	}
	defer bufX.Release()
	if err := bufX.WriteInt32(x); err != nil {
		return nil, err
	}

	gradWSlice := gradW.ToInt64Slice()
	bufGradW, err := e.hw.GetPooledBuffer(uint64(len(gradWSlice) * 8))
	if err != nil {
		return nil, err
	}
	defer bufGradW.Release()
	if err := bufGradW.WriteInt64(gradWSlice); err != nil {
		return nil, err
	}

	useBias := uint32(0)
	var bufGradBias *GpuBuffer
	var gradBiasSlice []int64
	if gradBias != nil {
		useBias = 1
		gradBiasSlice = gradBias.ToInt64Slice()
		bufGradBias, err = e.hw.GetPooledBuffer(uint64(len(gradBiasSlice) * 8))
		if err != nil {
			return nil, err
		}
		defer bufGradBias.Release()
		if err := bufGradBias.WriteInt64(gradBiasSlice); err != nil {
			return nil, err
		}
	} else {
		bufGradBias, err = e.hw.GetPooledBuffer(8)
		if err != nil {
			return nil, err
		}
		defer bufGradBias.Release()
	}

	pushDw := [4]uint32{
		uint32(inDim),
		uint32(outDim),
		uint32(shift),
		useBias,
	}
	totalWeights := outDim * inDim
	wgDw := uint32((totalWeights + 63) / 64)
	buffersDw := []*GpuBuffer{bufDOut, bufX, bufGradW, bufGradBias}
	if err := e.kernelMatMulBwdDw.Dispatch(buffersDw, unsafe.Pointer(&pushDw[0]), 16, wgDw); err != nil {
		return nil, fmt.Errorf("GPU MatMulBackward dW dispatch failed: %w", err)
	}

	readGradW, err := bufGradW.ReadInt64(len(gradWSlice))
	if err != nil {
		return nil, err
	}
	gradW.FromInt64Slice(readGradW)
	if gradBias != nil {
		readGradBias, err := bufGradBias.ReadInt64(len(gradBiasSlice))
		if err != nil {
			return nil, err
		}
		gradBias.FromInt64Slice(readGradBias)
	}

	return dX, nil
}

// MatMulBatchedBackward executes dX = dOut * W >> shift and accumulates dW across the batch on the GPU.
func (e *GpuEngine) MatMulBatchedBackward(dOut []int32, x []int32, batchSize int, w *DynamicBitTensor, gradW *DynamicBitTensor, gradBias *DynamicBitTensor, shift uint) ([]int32, error) {
	if w == nil {
		return nil, fmt.Errorf("weight tensor is nil")
	}
	shape := w.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("weight tensor must be 2D, got %v", shape)
	}
	outDim, inDim := shape[0], shape[1]
	if outDim <= 0 || inDim <= 0 {
		return nil, fmt.Errorf("invalid weight tensor dimensions: %dx%d", outDim, inDim)
	}
	if batchSize <= 0 {
		return nil, fmt.Errorf("batchSize must be > 0, got %d", batchSize)
	}
	expectedDOutLen := batchSize * outDim
	if len(dOut) != expectedDOutLen {
		return nil, fmt.Errorf("dOut length %d != batchSize*outDim %d", len(dOut), expectedDOutLen)
	}
	expectedXLen := batchSize * inDim
	if len(x) != expectedXLen {
		return nil, fmt.Errorf("x length %d != batchSize*inDim %d", len(x), expectedXLen)
	}
	if gradW == nil {
		return nil, fmt.Errorf("gradW tensor is nil")
	}
	if gradW.Len() != outDim*inDim {
		return nil, fmt.Errorf("gradW length %d != outDim*inDim %d", gradW.Len(), outDim*inDim)
	}
	if gradBias != nil && gradBias.Len() != outDim {
		return nil, fmt.Errorf("gradBias length %d != outDim %d", gradBias.Len(), outDim)
	}

	// 1. Compute dX on GPU across all batch items: size batchSize * inDim
	bufDOut, err := e.hw.GetPooledBuffer(uint64(len(dOut) * 4))
	if err != nil {
		return nil, err
	}
	defer bufDOut.Release()
	bufDOut.WriteInt32(dOut)

	words := w.Words()
	bufW, err := e.hw.GetPooledBuffer(uint64(len(words) * 4))
	if err != nil {
		return nil, err
	}
	defer bufW.Release()
	bufW.WriteUint32(words)

	totalDX := batchSize * inDim
	bufDX, err := e.hw.GetPooledBuffer(uint64(totalDX * 4))
	if err != nil {
		return nil, err
	}
	defer bufDX.Release()

	pushDx := [5]uint32{
		uint32(inDim),
		uint32(outDim),
		uint32(batchSize),
		uint32(w.BitWidth()),
		uint32(shift),
	}
	wgDx := uint32((totalDX + 63) / 64)
	buffersDx := []*GpuBuffer{bufDOut, bufW, bufDX}
	if err := e.kernelMatMulBatchedBwdDx.Dispatch(buffersDx, unsafe.Pointer(&pushDx[0]), 20, wgDx); err != nil {
		return nil, fmt.Errorf("GPU MatMulBatchedBackward dX dispatch failed: %w", err)
	}
	dX, err := bufDX.ReadInt32(totalDX)
	if err != nil {
		return nil, err
	}

	// 2. Accumulate dW and dBias on GPU
	bufX, err := e.hw.GetPooledBuffer(uint64(len(x) * 4))
	if err != nil {
		return nil, err
	}
	defer bufX.Release()
	if err := bufX.WriteInt32(x); err != nil {
		return nil, err
	}

	gradWSlice := gradW.ToInt64Slice()
	bufGradW, err := e.hw.GetPooledBuffer(uint64(len(gradWSlice) * 8))
	if err != nil {
		return nil, err
	}
	defer bufGradW.Release()
	if err := bufGradW.WriteInt64(gradWSlice); err != nil {
		return nil, err
	}

	useBias := uint32(0)
	var bufGradBias *GpuBuffer
	var gradBiasSlice []int64
	if gradBias != nil {
		useBias = 1
		gradBiasSlice = gradBias.ToInt64Slice()
		bufGradBias, err = e.hw.GetPooledBuffer(uint64(len(gradBiasSlice) * 8))
		if err != nil {
			return nil, err
		}
		defer bufGradBias.Release()
		if err := bufGradBias.WriteInt64(gradBiasSlice); err != nil {
			return nil, err
		}
	} else {
		bufGradBias, err = e.hw.GetPooledBuffer(8)
		if err != nil {
			return nil, err
		}
		defer bufGradBias.Release()
	}

	pushDw := [5]uint32{
		uint32(inDim),
		uint32(outDim),
		uint32(batchSize),
		uint32(shift),
		useBias,
	}
	totalWeights := outDim * inDim
	wgDw := uint32((totalWeights + 63) / 64)
	buffersDw := []*GpuBuffer{bufDOut, bufX, bufGradW, bufGradBias}
	if err := e.kernelMatMulBatchedBwdDw.Dispatch(buffersDw, unsafe.Pointer(&pushDw[0]), 20, wgDw); err != nil {
		return nil, fmt.Errorf("GPU MatMulBatchedBackward dW dispatch failed: %w", err)
	}

	readGradW, err := bufGradW.ReadInt64(len(gradWSlice))
	if err != nil {
		return nil, err
	}
	gradW.FromInt64Slice(readGradW)
	if gradBias != nil {
		readGradBias, err := bufGradBias.ReadInt64(len(gradBiasSlice))
		if err != nil {
			return nil, err
		}
		gradBias.FromInt64Slice(readGradBias)
	}

	return dX, nil
}

// ReLUForward applies element-wise ReLU: Y[i] = max(0, X[i]) on the GPU.
func (e *GpuEngine) ReLUForward(x []int32) ([]int32, error) {
	n := len(x)
	if n == 0 {
		return nil, fmt.Errorf("input slice is empty")
	}

	bufX, err := e.hw.GetPooledBuffer(uint64(n * 4))
	if err != nil {
		return nil, err
	}
	defer bufX.Release()
	if err := bufX.WriteInt32(x); err != nil {
		return nil, err
	}

	bufY, err := e.hw.GetPooledBuffer(uint64(n * 4))
	if err != nil {
		return nil, err
	}
	defer bufY.Release()

	push := [1]uint32{uint32(n)}
	wg := uint32((n + 63) / 64)
	buffers := []*GpuBuffer{bufX, bufY}
	if err := e.kernelReLUFwd.Dispatch(buffers, unsafe.Pointer(&push[0]), 4, wg); err != nil {
		return nil, fmt.Errorf("GPU ReLUForward dispatch failed: %w", err)
	}

	return bufY.ReadInt32(n)
}

// ReLUBackward computes element-wise ReLU gradient: dX[i] = (X[i] > 0 ? dOut[i] : 0) on the GPU.
func (e *GpuEngine) ReLUBackward(dOut []int32, x []int32) ([]int32, error) {
	n := len(x)
	if n == 0 {
		return nil, fmt.Errorf("input slice is empty")
	}
	if len(dOut) != n {
		return nil, fmt.Errorf("dOut len %d != x len %d", len(dOut), n)
	}

	bufDOut, err := e.hw.GetPooledBuffer(uint64(n * 4))
	if err != nil {
		return nil, err
	}
	defer bufDOut.Release()
	if err := bufDOut.WriteInt32(dOut); err != nil {
		return nil, err
	}

	bufX, err := e.hw.GetPooledBuffer(uint64(n * 4))
	if err != nil {
		return nil, err
	}
	defer bufX.Release()
	if err := bufX.WriteInt32(x); err != nil {
		return nil, err
	}

	bufDX, err := e.hw.GetPooledBuffer(uint64(n * 4))
	if err != nil {
		return nil, err
	}
	defer bufDX.Release()

	push := [1]uint32{uint32(n)}
	wg := uint32((n + 63) / 64)
	buffers := []*GpuBuffer{bufDOut, bufX, bufDX}
	if err := e.kernelReLUBwd.Dispatch(buffers, unsafe.Pointer(&push[0]), 4, wg); err != nil {
		return nil, fmt.Errorf("GPU ReLUBackward dispatch failed: %w", err)
	}

	return bufDX.ReadInt32(n)
}

// MSELoss computes mean squared error loss and error gradient on the GPU.
func (e *GpuEngine) MSELoss(preds, targets []int32, shift uint) (int32, []int32, error) {
	n := len(preds)
	if n == 0 {
		return 0, nil, fmt.Errorf("prediction slice is empty")
	}
	if len(targets) != n {
		return 0, nil, fmt.Errorf("targets len %d != preds len %d", len(targets), n)
	}

	bufPreds, err := e.hw.GetPooledBuffer(uint64(n * 4))
	if err != nil {
		return 0, nil, err
	}
	defer bufPreds.Release()
	if err := bufPreds.WriteInt32(preds); err != nil {
		return 0, nil, err
	}

	bufTargets, err := e.hw.GetPooledBuffer(uint64(n * 4))
	if err != nil {
		return 0, nil, err
	}
	defer bufTargets.Release()
	if err := bufTargets.WriteInt32(targets); err != nil {
		return 0, nil, err
	}

	bufGrad, err := e.hw.GetPooledBuffer(uint64(n * 4))
	if err != nil {
		return 0, nil, err
	}
	defer bufGrad.Release()

	bufLoss, err := e.hw.GetPooledBuffer(uint64(n * 8))
	if err != nil {
		return 0, nil, err
	}
	defer bufLoss.Release()

	push := [2]uint32{uint32(n), uint32(shift)}
	wg := uint32((n + 63) / 64)
	buffers := []*GpuBuffer{bufPreds, bufTargets, bufGrad, bufLoss}
	if err := e.kernelLossMSE.Dispatch(buffers, unsafe.Pointer(&push[0]), 8, wg); err != nil {
		return 0, nil, fmt.Errorf("GPU MSELoss dispatch failed: %w", err)
	}

	grad, err := bufGrad.ReadInt32(n)
	if err != nil {
		return 0, nil, err
	}
	losses, err := bufLoss.ReadInt64(n)
	if err != nil {
		return 0, nil, err
	}
	var totalLoss int64
	for _, l := range losses {
		totalLoss += l
	}
	meanLoss := int32(totalLoss / int64(n))

	return meanLoss, grad, nil
}

// DeviceReport queries and returns hardware capabilities, driver versions, and memory heaps.
func (e *GpuEngine) DeviceReport() DeviceReport {
	return e.hw.DeviceReport()
}

// OptimizerStep performs the momentum-based fixed-point parameter update and DBW overflow check entirely on GPU.
func (e *GpuEngine) OptimizerStep(p *DynamicBitTensor, g *DynamicBitTensor, v *DynamicBitTensor, config OptimizerConfig) (expandedBitWidth int, err error) {
	if p == nil || g == nil || v == nil {
		return 0, fmt.Errorf("optimizer tensors cannot be nil")
	}
	count := p.Len()
	if count == 0 {
		return p.BitWidth(), nil
	}
	if g.Len() != count || v.Len() != count {
		return 0, fmt.Errorf("tensor length mismatch: p=%d, g=%d, v=%d", count, g.Len(), v.Len())
	}

	wSlice := p.ToInt64Slice()
	gSlice := g.ToInt64Slice()
	vSlice := v.ToInt64Slice()

	bufWeights, err := e.hw.GetPooledBuffer(uint64(count * 8))
	if err != nil {
		return p.BitWidth(), err
	}
	defer bufWeights.Release()
	if err := bufWeights.WriteInt64(wSlice); err != nil {
		return p.BitWidth(), err
	}

	bufGrads, err := e.hw.GetPooledBuffer(uint64(count * 8))
	if err != nil {
		return p.BitWidth(), err
	}
	defer bufGrads.Release()
	if err := bufGrads.WriteInt64(gSlice); err != nil {
		return p.BitWidth(), err
	}

	bufVels, err := e.hw.GetPooledBuffer(uint64(count * 8))
	if err != nil {
		return p.BitWidth(), err
	}
	defer bufVels.Release()
	if err := bufVels.WriteInt64(vSlice); err != nil {
		return p.BitWidth(), err
	}

	// meta[0] initialized to current bit width
	bufMeta, err := e.hw.GetPooledBuffer(4)
	if err != nil {
		return p.BitWidth(), err
	}
	defer bufMeta.Release()
	if err := bufMeta.WriteUint32([]uint32{uint32(p.BitWidth())}); err != nil {
		return p.BitWidth(), err
	}

	optType := uint32(config.Type)
	pushOpt := [6]int32{
		int32(count),
		config.LearningRateQ16,
		config.MomentumBeta,
		int32(config.FractionalBits),
		int32(optType),
		config.WeightDecayQ16,
	}

	wg := uint32((count + 63) / 64)
	buffersOpt := []*GpuBuffer{bufWeights, bufGrads, bufVels, bufMeta}
	if err := e.kernelOptimizer.Dispatch(buffersOpt, unsafe.Pointer(&pushOpt[0]), 24, wg); err != nil {
		return p.BitWidth(), fmt.Errorf("GPU OptimizerStep dispatch failed: %w", err)
	}

	// Read updated state back from GPU into Go memory
	newV, err := bufVels.ReadInt64(count)
	if err != nil {
		return p.BitWidth(), err
	}
	v.FromInt64Slice(newV)

	newWeights, err := bufWeights.ReadInt64(count)
	if err != nil {
		return p.BitWidth(), err
	}
	p.FromInt64Slice(newWeights)

	// Check if DBW auto-widening occurred on the GPU
	metaSlice, err := bufMeta.ReadUint32(1)
	if err != nil {
		return p.BitWidth(), err
	}
	targetBitWidth := int(metaSlice[0])
	if targetBitWidth < p.BitWidth() {
		targetBitWidth = p.BitWidth()
	}

	if targetBitWidth == p.BitWidth() {
		return p.BitWidth(), nil
	}

	// 2. Perform DBW GPU Repacking into target bit width
	totalBits := count * targetBitWidth
	numWords := (totalBits + 31) / 32
	if numWords == 0 {
		numWords = 1
	}
	bufPacked, err := e.hw.GetPooledBuffer(uint64(numWords * 4))
	if err != nil {
		return p.BitWidth(), err
	}
	defer bufPacked.Release()
	bufPacked.Clear()

	pushRepack := [2]uint32{
		uint32(count),
		uint32(targetBitWidth),
	}
	buffersRepack := []*GpuBuffer{bufWeights, bufPacked}
	if err := e.kernelRepack.Dispatch(buffersRepack, unsafe.Pointer(&pushRepack[0]), 8, wg); err != nil {
		return p.BitWidth(), fmt.Errorf("GPU Repack dispatch failed: %w", err)
	}

	// Update tensor packed representation
	packedWords, err := bufPacked.ReadUint32(numWords)
	if err != nil {
		return p.BitWidth(), err
	}
	p.SetWords(packedWords, targetBitWidth)

	return targetBitWidth, nil
}

// MatMulForwardResident executes matrix multiplication using a persistent GPU weight buffer.
func (e *GpuEngine) MatMulForwardResident(bufW, bufBias *GpuBuffer, x []int32, outDim, inDim, bitWidth int, shift uint) ([]int32, error) {
	if bufW == nil {
		return nil, fmt.Errorf("persistent weight buffer is nil")
	}
	if outDim <= 0 || inDim <= 0 {
		return nil, fmt.Errorf("invalid dimensions: %dx%d", outDim, inDim)
	}
	if len(x) != inDim {
		return nil, fmt.Errorf("input vector len %d != inDim %d", len(x), inDim)
	}

	bufX, err := e.hw.GetPooledBuffer(uint64(len(x) * 4))
	if err != nil {
		return nil, err
	}
	defer bufX.Release()
	if err := bufX.WriteInt32(x); err != nil {
		return nil, err
	}

	useBias := uint32(0)
	var bBias *GpuBuffer
	if bufBias != nil {
		useBias = 1
		bBias = bufBias
	} else {
		bBias, err = e.hw.GetPooledBuffer(8)
		if err != nil {
			return nil, err
		}
		defer bBias.Release()
	}

	bufY, err := e.hw.GetPooledBuffer(uint64(outDim * 4))
	if err != nil {
		return nil, err
	}
	defer bufY.Release()

	push := [5]uint32{
		uint32(inDim),
		uint32(outDim),
		uint32(bitWidth),
		uint32(shift),
		useBias,
	}

	workgroups := uint32((outDim + 63) / 64)
	buffers := []*GpuBuffer{bufX, bufW, bBias, bufY}
	if err := e.kernelMatMulFwd.Dispatch(buffers, unsafe.Pointer(&push[0]), 20, workgroups); err != nil {
		return nil, fmt.Errorf("GPU MatMulForwardResident dispatch failed: %w", err)
	}

	return bufY.ReadInt32(outDim)
}

// MatMulBatchedForwardResident executes batched GEMM using a persistent GPU weight buffer.
func (e *GpuEngine) MatMulBatchedForwardResident(bufW, bufBias *GpuBuffer, x []int32, batchSize, outDim, inDim, bitWidth int, shift uint) ([]int32, error) {
	if bufW == nil {
		return nil, fmt.Errorf("persistent weight buffer is nil")
	}
	if batchSize <= 0 || outDim <= 0 || inDim <= 0 {
		return nil, fmt.Errorf("invalid batched dimensions: batch=%d, outDim=%d, inDim=%d", batchSize, outDim, inDim)
	}
	if len(x) != batchSize*inDim {
		return nil, fmt.Errorf("input batch length %d != batchSize*inDim (%d)", len(x), batchSize*inDim)
	}

	bufX, err := e.hw.GetPooledBuffer(uint64(len(x) * 4))
	if err != nil {
		return nil, err
	}
	defer bufX.Release()
	if err := bufX.WriteInt32(x); err != nil {
		return nil, err
	}

	useBias := uint32(0)
	var bBias *GpuBuffer
	if bufBias != nil {
		useBias = 1
		bBias = bufBias
	} else {
		bBias, err = e.hw.GetPooledBuffer(8)
		if err != nil {
			return nil, err
		}
		defer bBias.Release()
	}

	totalOut := batchSize * outDim
	bufY, err := e.hw.GetPooledBuffer(uint64(totalOut * 4))
	if err != nil {
		return nil, err
	}
	defer bufY.Release()

	push := [6]uint32{
		uint32(inDim),
		uint32(outDim),
		uint32(batchSize),
		uint32(bitWidth),
		uint32(shift),
		useBias,
	}

	workgroups := uint32((totalOut + 63) / 64)
	buffers := []*GpuBuffer{bufX, bufW, bBias, bufY}
	if err := e.kernelMatMulBatchedFwd.Dispatch(buffers, unsafe.Pointer(&push[0]), 24, workgroups); err != nil {
		return nil, fmt.Errorf("GPU MatMulBatchedForwardResident dispatch failed: %w", err)
	}

	return bufY.ReadInt32(totalOut)
}

// MatMulBackwardResident executes backpropagation using persistent GPU weight and gradient buffers.
func (e *GpuEngine) MatMulBackwardResident(dOut, x []int32, outDim, inDim, bitWidth int, shift uint, bufW, bufGradW, bufGradBias *GpuBuffer) ([]int32, error) {
	if bufW == nil || bufGradW == nil {
		return nil, fmt.Errorf("persistent weight and gradient buffers cannot be nil")
	}
	if len(dOut) != outDim {
		return nil, fmt.Errorf("dOut length %d != outDim %d", len(dOut), outDim)
	}
	if len(x) != inDim {
		return nil, fmt.Errorf("x length %d != inDim %d", len(x), inDim)
	}

	// 1. Compute dX = (dOut * W) >> shift
	bufDOut, err := e.hw.GetPooledBuffer(uint64(len(dOut) * 4))
	if err != nil {
		return nil, err
	}
	defer bufDOut.Release()
	if err := bufDOut.WriteInt32(dOut); err != nil {
		return nil, err
	}

	bufDX, err := e.hw.GetPooledBuffer(uint64(inDim * 4))
	if err != nil {
		return nil, err
	}
	defer bufDX.Release()

	pushDX := [4]uint32{
		uint32(inDim),
		uint32(outDim),
		uint32(bitWidth),
		uint32(shift),
	}

	wgDX := uint32((inDim + 63) / 64)
	buffersDX := []*GpuBuffer{bufDOut, bufW, bufDX}
	if err := e.kernelMatMulBwdDx.Dispatch(buffersDX, unsafe.Pointer(&pushDX[0]), 16, wgDX); err != nil {
		return nil, fmt.Errorf("GPU MatMulBackwardResident dX dispatch failed: %w", err)
	}

	// 2. Accumulate parameter gradients into persistent GPU buffers
	bufX, err := e.hw.GetPooledBuffer(uint64(len(x) * 4))
	if err != nil {
		return nil, err
	}
	defer bufX.Release()
	if err := bufX.WriteInt32(x); err != nil {
		return nil, err
	}

	useBias := uint32(0)
	var bGradBias *GpuBuffer
	if bufGradBias != nil {
		useBias = 1
		bGradBias = bufGradBias
	} else {
		bGradBias, err = e.hw.GetPooledBuffer(8)
		if err != nil {
			return nil, err
		}
		defer bGradBias.Release()
	}

	pushDW := [4]uint32{
		uint32(inDim),
		uint32(outDim),
		uint32(shift),
		useBias,
	}

	totalDW := outDim * inDim
	wgDW := uint32((totalDW + 63) / 64)
	buffersDW := []*GpuBuffer{bufDOut, bufX, bufGradW, bGradBias}
	if err := e.kernelMatMulBwdDw.Dispatch(buffersDW, unsafe.Pointer(&pushDW[0]), 16, wgDW); err != nil {
		return nil, fmt.Errorf("GPU MatMulBackwardResident dW dispatch failed: %w", err)
	}

	return bufDX.ReadInt32(inDim)
}

// MatMulBatchedBackwardResident executes batched backpropagation using persistent GPU weight and gradient buffers.
func (e *GpuEngine) MatMulBatchedBackwardResident(dOut, x []int32, batchSize, outDim, inDim, bitWidth int, shift uint, bufW, bufGradW, bufGradBias *GpuBuffer) ([]int32, error) {
	if bufW == nil || bufGradW == nil {
		return nil, fmt.Errorf("persistent weight and gradient buffers cannot be nil")
	}
	if batchSize <= 0 || outDim <= 0 || inDim <= 0 {
		return nil, fmt.Errorf("invalid batched backward dimensions")
	}
	if len(dOut) != batchSize*outDim {
		return nil, fmt.Errorf("dOut length %d != batchSize*outDim (%d)", len(dOut), batchSize*outDim)
	}
	if len(x) != batchSize*inDim {
		return nil, fmt.Errorf("x length %d != batchSize*inDim (%d)", len(x), batchSize*inDim)
	}

	// 1. Batched dX: dX[B, InDim] = (dOut[B, OutDim] * W[OutDim, InDim]) >> shift
	bufDOut, err := e.hw.GetPooledBuffer(uint64(len(dOut) * 4))
	if err != nil {
		return nil, err
	}
	defer bufDOut.Release()
	if err := bufDOut.WriteInt32(dOut); err != nil {
		return nil, err
	}

	totalDX := batchSize * inDim
	bufDX, err := e.hw.GetPooledBuffer(uint64(totalDX * 4))
	if err != nil {
		return nil, err
	}
	defer bufDX.Release()

	pushDX := [5]uint32{
		uint32(inDim),
		uint32(outDim),
		uint32(batchSize),
		uint32(bitWidth),
		uint32(shift),
	}

	wgDX := uint32((totalDX + 63) / 64)
	buffersDX := []*GpuBuffer{bufDOut, bufW, bufDX}
	if err := e.kernelMatMulBatchedBwdDx.Dispatch(buffersDX, unsafe.Pointer(&pushDX[0]), 20, wgDX); err != nil {
		return nil, fmt.Errorf("GPU MatMulBatchedBackwardResident dX dispatch failed: %w", err)
	}

	// 2. Batched dW: accumulate dOut^T * X into resident bufGradW and bufGradBias
	bufX, err := e.hw.GetPooledBuffer(uint64(len(x) * 4))
	if err != nil {
		return nil, err
	}
	defer bufX.Release()
	if err := bufX.WriteInt32(x); err != nil {
		return nil, err
	}

	useBias := uint32(0)
	var bGradBias *GpuBuffer
	if bufGradBias != nil {
		useBias = 1
		bGradBias = bufGradBias
	} else {
		bGradBias, err = e.hw.GetPooledBuffer(8)
		if err != nil {
			return nil, err
		}
		defer bGradBias.Release()
	}

	pushDW := [5]uint32{
		uint32(inDim),
		uint32(outDim),
		uint32(batchSize),
		uint32(shift),
		useBias,
	}

	totalDW := outDim * inDim
	wgDW := uint32((totalDW + 63) / 64)
	buffersDW := []*GpuBuffer{bufDOut, bufX, bufGradW, bGradBias}
	if err := e.kernelMatMulBatchedBwdDw.Dispatch(buffersDW, unsafe.Pointer(&pushDW[0]), 20, wgDW); err != nil {
		return nil, fmt.Errorf("GPU MatMulBatchedBackwardResident dW dispatch failed: %w", err)
	}

	return bufDX.ReadInt32(totalDX)
}

// OptimizerStepResident updates parameters directly in persistent GPU buffers with zero host copies during the step.
func (e *GpuEngine) OptimizerStepResident(bufWeights, bufGrads, bufVels, bufPacked *GpuBuffer, count int, currentBitWidth int, config OptimizerConfig) (int, error) {
	if count <= 0 {
		return currentBitWidth, nil
	}
	if bufWeights == nil || bufGrads == nil || bufVels == nil || bufPacked == nil {
		return currentBitWidth, fmt.Errorf("resident optimizer buffers cannot be nil")
	}

	bufMeta, err := e.hw.GetPooledBuffer(4)
	if err != nil {
		return currentBitWidth, err
	}
	defer bufMeta.Release()
	bufMeta.Clear()

	optType := uint32(0)
	switch config.Type {
	case OptimizerSGD:
		optType = 0
	case OptimizerSGDMomentum:
		optType = 1
	case OptimizerAdamQ:
		optType = 2
	}

	pushOpt := [6]int32{
		int32(count),
		config.LearningRateQ16,
		config.MomentumBeta,
		int32(config.FractionalBits),
		int32(optType),
		config.WeightDecayQ16,
	}

	wg := uint32((count + 63) / 64)
	buffersOpt := []*GpuBuffer{bufWeights, bufGrads, bufVels, bufMeta}
	if err := e.kernelOptimizer.Dispatch(buffersOpt, unsafe.Pointer(&pushOpt[0]), 24, wg); err != nil {
		return currentBitWidth, fmt.Errorf("GPU resident OptimizerStep dispatch failed: %w", err)
	}

	// Check if DBW auto-widening occurred on the GPU
	metaSlice, err := bufMeta.ReadUint32(1)
	if err != nil {
		return currentBitWidth, err
	}
	targetBitWidth := int(metaSlice[0])
	if targetBitWidth < currentBitWidth {
		targetBitWidth = currentBitWidth
	}
	if targetBitWidth < 2 {
		targetBitWidth = 2
	}
	if targetBitWidth > 64 {
		targetBitWidth = 64
	}

	// Perform DBW GPU Repacking directly on the GPU into bufPacked
	totalBits := count * targetBitWidth
	numWords := (totalBits + 31) / 32
	if numWords == 0 {
		numWords = 1
	}
	if err := bufPacked.SetActiveSize(uint64(numWords * 4)); err != nil {
		return currentBitWidth, err
	}
	bufPacked.Clear()

	pushRepack := [2]uint32{
		uint32(count),
		uint32(targetBitWidth),
	}
	buffersRepack := []*GpuBuffer{bufWeights, bufPacked}
	if err := e.kernelRepack.Dispatch(buffersRepack, unsafe.Pointer(&pushRepack[0]), 8, wg); err != nil {
		return currentBitWidth, fmt.Errorf("GPU resident Repack dispatch failed: %w", err)
	}

	return targetBitWidth, nil
}

// Close terminates all compute kernels.
func (e *GpuEngine) Close() {
	if e.kernelMatMulFwd != nil {
		e.kernelMatMulFwd.Close()
		e.kernelMatMulFwd = nil
	}
	if e.kernelMatMulBwdDx != nil {
		e.kernelMatMulBwdDx.Close()
		e.kernelMatMulBwdDx = nil
	}
	if e.kernelMatMulBwdDw != nil {
		e.kernelMatMulBwdDw.Close()
		e.kernelMatMulBwdDw = nil
	}
	if e.kernelMatMulBatchedFwd != nil {
		e.kernelMatMulBatchedFwd.Close()
		e.kernelMatMulBatchedFwd = nil
	}
	if e.kernelMatMulBatchedBwdDx != nil {
		e.kernelMatMulBatchedBwdDx.Close()
		e.kernelMatMulBatchedBwdDx = nil
	}
	if e.kernelMatMulBatchedBwdDw != nil {
		e.kernelMatMulBatchedBwdDw.Close()
		e.kernelMatMulBatchedBwdDw = nil
	}
	if e.kernelReLUFwd != nil {
		e.kernelReLUFwd.Close()
		e.kernelReLUFwd = nil
	}
	if e.kernelReLUBwd != nil {
		e.kernelReLUBwd.Close()
		e.kernelReLUBwd = nil
	}
	if e.kernelLossMSE != nil {
		e.kernelLossMSE.Close()
		e.kernelLossMSE = nil
	}
	if e.kernelOptimizer != nil {
		e.kernelOptimizer.Close()
		e.kernelOptimizer = nil
	}
	if e.kernelRepack != nil {
		e.kernelRepack.Close()
		e.kernelRepack = nil
	}
}
