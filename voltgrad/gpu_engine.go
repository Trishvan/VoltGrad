package voltgrad

import (
	"fmt"
	"unsafe"
)

// GpuEngine coordinates GPU compute kernels for matrix multiplication, backpropagation, and DBW optimization.
type GpuEngine struct {
	hw                 *HardwareContext
	kernelMatMulFwd    *ComputeKernel
	kernelMatMulBwdDx  *ComputeKernel
	kernelMatMulBwdDw  *ComputeKernel
	kernelOptimizer    *ComputeKernel
	kernelRepack       *ComputeKernel
}

// NewGpuEngine compiles all neural-network SPIR-V kernels onto the physical GPU.
func NewGpuEngine(hw *HardwareContext) (*GpuEngine, error) {
	// 1. MatMul Forward (4 bindings: InX, InW, InBias, OutY; 20 bytes push constants)
	kFwd, err := NewComputeKernel(hw, matmulForwardSPV, 4, 20)
	if err != nil {
		return nil, fmt.Errorf("failed creating matmul_forward kernel: %w", err)
	}

	// 2. MatMul Backward dX (3 bindings: InDOut, InW, OutDX; 16 bytes push constants)
	kBwdDx, err := NewComputeKernel(hw, matmulBackwardDxSPV, 3, 16)
	if err != nil {
		kFwd.Close()
		return nil, fmt.Errorf("failed creating matmul_backward_dx kernel: %w", err)
	}

	// 3. MatMul Backward dW (4 bindings: InDOut, InX, GradW, GradBias; 16 bytes push constants)
	kBwdDw, err := NewComputeKernel(hw, matmulBackwardDwSPV, 4, 16)
	if err != nil {
		kFwd.Close()
		kBwdDx.Close()
		return nil, fmt.Errorf("failed creating matmul_backward_dw kernel: %w", err)
	}

	// 4. Optimizer Step (4 bindings: Weights, Grads, Velocities, Meta; 24 bytes push constants)
	kOpt, err := NewComputeKernel(hw, optimizerStepSPV, 4, 24)
	if err != nil {
		kFwd.Close()
		kBwdDx.Close()
		kBwdDw.Close()
		return nil, fmt.Errorf("failed creating optimizer_step kernel: %w", err)
	}

	// 5. DBW Repack (2 bindings: Weights, OutPacked; 8 bytes push constants)
	kRepack, err := NewComputeKernel(hw, repackDbwSPV, 2, 8)
	if err != nil {
		kFwd.Close()
		kBwdDx.Close()
		kBwdDw.Close()
		kOpt.Close()
		return nil, fmt.Errorf("failed creating repack_dbw kernel: %w", err)
	}

	return &GpuEngine{
		hw:                hw,
		kernelMatMulFwd:   kFwd,
		kernelMatMulBwdDx: kBwdDx,
		kernelMatMulBwdDw: kBwdDw,
		kernelOptimizer:   kOpt,
		kernelRepack:      kRepack,
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

	bufX, err := e.hw.CreateHostCoherentBuffer(uint64(len(x) * 4))
	if err != nil {
		return nil, err
	}
	defer bufX.Free()
	bufX.WriteInt32(x)

	words := w.Words()
	bufW, err := e.hw.CreateHostCoherentBuffer(uint64(len(words) * 4))
	if err != nil {
		return nil, err
	}
	defer bufW.Free()
	bufW.WriteUint32(words)

	useBias := uint32(0)
	var bufBias *GpuBuffer
	if bias != nil {
		useBias = 1
		biasWords := bias.ToInt64Slice()
		bufBias, err = e.hw.CreateHostCoherentBuffer(uint64(len(biasWords) * 8))
		if err != nil {
			return nil, err
		}
		defer bufBias.Free()
		bufBias.WriteInt64(biasWords)
	} else {
		bufBias, err = e.hw.CreateHostCoherentBuffer(8)
		if err != nil {
			return nil, err
		}
		defer bufBias.Free()
	}

	bufY, err := e.hw.CreateHostCoherentBuffer(uint64(outDim * 4))
	if err != nil {
		return nil, err
	}
	defer bufY.Free()

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

	out := bufY.ReadInt32(outDim)
	return out, nil
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
	bufDOut, err := e.hw.CreateHostCoherentBuffer(uint64(len(dOut) * 4))
	if err != nil {
		return nil, err
	}
	defer bufDOut.Free()
	bufDOut.WriteInt32(dOut)

	words := w.Words()
	bufW, err := e.hw.CreateHostCoherentBuffer(uint64(len(words) * 4))
	if err != nil {
		return nil, err
	}
	defer bufW.Free()
	bufW.WriteUint32(words)

	bufDX, err := e.hw.CreateHostCoherentBuffer(uint64(inDim * 4))
	if err != nil {
		return nil, err
	}
	defer bufDX.Free()

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
	dX := bufDX.ReadInt32(inDim)

	// 2. Accumulate dW and dBias on GPU
	bufX, err := e.hw.CreateHostCoherentBuffer(uint64(len(x) * 4))
	if err != nil {
		return nil, err
	}
	defer bufX.Free()
	bufX.WriteInt32(x)

	gradWSlice := gradW.ToInt64Slice()
	bufGradW, err := e.hw.CreateHostCoherentBuffer(uint64(len(gradWSlice) * 8))
	if err != nil {
		return nil, err
	}
	defer bufGradW.Free()
	bufGradW.WriteInt64(gradWSlice)

	useBias := uint32(0)
	var bufGradBias *GpuBuffer
	var gradBiasSlice []int64
	if gradBias != nil {
		useBias = 1
		gradBiasSlice = gradBias.ToInt64Slice()
		bufGradBias, err = e.hw.CreateHostCoherentBuffer(uint64(len(gradBiasSlice) * 8))
		if err != nil {
			return nil, err
		}
		defer bufGradBias.Free()
		bufGradBias.WriteInt64(gradBiasSlice)
	} else {
		bufGradBias, err = e.hw.CreateHostCoherentBuffer(8)
		if err != nil {
			return nil, err
		}
		defer bufGradBias.Free()
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

	// Copy updated gradients back into tensors
	updatedGradW := bufGradW.ReadInt64(totalWeights)
	gradW.FromInt64Slice(updatedGradW)

	if gradBias != nil {
		updatedGradBias := bufGradBias.ReadInt64(outDim)
		gradBias.FromInt64Slice(updatedGradBias)
	}

	return dX, nil
}

// OptimizerStep executes parameter updates on the GPU, monitors Dynamic Bit Widening, and repacks storage.
func (e *GpuEngine) OptimizerStep(p *DynamicBitTensor, g *DynamicBitTensor, v *DynamicBitTensor, cfg OptimizerConfig) (int, error) {
	if p == nil || g == nil || v == nil {
		return 0, fmt.Errorf("tensor parameter, gradient, or velocity is nil")
	}
	count := p.Len()
	if g.Len() != count || v.Len() != count {
		return p.BitWidth(), fmt.Errorf("tensor length mismatch in OptimizerStep")
	}

	wSlice := p.ToInt64Slice()
	bufWeights, err := e.hw.CreateHostCoherentBuffer(uint64(count * 8))
	if err != nil {
		return p.BitWidth(), err
	}
	defer bufWeights.Free()
	bufWeights.WriteInt64(wSlice)

	gSlice := g.ToInt64Slice()
	bufGrads, err := e.hw.CreateHostCoherentBuffer(uint64(count * 8))
	if err != nil {
		return p.BitWidth(), err
	}
	defer bufGrads.Free()
	bufGrads.WriteInt64(gSlice)

	vSlice := v.ToInt64Slice()
	bufVelocities, err := e.hw.CreateHostCoherentBuffer(uint64(count * 8))
	if err != nil {
		return p.BitWidth(), err
	}
	defer bufVelocities.Free()
	bufVelocities.WriteInt64(vSlice)

	// meta[0] initialized to current bit width
	bufMeta, err := e.hw.CreateHostCoherentBuffer(4)
	if err != nil {
		return p.BitWidth(), err
	}
	defer bufMeta.Free()
	bufMeta.WriteUint32([]uint32{uint32(p.BitWidth())})

	optType := uint32(cfg.Type)
	pushOpt := [6]int32{
		int32(count),
		cfg.LearningRateQ16,
		cfg.MomentumBeta,
		int32(cfg.FractionalBits),
		int32(optType),
		cfg.WeightDecayQ16,
	}

	wg := uint32((count + 63) / 64)
	buffersOpt := []*GpuBuffer{bufWeights, bufGrads, bufVelocities, bufMeta}
	if err := e.kernelOptimizer.Dispatch(buffersOpt, unsafe.Pointer(&pushOpt[0]), 24, wg); err != nil {
		return p.BitWidth(), fmt.Errorf("GPU OptimizerStep dispatch failed: %w", err)
	}

	// Read updated velocity
	newV := bufVelocities.ReadInt64(count)
	v.FromInt64Slice(newV)

	// Check if DBW auto-widening occurred on the GPU
	metaResult := bufMeta.ReadUint32(1)[0]
	targetBitWidth := int(metaResult)
	if targetBitWidth < p.BitWidth() {
		targetBitWidth = p.BitWidth()
	}

	// Repack weights into VRAM bitstream at targetBitWidth
	totalBits := count * targetBitWidth
	numWords := (totalBits + 31) / 32
	if numWords == 0 {
		numWords = 1
	}

	bufPacked, err := e.hw.CreateHostCoherentBuffer(uint64(numWords * 4))
	if err != nil {
		return p.BitWidth(), err
	}
	defer bufPacked.Free()
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
	packedWords := bufPacked.ReadUint32(numWords)
	p.SetWords(packedWords, targetBitWidth)

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
	if e.kernelOptimizer != nil {
		e.kernelOptimizer.Close()
		e.kernelOptimizer = nil
	}
	if e.kernelRepack != nil {
		e.kernelRepack.Close()
		e.kernelRepack = nil
	}
}
