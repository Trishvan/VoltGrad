package test

import (
	"fmt"
	"math/rand"
	"os"
	"testing"

	. "github.com/Trishvan/VoltGrad/voltgrad/core"
)

// Test32BitExecutionPathDeviceInitialization verifies that HardwareOptions{Force32Bit: true}
// and VOLTGRAD_FORCE_32BIT=1 properly configure the 32-bit execution path.
func Test32BitExecutionPathDeviceInitialization(t *testing.T) {
	hw, err := NewHardwareContextWithOptions(HardwareOptions{Force32Bit: true})
	if err != nil {
		t.Skipf("Skipping Vulkan test (hardware initialization: %v)", err)
	}
	defer hw.Close()

	if hw.Is64Bit() {
		t.Fatalf("Expected Is64Bit() to be false when Force32Bit is true, got true")
	}
	if hw.MaxGpuBitWidth() != 32 {
		t.Fatalf("Expected MaxGpuBitWidth() to be 32, got %d", hw.MaxGpuBitWidth())
	}

	report := hw.DeviceReport()
	if report.Is64BitExecution {
		t.Fatalf("Expected DeviceReport.Is64BitExecution to be false, got true")
	}

	gpu, err := NewGpuEngine(hw)
	if err != nil {
		t.Fatalf("Failed creating GpuEngine in 32-bit mode: %v", err)
	}
	defer gpu.Close()

	if gpu.HardwareContext() != hw {
		t.Fatalf("Expected GpuEngine.HardwareContext() to match initialized context")
	}
	t.Logf("Successfully verified 32-bit execution path initialization on %s", report.DeviceName)
}

// Test32BitExecutionPathEnvVarOverride verifies that VOLTGRAD_FORCE_32BIT=1 enables 32-bit mode.
func Test32BitExecutionPathEnvVarOverride(t *testing.T) {
	os.Setenv("VOLTGRAD_FORCE_32BIT", "1")
	defer os.Unsetenv("VOLTGRAD_FORCE_32BIT")

	hw, err := NewHardwareContext()
	if err != nil {
		t.Skipf("Skipping Vulkan test (hardware initialization: %v)", err)
	}
	defer hw.Close()

	if hw.Is64Bit() {
		t.Fatalf("Expected Is64Bit() to be false with VOLTGRAD_FORCE_32BIT=1")
	}
}

// Test32BitPathGEMMParity tests that 32-bit compute shaders (pure GLSL 4.50)
// produce bit-for-bit identical results to the 64-bit CPU mathematical reference.
func Test32BitPathGEMMParity(t *testing.T) {
	hw, err := NewHardwareContextWithOptions(HardwareOptions{Force32Bit: true})
	if err != nil {
		t.Skipf("Skipping Vulkan test: %v", err)
	}
	defer hw.Close()

	gpu, err := NewGpuEngine(hw)
	if err != nil {
		t.Fatalf("Failed to initialize 32-bit GpuEngine: %v", err)
	}
	defer gpu.Close()

	inDim, outDim := 8, 4
	shift := uint(16)
	bitWidths := []int{2, 3, 4, 8, 12, 16, 24, 32}

	for _, bw := range bitWidths {
		t.Run(fmt.Sprintf("%d-bit", bw), func(t *testing.T) {
			w := NewDynamicBitTensor([]int{outDim, inDim}, bw)
			bias := NewDynamicBitTensor([]int{outDim}, 16)

			r := rand.New(rand.NewSource(int64(42 + bw)))
			for i := 0; i < outDim*inDim; i++ {
				maxVal := int64(1)<<(bw-1) - 1
				val := r.Int63n(maxVal*2+1) - maxVal
				w.Set(i, val)
			}
			for i := 0; i < outDim; i++ {
				bias.Set(i, int64(r.Int31n(1000)-500))
			}

			x := make([]int32, inDim)
			for i := 0; i < inDim; i++ {
				x[i] = r.Int31n(2000) - 1000
			}

			// 1. CPU reference
			cpuOut, err := MatMulVecWithBias64(w, x, bias, shift)
			if err != nil {
				t.Fatalf("CPU forward failed: %v", err)
			}

			// 2. 32-bit GPU execution
			gpuOut, err := gpu.MatMulForward(w, x, bias, shift)
			if err != nil {
				t.Fatalf("32-bit GPU forward failed: %v", err)
			}

			// Parity check
			for i := 0; i < outDim; i++ {
				if cpuOut[i] != gpuOut[i] {
					t.Fatalf("Parity mismatch at [%d]: CPU=%d, GPU32=%d (bitWidth=%d)",
						i, cpuOut[i], gpuOut[i], bw)
				}
			}
		})
	}
}

// Test32BitPathBatchedGEMMParity tests batched GEMM forward on the 32-bit GPU execution path.
func Test32BitPathBatchedGEMMParity(t *testing.T) {
	hw, err := NewHardwareContextWithOptions(HardwareOptions{Force32Bit: true})
	if err != nil {
		t.Skipf("Skipping Vulkan test: %v", err)
	}
	defer hw.Close()

	gpu, err := NewGpuEngine(hw)
	if err != nil {
		t.Fatalf("Failed to initialize 32-bit GpuEngine: %v", err)
	}
	defer gpu.Close()

	batchSizes := []int{1, 4, 16}
	inDim, outDim := 6, 3
	shift := uint(16)

	for _, bSize := range batchSizes {
		for _, bw := range []int{4, 8, 16, 32} {
			w := NewDynamicBitTensor([]int{outDim, inDim}, bw)
			bias := NewDynamicBitTensor([]int{outDim}, 16)

			r := rand.New(rand.NewSource(int64(100*bSize + bw)))
			for i := 0; i < outDim*inDim; i++ {
				maxVal := int64(1)<<(bw-1) - 1
				val := r.Int63n(maxVal*2+1) - maxVal
				w.Set(i, val)
			}
			for i := 0; i < outDim; i++ {
				bias.Set(i, int64(r.Int31n(1000)-500))
			}

			flatX := make([]int32, bSize*inDim)
			for i := range flatX {
				flatX[i] = r.Int31n(2000) - 1000
			}

			// CPU batched reference
			cpuOut := make([]int32, bSize*outDim)
			for b := 0; b < bSize; b++ {
				sampleX := flatX[b*inDim : (b+1)*inDim]
				sampleY, err := MatMulVecWithBias64(w, sampleX, bias, shift)
				if err != nil {
					t.Fatalf("CPU forward failed: %v", err)
				}
				copy(cpuOut[b*outDim:(b+1)*outDim], sampleY)
			}

			// 32-bit GPU batched
			gpuOut, err := gpu.MatMulBatchedForward(w, flatX, bSize, bias, shift)
			if err != nil {
				t.Fatalf("32-bit GPU batched forward failed: %v", err)
			}

			for i := range cpuOut {
				if cpuOut[i] != gpuOut[i] {
					t.Fatalf("Batched parity mismatch at [%d]: CPU=%d, GPU32=%d (batchSize=%d, bw=%d)",
						i, cpuOut[i], gpuOut[i], bSize, bw)
				}
			}
		}
	}
}

// Test32BitPathBackwardParity tests dX and dW backpropagation parity on 32-bit GPU path.
func Test32BitPathBackwardParity(t *testing.T) {
	hw, err := NewHardwareContextWithOptions(HardwareOptions{Force32Bit: true})
	if err != nil {
		t.Skipf("Skipping Vulkan test: %v", err)
	}
	defer hw.Close()

	gpu, err := NewGpuEngine(hw)
	if err != nil {
		t.Fatalf("Failed to initialize 32-bit GpuEngine: %v", err)
	}
	defer gpu.Close()

	inDim, outDim := 6, 4
	shift := uint(16)
	bw := 16

	w := NewDynamicBitTensor([]int{outDim, inDim}, bw)
	r := rand.New(rand.NewSource(777))
	for i := 0; i < outDim*inDim; i++ {
		w.Set(i, int64(r.Int31n(2000)-1000))
	}

	x := make([]int32, inDim)
	for i := range x {
		x[i] = r.Int31n(1000) - 500
	}
	dOut := make([]int32, outDim)
	for i := range dOut {
		dOut[i] = r.Int31n(1000) - 500
	}

	// CPU reference
	cpuGradW := NewDynamicBitTensor([]int{outDim, inDim}, bw)
	cpuDX, err := MatMulVecBackward(dOut, x, w, cpuGradW, shift)
	if err != nil {
		t.Fatalf("CPU backward failed: %v", err)
	}

	// 32-bit GPU
	gpuGradW := NewDynamicBitTensor([]int{outDim, inDim}, bw)
	gpuDX, err := gpu.MatMulBackward(dOut, x, w, gpuGradW, nil, shift)
	if err != nil {
		t.Fatalf("32-bit GPU backward failed: %v", err)
	}

	for i := 0; i < inDim; i++ {
		if cpuDX[i] != gpuDX[i] {
			t.Fatalf("dX mismatch at [%d]: CPU=%d, GPU32=%d", i, cpuDX[i], gpuDX[i])
		}
	}
	for i := 0; i < outDim*inDim; i++ {
		if cpuGradW.Get(i) != gpuGradW.Get(i) {
			t.Fatalf("dW mismatch at [%d]: CPU=%d, GPU32=%d", i, cpuGradW.Get(i), gpuGradW.Get(i))
		}
	}
}

// Test32BitPathMSELossParity tests MSE loss on the 32-bit GPU execution path.
func Test32BitPathMSELossParity(t *testing.T) {
	hw, err := NewHardwareContextWithOptions(HardwareOptions{Force32Bit: true})
	if err != nil {
		t.Skipf("Skipping Vulkan test: %v", err)
	}
	defer hw.Close()

	gpu, err := NewGpuEngine(hw)
	if err != nil {
		t.Fatalf("Failed to initialize 32-bit GpuEngine: %v", err)
	}
	defer gpu.Close()

	n := 128
	shift := uint(16)
	r := rand.New(rand.NewSource(999))
	preds := make([]int32, n)
	targets := make([]int32, n)
	for i := 0; i < n; i++ {
		preds[i] = r.Int31n(10000) - 5000
		targets[i] = r.Int31n(10000) - 5000
	}

	// CPU loss
	cpuLoss, cpuGrad, err := ComputeLoss(LossMSE, preds, targets, shift)
	if err != nil {
		t.Fatalf("CPU loss failed: %v", err)
	}

	// 32-bit GPU loss
	gpuLoss, gpuGrad, err := gpu.MSELoss(preds, targets, shift)
	if err != nil {
		t.Fatalf("32-bit GPU loss failed: %v", err)
	}

	if cpuLoss != gpuLoss {
		t.Fatalf("Loss mismatch: CPU=%d, GPU32=%d", cpuLoss, gpuLoss)
	}
	for i := 0; i < n; i++ {
		if cpuGrad[i] != gpuGrad[i] {
			t.Fatalf("Loss grad mismatch at [%d]: CPU=%d, GPU32=%d", i, cpuGrad[i], gpuGrad[i])
		}
	}
}

// Test32BitPathDBWPrecisionGracefulFallback verifies that when a tensor requires > 32 bits,
// the 32-bit GPU runtime gracefully falls back to the exact 64-bit CPU reference without error.
func Test32BitPathDBWPrecisionGracefulFallback(t *testing.T) {
	hw, err := NewHardwareContextWithOptions(HardwareOptions{Force32Bit: true})
	if err != nil {
		t.Skipf("Skipping Vulkan test: %v", err)
	}
	defer hw.Close()

	gpu, err := NewGpuEngine(hw)
	if err != nil {
		t.Fatalf("Failed to initialize 32-bit GpuEngine: %v", err)
	}
	defer gpu.Close()

	inDim, outDim := 4, 2
	shift := uint(16)

	// Create a 40-bit tensor (> 32-bit)
	w40 := NewDynamicBitTensor([]int{outDim, inDim}, 40)
	w40.Set(0, int64(1)<<36)
	w40.Set(1, -(int64(1) << 35))
	w40.Set(2, 5000)
	w40.Set(3, -7000)

	x := []int32{100, -200, 300, -400}

	// 1. MatMulForward with W=40 on 32-bit engine -> fallback to CPU reference
	gpuOut, err := gpu.MatMulForward(w40, x, nil, shift)
	if err != nil {
		t.Fatalf("MatMulForward with >32-bit tensor on 32-bit GPU failed: %v", err)
	}
	cpuOut, err := MatMulVecWithBias64(w40, x, nil, shift)
	if err != nil {
		t.Fatalf("CPU reference failed: %v", err)
	}
	for i := range cpuOut {
		if cpuOut[i] != gpuOut[i] {
			t.Fatalf("Fallback parity mismatch at [%d]: CPU=%d, GPU=%d", i, cpuOut[i], gpuOut[i])
		}
	}

	// 2. Batched Forward with W=40 on 32-bit engine
	flatX := []int32{100, -200, 300, -400, 50, -50, 100, -100}
	batchSize := 2
	gpuBatchedOut, err := gpu.MatMulBatchedForward(w40, flatX, batchSize, nil, shift)
	if err != nil {
		t.Fatalf("MatMulBatchedForward with >32-bit tensor failed: %v", err)
	}
	if len(gpuBatchedOut) != batchSize*outDim {
		t.Fatalf("Expected output len %d, got %d", batchSize*outDim, len(gpuBatchedOut))
	}

	// 3. Backward with W=40 on 32-bit engine
	dOut := []int32{10, -20}
	gradW := NewDynamicBitTensor([]int{outDim, inDim}, 40)
	gpuDX, err := gpu.MatMulBackward(dOut, x, w40, gradW, nil, shift)
	if err != nil {
		t.Fatalf("MatMulBackward with >32-bit tensor failed: %v", err)
	}
	if len(gpuDX) != inDim {
		t.Fatalf("Expected dX len %d, got %d", inDim, len(gpuDX))
	}

	// 4. LinearLayer ForwardCtx and BackwardCtx with W=40
	layer := NewLinearLayer("fc40", inDim, outDim, 40, false)
	layer.Weight.Set(0, int64(1)<<36)
	ctx := NewComputeContext(nil)
	ctx.SetGpuEngine(gpu)

	layerOut, err := layer.ForwardCtx(ctx, x, shift)
	if err != nil {
		t.Fatalf("LinearLayer.ForwardCtx failed with >32-bit tensor: %v", err)
	}
	if len(layerOut) != outDim {
		t.Fatalf("Expected layerOut len %d, got %d", outDim, len(layerOut))
	}

	layerDX, err := layer.BackwardCtx(ctx, dOut, x, shift)
	if err != nil {
		t.Fatalf("LinearLayer.BackwardCtx failed with >32-bit tensor: %v", err)
	}
	if len(layerDX) != inDim {
		t.Fatalf("Expected layerDX len %d, got %d", inDim, len(layerDX))
	}

	t.Log(">>> Verified graceful CPU fallback for >32-bit tensors on 32-bit GPU engine! <<<")
}

// TestPersistentTensor32BitVs64BitMemoryFootprint compares buffer allocation between 32-bit and 64-bit paths.
func TestPersistentTensor32BitVs64BitMemoryFootprint(t *testing.T) {
	hw32, err := NewHardwareContextWithOptions(HardwareOptions{Force32Bit: true})
	if err != nil {
		t.Skipf("Skipping Vulkan test: %v", err)
	}
	defer hw32.Close()

	numElements := 1024
	tensor := NewDynamicBitTensor([]int{numElements}, 8)

	pt32, err := NewPersistentGpuTensor(hw32, tensor)
	if err != nil {
		t.Fatalf("Failed creating 32-bit persistent tensor: %v", err)
	}
	defer pt32.Free()

	if pt32.Is64Bit() {
		t.Fatalf("Expected pt32.Is64Bit() to be false")
	}
	// In 32-bit mode, buffer capacity is len * 4 bytes
	if pt32.BufferParam().Capacity() != uint64(numElements*4) {
		t.Fatalf("Expected 32-bit persistent buffer capacity %d, got %d",
			numElements*4, pt32.BufferParam().Capacity())
	}

	t.Logf("32-bit persistent tensor allocated %d bytes/buffer (%d bytes total across 4 buffers)",
		pt32.BufferParam().Capacity(), pt32.BufferParam().Capacity()*4)

	// If physical GPU supports 64-bit, compare with 64-bit allocation
	if hw32.HasShaderInt64() {
		hw64, err := NewHardwareContextWithOptions(HardwareOptions{Force32Bit: false})
		if err != nil {
			t.Fatalf("Failed initializing 64-bit hardware: %v", err)
		}
		defer hw64.Close()

		pt64, err := NewPersistentGpuTensor(hw64, tensor)
		if err != nil {
			t.Fatalf("Failed creating 64-bit persistent tensor: %v", err)
		}
		defer pt64.Free()

		if !pt64.Is64Bit() {
			t.Fatalf("Expected pt64.Is64Bit() to be true")
		}
		if pt64.BufferParam().Capacity() != uint64(numElements*8) {
			t.Fatalf("Expected 64-bit persistent buffer capacity %d, got %d",
				numElements*8, pt64.BufferParam().Capacity())
		}

		ratio := float64(pt32.BufferParam().Capacity()) / float64(pt64.BufferParam().Capacity())
		if ratio != 0.5 {
			t.Fatalf("Expected 32-bit/64-bit capacity ratio of 0.5 (50%% VRAM reduction), got %f", ratio)
		}
		t.Logf(">>> 32-bit execution path achieves exactly 50%% VRAM capacity reduction (4 vs 8 bytes/element) <<<")
	}
}

// Test32BitOptimizerParity tests SGD, SGDMomentum, and AdamQ on the 32-bit GPU path against the CPU reference.
func Test32BitOptimizerParity(t *testing.T) {
	hw, err := NewHardwareContextWithOptions(HardwareOptions{Force32Bit: true})
	if err != nil {
		t.Skipf("Skipping Vulkan test: %v", err)
	}
	defer hw.Close()

	gpu, err := NewGpuEngine(hw)
	if err != nil {
		t.Fatalf("Failed to initialize 32-bit GpuEngine: %v", err)
	}
	defer gpu.Close()

	optConfigs := []OptimizerConfig{
		{Type: OptimizerSGD, LearningRateQ16: 655, FractionalBits: 16},
		{Type: OptimizerSGDMomentum, LearningRateQ16: 655, MomentumBeta: 58982, FractionalBits: 16},
		{Type: OptimizerAdamQ, LearningRateQ16: 655, MomentumBeta: 58982, FractionalBits: 16},
	}

	for _, cfg := range optConfigs {
		t.Run(fmt.Sprintf("OptType-%d", cfg.Type), func(t *testing.T) {
			count := 16
			pCPU := NewDynamicBitTensor([]int{count}, 4)
			gCPU := NewDynamicBitTensor([]int{count}, 4)
			vCPU := NewDynamicBitTensor([]int{count}, 4)

			pGPU := NewDynamicBitTensor([]int{count}, 4)
			gGPU := NewDynamicBitTensor([]int{count}, 4)
			vGPU := NewDynamicBitTensor([]int{count}, 4)

			r := rand.New(rand.NewSource(12345))
			for i := 0; i < count; i++ {
				val := int64(r.Int31n(14) - 7)
				grad := int64(r.Int31n(14) - 7)
				pCPU.Set(i, val)
				pGPU.Set(i, val)
				gCPU.Set(i, grad)
				gGPU.Set(i, grad)
			}

			// Single param step on CPU
			opt := NewDynamicBitOptimizerWithVelocities(cfg, nil)
			opt.StepParamCPU(0, pCPU, gCPU, vCPU)

			// Single param step on 32-bit GPU
			_, err := gpu.OptimizerStep(pGPU, gGPU, vGPU, cfg)
			if err != nil {
				t.Fatalf("32-bit GPU OptimizerStep failed: %v", err)
			}

			// Parity check
			for i := 0; i < count; i++ {
				if pCPU.Get(i) != pGPU.Get(i) {
					t.Fatalf("Param parity mismatch at [%d]: CPU=%d, GPU32=%d", i, pCPU.Get(i), pGPU.Get(i))
				}
				if vCPU.Get(i) != vGPU.Get(i) {
					t.Fatalf("Velocity parity mismatch at [%d]: CPU=%d, GPU32=%d", i, vCPU.Get(i), vGPU.Get(i))
				}
			}
			if pCPU.BitWidth() != pGPU.BitWidth() {
				t.Fatalf("BitWidth mismatch: CPU=%d, GPU32=%d", pCPU.BitWidth(), pGPU.BitWidth())
			}
		})
	}
}

// Test32BitResidentPipelineEndToEnd trains a model on the 32-bit GPU execution path with persistent tensors.
func Test32BitResidentPipelineEndToEnd(t *testing.T) {
	hw, err := NewHardwareContextWithOptions(HardwareOptions{Force32Bit: true})
	if err != nil {
		t.Skipf("Skipping Vulkan test: %v", err)
	}
	defer hw.Close()

	gpu, err := NewGpuEngine(hw)
	if err != nil {
		t.Fatalf("Failed to initialize GpuEngine: %v", err)
	}
	defer gpu.Close()

	layer := NewLinearLayer("fc1", 4, 2, 2, true)
	if err := layer.BindGPU(hw); err != nil {
		t.Fatalf("Failed to bind layer to 32-bit GPU: %v", err)
	}
	defer layer.FreeGPU()

	ctx := NewComputeContext(nil)
	ctx.SetGpuEngine(gpu)

	optCfg := OptimizerConfig{
		Type:            OptimizerSGDMomentum,
		LearningRateQ16: 6553, // ~0.1
		MomentumBeta:    58982,
		FractionalBits:  16,
	}

	x := []int32{1000, 2000, -1000, -2000, 500, 1500, -500, -1500} // batch size 2
	targets := []int32{500, -500, 250, -250}
	batchSize := 2

	var prevLoss int32 = 1<<30
	for step := 0; step < 5; step++ {
		layer.ZeroGrad()

		// 1. Forward Batched Resident
		preds, err := layer.ForwardBatchedCtx(ctx, x, batchSize, 16)
		if err != nil {
			t.Fatalf("ForwardBatchedCtx failed at step %d: %v", step, err)
		}

		// 2. MSE Loss on 32-bit GPU
		loss, dLoss, err := gpu.MSELoss(preds, targets, 16)
		if err != nil {
			t.Fatalf("MSELoss failed at step %d: %v", step, err)
		}

		// 3. Backward Batched Resident
		_, err = layer.BackwardBatchedCtx(ctx, dLoss, x, batchSize, 16)
		if err != nil {
			t.Fatalf("BackwardBatchedCtx failed at step %d: %v", step, err)
		}

		// 4. Resident Optimizer Step on GPU
		_, err = gpu.OptimizerStepResident(
			layer.GpuWeight.BufferInt64(),
			layer.GpuWeight.BufferGrad(),
			layer.GpuWeight.BufferVel(),
			layer.GpuWeight.BufferPacked(),
			layer.Weight.Len(),
			layer.Weight.BitWidth(),
			optCfg,
		)
		if err != nil {
			t.Fatalf("OptimizerStepResident failed at step %d: %v", step, err)
		}

		prevLoss = loss
	}

	// Sync back to host and verify valid state
	if err := layer.SyncToHost(); err != nil {
		t.Fatalf("SyncToHost failed: %v", err)
	}

	t.Logf("Final 32-bit resident step completed with loss=%d, weight bit-width=%d", prevLoss, layer.Weight.BitWidth())
}
