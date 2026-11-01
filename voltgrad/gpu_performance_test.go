package voltgrad

import (
	"fmt"
	"math/rand"
	"sync"
	"testing"
	"unsafe"
)

func randInt64ForBits(r *rand.Rand, bitWidth int) int64 {
	if bitWidth >= 63 {
		return r.Int63() - (1 << 62)
	}
	maxVal := (int64(1) << (bitWidth - 1)) - 1
	if maxVal <= 0 {
		return 0
	}
	return r.Int63n(maxVal*2+1) - maxVal
}

func TestBatchedGEMMForwardParity(t *testing.T) {
	hw, err := NewHardwareContext()
	if err != nil {
		t.Skipf("Skipping Vulkan test: %v", err)
	}
	defer hw.Close()

	gpu, err := NewGpuEngine(hw)
	if err != nil {
		t.Fatalf("Failed to initialize GpuEngine: %v", err)
	}
	defer gpu.Close()

	batchSizes := []int{1, 4, 16}
	inDim, outDim := 8, 4
	shift := uint(16)

	for _, bSize := range batchSizes {
		for _, bitWidth := range []int{4, 8, 16, 32, 64} {
			w := NewDynamicBitTensor([]int{outDim, inDim}, bitWidth)
			bias := NewDynamicBitTensor([]int{outDim}, 32)

			// Fill weights and bias
			r := rand.New(rand.NewSource(42))
			for i := 0; i < outDim*inDim; i++ {
				w.Set(i, randInt64ForBits(r, bitWidth))
			}
			for i := 0; i < outDim; i++ {
				bias.Set(i, int64(r.Int31n(10000)-5000))
			}

			// Generate batch of inputs
			flatX := make([]int32, bSize*inDim)
			sampleInputs := make([][]int32, bSize)
			for b := 0; b < bSize; b++ {
				sampleInputs[b] = make([]int32, inDim)
				for c := 0; c < inDim; c++ {
					v := r.Int31n(20000) - 10000
					flatX[b*inDim+c] = v
					sampleInputs[b][c] = v
				}
			}

			// 1. Batched GPU forward
			batchedY, err := gpu.MatMulBatchedForward(w, flatX, bSize, bias, shift)
			if err != nil {
				t.Fatalf("MatMulBatchedForward failed (B=%d, W=%d): %v", bSize, bitWidth, err)
			}

			if len(batchedY) != bSize*outDim {
				t.Fatalf("Expected batched output len %d, got %d", bSize*outDim, len(batchedY))
			}

			// 2. Sample-by-sample GPU forward & CPU reference
			for b := 0; b < bSize; b++ {
				singleY, err := gpu.MatMulForward(w, sampleInputs[b], bias, shift)
				if err != nil {
					t.Fatalf("MatMulForward failed at sample %d: %v", b, err)
				}
				cpuY, err := MatMulVecWithBias64(w, sampleInputs[b], bias, shift)
				if err != nil {
					t.Fatalf("MatMulVecWithBias64 failed at sample %d: %v", b, err)
				}

				for rIdx := 0; rIdx < outDim; rIdx++ {
					batchVal := batchedY[b*outDim+rIdx]
					singleVal := singleY[rIdx]
					cpuVal := cpuY[rIdx]

					if batchVal != singleVal {
						t.Errorf("Sample %d elem %d: Batched GPU (%d) != Single GPU (%d)", b, rIdx, batchVal, singleVal)
					}
					if batchVal != cpuVal {
						t.Errorf("Sample %d elem %d: Batched GPU (%d) != CPU (%d)", b, rIdx, batchVal, cpuVal)
					}
				}
			}
		}
	}
	t.Log(">>> Batched GEMM forward pass parity verified bit-for-bit across all batch sizes! <<<")
}

func TestBatchedGEMMBackwardParity(t *testing.T) {
	hw, err := NewHardwareContext()
	if err != nil {
		t.Skipf("Skipping Vulkan test: %v", err)
	}
	defer hw.Close()

	gpu, err := NewGpuEngine(hw)
	if err != nil {
		t.Fatalf("Failed to initialize GpuEngine: %v", err)
	}
	defer gpu.Close()

	batchSizes := []int{1, 4, 16}
	inDim, outDim := 6, 3
	shift := uint(16)

	for _, bSize := range batchSizes {
		for _, bitWidth := range []int{4, 8, 16, 32, 64} {
			w := NewDynamicBitTensor([]int{outDim, inDim}, bitWidth)
			dW_batched := NewDynamicBitTensor([]int{outDim, inDim}, 64)
			dBias_batched := NewDynamicBitTensor([]int{outDim}, 64)

			dW_single := NewDynamicBitTensor([]int{outDim, inDim}, 64)
			dBias_single := NewDynamicBitTensor([]int{outDim}, 64)

			r := rand.New(rand.NewSource(123))
			for i := 0; i < outDim*inDim; i++ {
				w.Set(i, randInt64ForBits(r, bitWidth))
			}

			flatX := make([]int32, bSize*inDim)
			flatDOut := make([]int32, bSize*outDim)
			sampleX := make([][]int32, bSize)
			sampleDOut := make([][]int32, bSize)

			for b := 0; b < bSize; b++ {
				sampleX[b] = make([]int32, inDim)
				sampleDOut[b] = make([]int32, outDim)
				for c := 0; c < inDim; c++ {
					v := r.Int31n(10000) - 5000
					flatX[b*inDim+c] = v
					sampleX[b][c] = v
				}
				for row := 0; row < outDim; row++ {
					v := r.Int31n(10000) - 5000
					flatDOut[b*outDim+row] = v
					sampleDOut[b][row] = v
				}
			}

			// 1. Batched Backward
			batchedDX, err := gpu.MatMulBatchedBackward(flatDOut, flatX, bSize, w, dW_batched, dBias_batched, shift)
			if err != nil {
				t.Fatalf("MatMulBatchedBackward failed (B=%d, W=%d): %v", bSize, bitWidth, err)
			}

			// 2. Sample-by-sample Backward
			singleDXAll := make([]int32, bSize*inDim)
			for b := 0; b < bSize; b++ {
				singleDX, err := gpu.MatMulBackward(sampleDOut[b], sampleX[b], w, dW_single, dBias_single, shift)
				if err != nil {
					t.Fatalf("MatMulBackward failed at sample %d: %v", b, err)
				}
				copy(singleDXAll[b*inDim:(b+1)*inDim], singleDX)
			}

			// Verify dX parity
			for i := 0; i < bSize*inDim; i++ {
				if batchedDX[i] != singleDXAll[i] {
					t.Fatalf("dX mismatch at index %d: Batched (%d) != Sample-by-sample (%d)",
						i, batchedDX[i], singleDXAll[i])
				}
			}

			// Verify dW gradient accumulation parity
			for i := 0; i < outDim*inDim; i++ {
				if dW_batched.Get(i) != dW_single.Get(i) {
					t.Fatalf("dW mismatch at index %d: Batched (%d) != Sample-by-sample (%d)",
						i, dW_batched.Get(i), dW_single.Get(i))
				}
			}

			// Verify dBias gradient accumulation parity
			for i := 0; i < outDim; i++ {
				if dBias_batched.Get(i) != dBias_single.Get(i) {
					t.Fatalf("dBias mismatch at index %d: Batched (%d) != Sample-by-sample (%d)",
						i, dBias_batched.Get(i), dBias_single.Get(i))
				}
			}
		}
	}
	t.Log(">>> Batched GEMM backward pass parity verified bit-for-bit across all batch sizes! <<<")
}

func TestGPUActivationParity(t *testing.T) {
	hw, err := NewHardwareContext()
	if err != nil {
		t.Skipf("Skipping Vulkan test: %v", err)
	}
	defer hw.Close()

	gpu, err := NewGpuEngine(hw)
	if err != nil {
		t.Fatalf("Failed to initialize GpuEngine: %v", err)
	}
	defer gpu.Close()

	testInputs := []int32{
		-1000, -1, 0, 1, 42, 65536, -65536, 2147483647, -2147483648,
	}

	// 1. ReLU Forward
	gpuOut, err := gpu.ReLUForward(testInputs)
	if err != nil {
		t.Fatalf("ReLUForward failed: %v", err)
	}
	cpuOut := ReLU(testInputs)

	for i := range testInputs {
		if gpuOut[i] != cpuOut[i] {
			t.Errorf("ReLU mismatch at %d: GPU=%d, CPU=%d", i, gpuOut[i], cpuOut[i])
		}
	}

	// 2. ReLU Backward
	dOut := []int32{10, 20, 30, 40, 50, 60, 70, 80, 90}
	gpuDX, err := gpu.ReLUBackward(dOut, testInputs)
	if err != nil {
		t.Fatalf("ReLUBackward failed: %v", err)
	}
	cpuDX := ReLUBackward(dOut, testInputs)

	for i := range testInputs {
		if gpuDX[i] != cpuDX[i] {
			t.Errorf("ReLUBackward mismatch at %d: GPU=%d, CPU=%d", i, gpuDX[i], cpuDX[i])
		}
	}
	t.Log(">>> GPU ReLU forward & backward parity verified bit-for-bit! <<<")
}

func TestGPUMSELossParity(t *testing.T) {
	hw, err := NewHardwareContext()
	if err != nil {
		t.Skipf("Skipping Vulkan test: %v", err)
	}
	defer hw.Close()

	gpu, err := NewGpuEngine(hw)
	if err != nil {
		t.Fatalf("Failed to initialize GpuEngine: %v", err)
	}
	defer gpu.Close()

	preds := []int32{1000, 2500, -1500, 32000, -5000}
	targets := []int32{950, 2000, -1200, 30000, -6000}
	shift := uint(16)

	gpuLoss, gpuGrad, err := gpu.MSELoss(preds, targets, shift)
	if err != nil {
		t.Fatalf("MSELoss failed: %v", err)
	}

	cpuLoss, cpuGrad, err := ComputeLoss(LossMSE, preds, targets, shift)
	if err != nil {
		t.Fatalf("ComputeLoss failed: %v", err)
	}

	if gpuLoss != cpuLoss {
		t.Errorf("Loss scalar mismatch: GPU=%d, CPU=%d", gpuLoss, cpuLoss)
	}

	for i := range preds {
		if gpuGrad[i] != cpuGrad[i] {
			t.Errorf("Loss grad mismatch at %d: GPU=%d, CPU=%d", i, gpuGrad[i], cpuGrad[i])
		}
	}
	t.Log(">>> GPU MSE Loss & reduction parity verified bit-for-bit with CPU! <<<")
}

func TestDeviceCapabilityReport(t *testing.T) {
	hw, err := NewHardwareContext()
	if err != nil {
		t.Skipf("Skipping Vulkan test: %v", err)
	}
	defer hw.Close()

	report := hw.DeviceReport()
	if report.DeviceName == "" {
		t.Errorf("DeviceName is empty")
	}
	if report.VendorID == 0 {
		t.Errorf("VendorID is zero")
	}
	if report.APIVersion == "" {
		t.Errorf("APIVersion is empty")
	}
	if !report.HasShaderInt64 {
		t.Errorf("HasShaderInt64 must be true")
	}
	if len(report.MemoryHeaps) == 0 {
		t.Errorf("Expected at least 1 memory heap")
	}

	reportStr := report.String()
	t.Logf("\n=== Vulkan Device Capability Telemetry ===\n%s\n", reportStr)
}

// TestAsyncDispatchAndProfiling verifies asynchronous execution tokens and GPU profiling capabilities.
func TestAsyncDispatchAndProfiling(t *testing.T) {
	hw, err := NewHardwareContext()
	if err != nil {
		t.Skipf("Skipping Vulkan test: %v", err)
	}
	defer hw.Close()

	gpu, err := NewGpuEngine(hw)
	if err != nil {
		t.Fatalf("Failed to initialize GpuEngine: %v", err)
	}
	defer gpu.Close()

	// Create input and output buffers for ReLU kernel
	n := 1024
	bufX, err := hw.CreateHostCoherentBuffer(uint64(n * 4))
	if err != nil {
		t.Fatalf("CreateHostCoherentBuffer X failed: %v", err)
	}
	defer bufX.Free()

	bufY, err := hw.CreateHostCoherentBuffer(uint64(n * 4))
	if err != nil {
		t.Fatalf("CreateHostCoherentBuffer Y failed: %v", err)
	}
	defer bufY.Free()

	input := make([]int32, n)
	for i := range input {
		input[i] = int32(i - n/2)
	}
	if err := bufX.WriteInt32(input); err != nil {
		t.Fatalf("WriteInt32 failed: %v", err)
	}

	push := [1]uint32{uint32(n)}
	wg := uint32((n + 63) / 64)
	buffers := []*GpuBuffer{bufX, bufY}

	// 1. Asynchronous Dispatch
	token, err := gpu.kernelReLUFwd.DispatchAsync(buffers, unsafe.Pointer(&push[0]), 4, wg)
	if err != nil {
		t.Fatalf("DispatchAsync failed: %v", err)
	}
	if token == nil {
		t.Fatalf("Expected non-nil ExecutionToken")
	}

	// Wait for token completion
	if err := token.Wait(2000000000); err != nil {
		t.Fatalf("ExecutionToken.Wait failed: %v", err)
	}

	res, err := bufY.ReadInt32(n)
	if err != nil {
		t.Fatalf("ReadInt32 failed: %v", err)
	}
	for i, v := range res {
		expected := int32(0)
		if input[i] > 0 {
			expected = input[i]
		}
		if v != expected {
			t.Fatalf("Async dispatch result mismatch at %d: got %d, expected %d", i, v, expected)
		}
	}

	// 2. Profiled Execution
	dur, err := gpu.kernelReLUFwd.DispatchProfiled(buffers, unsafe.Pointer(&push[0]), 4, wg)
	if err != nil {
		t.Fatalf("DispatchProfiled failed: %v", err)
	}
	if dur <= 0 {
		t.Fatalf("Expected positive execution duration from DispatchProfiled, got %v", dur)
	}

	t.Logf(">>> GPU Asynchronous Dispatch and Timestamp Profiling VERIFIED! Duration: %v <<<", dur)
}

// TestPersistentGpuTensorLifecycle validates allocation, GPU persistence, and host synchronization of resident tensors.
func TestPersistentGpuTensorLifecycle(t *testing.T) {
	hw, err := NewHardwareContext()
	if err != nil {
		t.Skipf("Skipping Vulkan test: %v", err)
	}
	defer hw.Close()

	layer := NewLinearLayer("fc_persistent", 8, 4, 4, true)
	for i := 0; i < layer.Weight.Len(); i++ {
		layer.Weight.Set(i, int64(i+1))
	}
	for i := 0; i < layer.Bias.Len(); i++ {
		layer.Bias.Set(i, int64((i+1)*10))
	}

	// 1. Bind layer to GPU
	if err := layer.BindGPU(hw); err != nil {
		t.Fatalf("BindGPU failed: %v", err)
	}
	defer layer.FreeGPU()

	if layer.GpuWeight == nil || layer.GpuWeight.Buffer() == nil {
		t.Fatalf("Expected non-nil persistent GPU weight buffer")
	}
	if layer.GpuBias == nil || layer.GpuBias.Buffer() == nil {
		t.Fatalf("Expected non-nil persistent GPU bias buffer")
	}

	// 2. Modify CPU tensor, sync to GPU, and read back
	layer.Weight.Set(0, 7)
	if err := layer.GpuWeight.SyncToGPU(); err != nil {
		t.Fatalf("SyncToGPU failed: %v", err)
	}

	// Reset CPU memory to zero
	layer.Weight.Set(0, 0)
	if err := layer.SyncToHost(); err != nil {
		t.Fatalf("SyncToHost failed: %v", err)
	}

	if val := layer.Weight.Get(0); val != 7 {
		t.Fatalf("Expected synced weight value 7, got %d", val)
	}

	t.Log(">>> Persistent GPU Tensor Lifecycle and Host Synchronization VERIFIED! <<<")
}

// TestDeviceTimestampPeriod verifies that timestampPeriod is properly queried and parsed from physical device limits.
func TestDeviceTimestampPeriod(t *testing.T) {
	hw, err := NewHardwareContext()
	if err != nil {
		t.Skipf("Skipping Vulkan test: %v", err)
	}
	defer hw.Close()

	period := hw.TimestampPeriod()
	if period <= 0 {
		t.Fatalf("Expected positive timestamp period, got %f", period)
	}
	report := hw.DeviceReport()
	if report.TimestampPeriod <= 0 {
		t.Fatalf("Expected positive TimestampPeriod in DeviceReport, got %f", report.TimestampPeriod)
	}
	t.Logf(">>> Vulkan Hardware Timestamp Period Verified: %.2f ns/tick <<<", period)
}

// TestConcurrentAsyncAndProfiledDispatchSafety tests concurrent execution of DispatchAsync,
// token.Wait(), and DispatchProfiled on the same ComputeKernel under contention.
func TestConcurrentAsyncAndProfiledDispatchSafety(t *testing.T) {
	hw, err := NewHardwareContext()
	if err != nil {
		t.Skipf("Skipping Vulkan test: %v", err)
	}
	defer hw.Close()

	gpu, err := NewGpuEngine(hw)
	if err != nil {
		t.Fatalf("Failed to initialize GpuEngine: %v", err)
	}
	defer gpu.Close()

	n := 512
	wg := uint32((n + 63) / 64)
	push := [1]uint32{uint32(n)}

	numWorkers := 8
	iterations := 10
	var wgWorkers sync.WaitGroup
	errCh := make(chan error, numWorkers*iterations*2)

	for w := 0; w < numWorkers; w++ {
		wgWorkers.Add(1)
		workerID := w
		go func() {
			defer wgWorkers.Done()
			bufX, err := hw.CreateHostCoherentBuffer(uint64(n * 4))
			if err != nil {
				errCh <- err
				return
			}
			defer bufX.Free()

			bufY, err := hw.CreateHostCoherentBuffer(uint64(n * 4))
			if err != nil {
				errCh <- err
				return
			}
			defer bufY.Free()

			input := make([]int32, n)
			for i := range input {
				input[i] = int32((i % 20) - 10)
			}
			if err := bufX.WriteInt32(input); err != nil {
				errCh <- err
				return
			}
			buffers := []*GpuBuffer{bufX, bufY}

			for it := 0; it < iterations; it++ {
				if (workerID+it)%2 == 0 {
					token, err := gpu.kernelReLUFwd.DispatchAsync(buffers, unsafe.Pointer(&push[0]), 4, wg)
					if err != nil {
						errCh <- fmt.Errorf("worker %d async dispatch failed: %w", workerID, err)
						return
					}
					if err := token.Wait(3000000000); err != nil {
						errCh <- fmt.Errorf("worker %d token wait failed: %w", workerID, err)
						return
					}
				} else {
					dur, err := gpu.kernelReLUFwd.DispatchProfiled(buffers, unsafe.Pointer(&push[0]), 4, wg)
					if err != nil {
						errCh <- fmt.Errorf("worker %d profiled dispatch failed: %w", workerID, err)
						return
					}
					if dur <= 0 {
						errCh <- fmt.Errorf("worker %d invalid profiled duration: %v", workerID, dur)
						return
					}
				}

				out, err := bufY.ReadInt32(n)
				if err != nil {
					errCh <- fmt.Errorf("worker %d ReadInt32 failed: %w", workerID, err)
					return
				}
				for i, val := range out {
					expected := input[i]
					if expected < 0 {
						expected = 0
					}
					if val != expected {
						errCh <- fmt.Errorf("worker %d output mismatch at %d: got %d, expected %d", workerID, i, val, expected)
						return
					}
				}
			}
		}()
	}

	wgWorkers.Wait()
	close(errCh)

	for err := range errCh {
		t.Fatal(err)
	}

	t.Log(">>> Concurrent Asynchronous and Profiled Dispatch Safety VERIFIED! <<<")
}
