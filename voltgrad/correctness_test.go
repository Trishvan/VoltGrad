package voltgrad

import (
	"math"
	"math/rand"
	"reflect"
	"testing"
	"unsafe"
)

// TestShaderInt64Validation verifies that shaderInt64 is supported and enabled on the hardware device.
func TestShaderInt64Validation(t *testing.T) {
	hw, err := NewHardwareContext()
	if err != nil {
		t.Skipf("Skipping Vulkan test (hardware initialization: %v)", err)
	}
	defer hw.Close()

	if !hw.HasShaderInt64() {
		t.Fatalf("Expected HasShaderInt64 to be true, but got false")
	}
	t.Log(">>> shaderInt64 is successfully queried, enabled, and validated on physical GPU! <<<")
}

// Test64BitDBWCorrectness validates that DynamicBitTensor accurately handles 33-bit through 64-bit values.
func Test64BitDBWCorrectness(t *testing.T) {
	tensor := NewDynamicBitTensor([]int{6}, 2)

	testValues := []int64{
		math.MinInt64,
		math.MaxInt64,
		int64(1) << 35,
		-(int64(1) << 45),
		int64(1) << 62,
		-(int64(1) << 62),
	}

	for i, val := range testValues {
		tensor.Set(i, val)
	}

	if tensor.BitWidth() != 64 {
		t.Fatalf("Expected bit width 64 after setting 64-bit values, got %d", tensor.BitWidth())
	}

	for i, expected := range testValues {
		actual := tensor.Get(i)
		if actual != expected {
			t.Errorf("Index %d: expected %d, got %d", i, expected, actual)
		}
	}

	// Test addition with auto-widening on 64-bit
	t64 := NewDynamicBitTensor([]int{2}, 32)
	t64.Set(0, 1<<31-1)
	newV := t64.Add(0, 1000)
	if newV != (1<<31 - 1 + 1000) {
		t.Fatalf("Add error: expected %d, got %d", 1<<31-1+1000, newV)
	}
	if t64.BitWidth() < 33 {
		t.Fatalf("Expected auto-widening past 32 bits, got %d", t64.BitWidth())
	}

	t.Log(">>> 64-bit Dynamic Bit Widening (DBW) correctness verified! <<<")
}

// TestOverflowAndBoundaryConditions validates edge cases across all bit widths from 2 to 64.
func TestOverflowAndBoundaryConditions(t *testing.T) {
	// Test each bit width representable range
	for w := 2; w <= 64; w++ {
		minVal, maxVal := MinMax64(w)

		if BitsNeeded64(minVal) > w {
			t.Errorf("Width %d: BitsNeeded64(minVal %d) = %d > %d", w, minVal, BitsNeeded64(minVal), w)
		}
		if BitsNeeded64(maxVal) > w {
			t.Errorf("Width %d: BitsNeeded64(maxVal %d) = %d > %d", w, maxVal, BitsNeeded64(maxVal), w)
		}

		if w < 64 {
			// Boundary overflow: maxVal + 1 must require w + 1
			if BitsNeeded64(maxVal+1) <= w {
				t.Errorf("Width %d: maxVal+1 (%d) should require > %d bits", w, maxVal+1, w)
			}
			// Boundary underflow: minVal - 1 must require w + 1
			if BitsNeeded64(minVal-1) <= w {
				t.Errorf("Width %d: minVal-1 (%d) should require > %d bits", w, minVal-1, w)
			}
		}
	}

	// Test byte boundary crossings (7, 8, 9, 15, 16, 17, 31, 32, 33)
	byteBoundaries := []int{7, 8, 9, 15, 16, 17, 31, 32, 33}
	for _, bw := range byteBoundaries {
		tB := NewDynamicBitTensor([]int{8}, bw)
		minV, maxV := MinMax64(bw)
		for i := 0; i < 8; i++ {
			if i%2 == 0 {
				tB.Set(i, minV)
			} else {
				tB.Set(i, maxV)
			}
		}

		for i := 0; i < 8; i++ {
			expected := minV
			if i%2 != 0 {
				expected = maxV
			}
			if tB.Get(i) != expected {
				t.Errorf("Byte boundary %d: element %d expected %d, got %d", bw, i, expected, tB.Get(i))
			}
		}
	}

	t.Log(">>> Overflow and boundary conditions verified across all bit-widths! <<<")
}

// TestGPUCPUOptimizerDiscrepancies tests SGD, Momentum, Adam-Q with Weight Decay for bit-for-bit parity.
func TestGPUCPUOptimizerDiscrepancies(t *testing.T) {
	hw, err := NewHardwareContext()
	if err != nil {
		t.Skipf("Skipping Vulkan test (hardware initialization: %v)", err)
	}
	defer hw.Close()

	gpu, err := NewGpuEngine(hw)
	if err != nil {
		t.Fatalf("Failed to init GPU engine: %v", err)
	}
	defer gpu.Close()

	optConfigs := []OptimizerConfig{
		{
			Type:            OptimizerSGD,
			LearningRateQ16: 6553, // 0.1
			MomentumBeta:    0,
			WeightDecayQ16:  655, // 0.01 weight decay
			FractionalBits:  Q16FractionalBits,
		},
		{
			Type:            OptimizerSGDMomentum,
			LearningRateQ16: 3276, // 0.05
			MomentumBeta:    58982, // 0.9
			WeightDecayQ16:  327, // 0.005 weight decay
			FractionalBits:  Q16FractionalBits,
		},
		{
			Type:            OptimizerAdamQ,
			LearningRateQ16: 3276, // 0.05
			MomentumBeta:    58982, // 0.9
			WeightDecayQ16:  0,
			FractionalBits:  Q16FractionalBits,
		},
	}

	for _, cfg := range optConfigs {
		paramGPU := NewDynamicBitTensor([]int{4}, 4)
		paramGPU.SetInt32(0, 100)
		paramGPU.SetInt32(1, -250)
		paramGPU.SetInt32(2, 500)
		paramGPU.SetInt32(3, -1000)

		paramCPU := paramGPU.Clone()

		grad := NewDynamicBitTensor([]int{4}, 16)
		grad.SetInt32(0, 32768)
		grad.SetInt32(1, -65536)
		grad.SetInt32(2, 131072)
		grad.SetInt32(3, -262144)

		velocityGPU := NewDynamicBitTensor([]int{4}, 4)
		velocityCPU := velocityGPU.Clone()

		// Run CPU
		cpuOpt := &DynamicBitOptimizer{
			config:     cfg,
			velocities: [][]*DynamicBitTensor{{velocityCPU}},
		}
		mockModel := &singleTensorModel{param: paramCPU, grad: grad}
		_, err := cpuOpt.Step(mockModel)
		if err != nil {
			t.Fatalf("CPU step failed: %v", err)
		}

		// Run GPU
		_, err = gpu.OptimizerStep(paramGPU, grad, velocityGPU, cfg)
		if err != nil {
			t.Fatalf("GPU step failed: %v", err)
		}

		cpuW := paramCPU.ToInt32Slice()
		gpuW := paramGPU.ToInt32Slice()
		if !reflect.DeepEqual(cpuW, gpuW) {
			t.Fatalf("Optimizer discrepancy for opt_type %d: CPU=%v, GPU=%v", cfg.Type, cpuW, gpuW)
		}

		cpuV := velocityCPU.ToInt32Slice()
		gpuV := velocityGPU.ToInt32Slice()
		if !reflect.DeepEqual(cpuV, gpuV) {
			t.Fatalf("Velocity discrepancy for opt_type %d: CPU=%v, GPU=%v", cfg.Type, cpuV, gpuV)
		}
	}

	t.Log(">>> GPU and CPU optimizer parity verified across SGD, Momentum, Adam-Q, and Weight Decay! <<<")
}

// TestFull64BitGPUDBWOptimizer validates that true 64-bit parameters (>32-bit values) are processed on GPU with zero truncation.
func TestFull64BitGPUDBWOptimizer(t *testing.T) {
	hw, err := NewHardwareContext()
	if err != nil {
		t.Skipf("Skipping Vulkan test (hardware initialization: %v)", err)
	}
	defer hw.Close()

	gpu, err := NewGpuEngine(hw)
	if err != nil {
		t.Fatalf("Failed to init GPU engine: %v", err)
	}
	defer gpu.Close()

	testWeights := []int64{
		int64(1) << 35,
		-(int64(1) << 45),
		math.MaxInt64 - 100000,
		math.MinInt64 + 100000,
	}

	testGrads := []int64{
		int64(1) << 30,
		-(int64(1) << 32),
		int64(100000),
		-int64(100000),
	}

	cfg := OptimizerConfig{
		Type:            OptimizerSGDMomentum,
		LearningRateQ16: 655,   // ~0.01
		MomentumBeta:    58982, // 0.9
		WeightDecayQ16:  65,    // ~0.001
		FractionalBits:  Q16FractionalBits,
	}

	paramGPU := NewDynamicBitTensor([]int{len(testWeights)}, 64)
	for i, w := range testWeights {
		paramGPU.Set(i, w)
	}
	paramCPU := paramGPU.Clone()

	gradGPU := NewDynamicBitTensor([]int{len(testGrads)}, 64)
	for i, g := range testGrads {
		gradGPU.Set(i, g)
	}
	gradCPU := gradGPU.Clone()

	velGPU := NewDynamicBitTensor([]int{len(testWeights)}, 64)
	velCPU := velGPU.Clone()

	// CPU step
	cpuOpt := &DynamicBitOptimizer{
		config:     cfg,
		velocities: [][]*DynamicBitTensor{{velCPU}},
	}
	mockModel := &singleTensorModel{param: paramCPU, grad: gradCPU}
	_, err = cpuOpt.Step(mockModel)
	if err != nil {
		t.Fatalf("CPU step failed: %v", err)
	}

	// GPU step
	targetBits, err := gpu.OptimizerStep(paramGPU, gradGPU, velGPU, cfg)
	if err != nil {
		t.Fatalf("GPU OptimizerStep failed: %v", err)
	}

	// Verify bit widths
	if paramGPU.BitWidth() != paramCPU.BitWidth() {
		t.Errorf("Bit width mismatch: GPU=%d, CPU=%d (targetBits=%d)", paramGPU.BitWidth(), paramCPU.BitWidth(), targetBits)
	}

	// Verify 64-bit parameter values
	for i := 0; i < len(testWeights); i++ {
		cpuVal := paramCPU.Get(i)
		gpuVal := paramGPU.Get(i)
		if cpuVal != gpuVal {
			t.Errorf("Index %d parameter mismatch: CPU=%d, GPU=%d", i, cpuVal, gpuVal)
		}

		cpuV := velCPU.Get(i)
		gpuV := velGPU.Get(i)
		if cpuV != gpuV {
			t.Errorf("Index %d velocity mismatch: CPU=%d, GPU=%d", i, cpuV, gpuV)
		}
	}

	t.Log(">>> Full 64-bit GPU DBW parameter & velocity updates verified bit-for-bit with CPU! <<<")
}

// TestRandomizedGPUCPUParity verifies randomized tensors, bit-widths, and optimizer updates match between GPU and CPU.
func TestRandomizedGPUCPUParity(t *testing.T) {
	hw, err := NewHardwareContext()
	if err != nil {
		t.Skipf("Skipping Vulkan test (hardware initialization: %v)", err)
	}
	defer hw.Close()

	gpu, err := NewGpuEngine(hw)
	if err != nil {
		t.Fatalf("Failed to init GPU engine: %v", err)
	}
	defer gpu.Close()

	rng := rand.New(rand.NewSource(42))
	optTypes := []OptimizerType{OptimizerSGD, OptimizerSGDMomentum, OptimizerAdamQ}

	for trial := 0; trial < 10; trial++ {
		size := rng.Intn(64) + 8     // 8..71 elements
		initBits := rng.Intn(63) + 2 // 2..64 bits (full arbitrary precision spectrum)
		minV, maxV := MinMax64(initBits)
		span := maxV - minV
		if span <= 0 || span > math.MaxInt64/2 {
			span = 1000000
		}

		paramGPU := NewDynamicBitTensor([]int{size}, initBits)
		gradGPU := NewDynamicBitTensor([]int{size}, initBits)
		velGPU := NewDynamicBitTensor([]int{size}, initBits)

		for i := 0; i < size; i++ {
			v := minV + (rng.Int63() % span)
			paramGPU.Set(i, v)
			g := (rng.Int63() % 65536) - 32768
			gradGPU.Set(i, g)
		}

		paramCPU := paramGPU.Clone()
		gradCPU := gradGPU.Clone()
		velCPU := velGPU.Clone()

		cfg := OptimizerConfig{
			Type:            optTypes[rng.Intn(len(optTypes))],
			LearningRateQ16: int32(rng.Intn(6553) + 100),
			MomentumBeta:    int32(rng.Intn(30000) + 30000),
			WeightDecayQ16:  int32(rng.Intn(1000)),
			FractionalBits:  Q16FractionalBits,
		}

		// CPU step
		cpuOpt := &DynamicBitOptimizer{
			config:     cfg,
			velocities: [][]*DynamicBitTensor{{velCPU}},
		}
		mockModel := &singleTensorModel{param: paramCPU, grad: gradCPU}
		_, err = cpuOpt.Step(mockModel)
		if err != nil {
			t.Fatalf("Trial %d: CPU step failed: %v", trial, err)
		}

		// GPU step
		_, err = gpu.OptimizerStep(paramGPU, gradGPU, velGPU, cfg)
		if err != nil {
			t.Fatalf("Trial %d: GPU OptimizerStep failed: %v", trial, err)
		}

		// Compare bit widths
		if paramGPU.BitWidth() != paramCPU.BitWidth() {
			t.Fatalf("Trial %d: BitWidth mismatch: CPU=%d, GPU=%d", trial, paramCPU.BitWidth(), paramGPU.BitWidth())
		}

		// Compare values
		for i := 0; i < size; i++ {
			if paramCPU.Get(i) != paramGPU.Get(i) {
				t.Fatalf("Trial %d: Param[%d] mismatch: CPU=%d, GPU=%d", trial, i, paramCPU.Get(i), paramGPU.Get(i))
			}
			if velCPU.Get(i) != velGPU.Get(i) {
				t.Fatalf("Trial %d: Velocity[%d] mismatch: CPU=%d, GPU=%d", trial, i, velCPU.Get(i), velGPU.Get(i))
			}
		}
	}

	t.Log(">>> Randomized GPU/CPU parity verified across all randomized trials! <<<")
}

// TestFull64BitMatMulForwardBackwardParity verifies forward and backward passes with >32-bit weights and overflow clamping.
func TestFull64BitMatMulForwardBackwardParity(t *testing.T) {
	hw, err := NewHardwareContext()
	if err != nil {
		t.Skipf("Skipping Vulkan test (hardware initialization: %v)", err)
	}
	defer hw.Close()

	gpu, err := NewGpuEngine(hw)
	if err != nil {
		t.Fatalf("Failed to init GPU engine: %v", err)
	}
	defer gpu.Close()

	inDim := 4
	outDim := 3
	shift := uint(16)

	// Weights with true 64-bit values (>32-bit range)
	wWeights := []int64{
		int64(1) << 35, -(int64(1) << 35), int64(1) << 40, -(int64(1) << 40),
		int64(1) << 45, -(int64(1) << 45), int64(1) << 50, -(int64(1) << 50),
		math.MaxInt64 / 2, -(math.MaxInt64 / 2), 65536, -65536,
	}

	w := NewDynamicBitTensor([]int{outDim, inDim}, 64)
	for i, val := range wWeights {
		w.Set(i, val)
	}

	bias := NewDynamicBitTensor([]int{outDim}, 64)
	bias.Set(0, 1000)
	bias.Set(1, -2000)
	bias.Set(2, 3000)

	x := []int32{65536, -32768, 16384, -8192}
	dOut := []int32{32768, -16384, 8192}

	// 1. Forward Pass
	cpuOut, err := MatMulVecWithBias64(w, x, bias, shift)
	if err != nil {
		t.Fatalf("CPU MatMulVecWithBias64 failed: %v", err)
	}

	gpuOut, err := gpu.MatMulForward(w, x, bias, shift)
	if err != nil {
		t.Fatalf("GPU MatMulForward failed: %v", err)
	}

	if !reflect.DeepEqual(cpuOut, gpuOut) {
		t.Fatalf("MatMulForward 64-bit mismatch: CPU=%v, GPU=%v", cpuOut, gpuOut)
	}

	// 2. Backward Pass
	gradW_CPU := NewDynamicBitTensor([]int{outDim, inDim}, 64)
	gradW_GPU := NewDynamicBitTensor([]int{outDim, inDim}, 64)
	gradBias_CPU := NewDynamicBitTensor([]int{outDim}, 64)
	gradBias_GPU := NewDynamicBitTensor([]int{outDim}, 64)

	cpuDX, err := MatMulVecBackward(dOut, x, w, gradW_CPU, shift)
	if err != nil {
		t.Fatalf("CPU MatMulVecBackward failed: %v", err)
	}
	for i := 0; i < outDim; i++ {
		gradBias_CPU.Add(i, int64(dOut[i]))
	}

	gpuDX, err := gpu.MatMulBackward(dOut, x, w, gradW_GPU, gradBias_GPU, shift)
	if err != nil {
		t.Fatalf("GPU MatMulBackward failed: %v", err)
	}

	if !reflect.DeepEqual(cpuDX, gpuDX) {
		t.Fatalf("MatMulBackward dX mismatch: CPU=%v, GPU=%v", cpuDX, gpuDX)
	}

	for i := 0; i < outDim*inDim; i++ {
		if gradW_CPU.Get(i) != gradW_GPU.Get(i) {
			t.Fatalf("gradW[%d] mismatch: CPU=%d, GPU=%d", i, gradW_CPU.Get(i), gradW_GPU.Get(i))
		}
	}
	for i := 0; i < outDim; i++ {
		if gradBias_CPU.Get(i) != gradBias_GPU.Get(i) {
			t.Fatalf("gradBias[%d] mismatch: CPU=%d, GPU=%d", i, gradBias_CPU.Get(i), gradBias_GPU.Get(i))
		}
	}

	t.Log(">>> Full 64-bit MatMul Forward & Backward parity verified bit-for-bit! <<<")
}

// TestRandomizedMatMulForwardBackwardParity validates random dimension and 2..64 bit-width parity.
func TestRandomizedMatMulForwardBackwardParity(t *testing.T) {
	hw, err := NewHardwareContext()
	if err != nil {
		t.Skipf("Skipping Vulkan test (hardware initialization: %v)", err)
	}
	defer hw.Close()

	gpu, err := NewGpuEngine(hw)
	if err != nil {
		t.Fatalf("Failed to init GPU engine: %v", err)
	}
	defer gpu.Close()

	rng := rand.New(rand.NewSource(123))

	for trial := 0; trial < 10; trial++ {
		inDim := rng.Intn(16) + 2
		outDim := rng.Intn(16) + 2
		shift := uint(rng.Intn(16) + 1)
		bitWidth := rng.Intn(63) + 2 // 2..64 bits

		minV, maxV := MinMax64(bitWidth)
		span := maxV - minV
		if span <= 0 || span > math.MaxInt64/2 {
			span = 1000000
		}

		w := NewDynamicBitTensor([]int{outDim, inDim}, bitWidth)
		for i := 0; i < outDim*inDim; i++ {
			w.Set(i, minV+(rng.Int63()%span))
		}

		bias := NewDynamicBitTensor([]int{outDim}, bitWidth)
		for i := 0; i < outDim; i++ {
			bias.Set(i, minV+(rng.Int63()%span))
		}

		x := make([]int32, inDim)
		for i := 0; i < inDim; i++ {
			x[i] = int32((rng.Int63() % 65536) - 32768)
		}

		dOut := make([]int32, outDim)
		for i := 0; i < outDim; i++ {
			dOut[i] = int32((rng.Int63() % 65536) - 32768)
		}

		// Forward
		cpuOut, err := MatMulVecWithBias64(w, x, bias, shift)
		if err != nil {
			t.Fatalf("Trial %d: CPU MatMulVecWithBias64 failed: %v", trial, err)
		}

		gpuOut, err := gpu.MatMulForward(w, x, bias, shift)
		if err != nil {
			t.Fatalf("Trial %d: GPU MatMulForward failed: %v", trial, err)
		}

		if !reflect.DeepEqual(cpuOut, gpuOut) {
			t.Fatalf("Trial %d: Forward mismatch: CPU=%v, GPU=%v", trial, cpuOut, gpuOut)
		}

		// Backward
		gradW_CPU := NewDynamicBitTensor([]int{outDim, inDim}, bitWidth)
		gradW_GPU := NewDynamicBitTensor([]int{outDim, inDim}, bitWidth)
		gradBias_CPU := NewDynamicBitTensor([]int{outDim}, bitWidth)
		gradBias_GPU := NewDynamicBitTensor([]int{outDim}, bitWidth)

		cpuDX, err := MatMulVecBackward(dOut, x, w, gradW_CPU, shift)
		if err != nil {
			t.Fatalf("Trial %d: CPU MatMulVecBackward failed: %v", trial, err)
		}
		for i := 0; i < outDim; i++ {
			gradBias_CPU.Add(i, int64(dOut[i]))
		}

		gpuDX, err := gpu.MatMulBackward(dOut, x, w, gradW_GPU, gradBias_GPU, shift)
		if err != nil {
			t.Fatalf("Trial %d: GPU MatMulBackward failed: %v", trial, err)
		}

		if !reflect.DeepEqual(cpuDX, gpuDX) {
			t.Fatalf("Trial %d: Backward dX mismatch: CPU=%v, GPU=%v", trial, cpuDX, gpuDX)
		}

		for i := 0; i < outDim*inDim; i++ {
			if gradW_CPU.Get(i) != gradW_GPU.Get(i) {
				t.Fatalf("Trial %d: gradW[%d] mismatch: CPU=%d, GPU=%d", trial, i, gradW_CPU.Get(i), gradW_GPU.Get(i))
			}
		}
		for i := 0; i < outDim; i++ {
			if gradBias_CPU.Get(i) != gradBias_GPU.Get(i) {
				t.Fatalf("Trial %d: gradBias[%d] mismatch: CPU=%d, GPU=%d", trial, i, gradBias_CPU.Get(i), gradBias_GPU.Get(i))
			}
		}
	}

	t.Log(">>> Randomized 2..64-bit MatMul Forward & Backward parity verified! <<<")
}

// TestRandomizedDBWRepackParity verifies bitstream packing on GPU against CPU across all widths 2..64.
func TestRandomizedDBWRepackParity(t *testing.T) {
	hw, err := NewHardwareContext()
	if err != nil {
		t.Skipf("Skipping Vulkan test (hardware initialization: %v)", err)
	}
	defer hw.Close()

	gpu, err := NewGpuEngine(hw)
	if err != nil {
		t.Fatalf("Failed to init GPU engine: %v", err)
	}
	defer gpu.Close()

	rng := rand.New(rand.NewSource(999))
	count := 25

	for w := 2; w <= 64; w++ {
		minV, maxV := MinMax64(w)
		span := maxV - minV
		if span <= 0 || span > math.MaxInt64/2 {
			span = 1000000
		}

		values := make([]int64, count)
		tensorCPU := NewDynamicBitTensor([]int{count}, w)
		for i := 0; i < count; i++ {
			val := minV + (rng.Int63() % span)
			values[i] = val
			tensorCPU.Set(i, val)
		}

		// GPU repack
		bufWeights, err := hw.CreateHostCoherentBuffer(uint64(count * 8))
		if err != nil {
			t.Fatalf("CreateHostCoherentBuffer failed: %v", err)
		}
		if err := bufWeights.WriteInt64(values); err != nil {
			bufWeights.Free()
			t.Fatalf("WriteInt64 failed: %v", err)
		}

		totalBits := count * w
		numWords := (totalBits + 31) / 32
		if numWords == 0 {
			numWords = 1
		}
		bufPacked, err := hw.CreateHostCoherentBuffer(uint64(numWords * 4))
		if err != nil {
			bufWeights.Free()
			t.Fatalf("CreateHostCoherentBuffer packed failed: %v", err)
		}
		bufPacked.Clear()

		pushRepack := [2]uint32{uint32(count), uint32(w)}
		wg := uint32((count + 63) / 64)
		if err := gpu.kernelRepack.Dispatch([]*GpuBuffer{bufWeights, bufPacked}, unsafe.Pointer(&pushRepack[0]), 8, wg); err != nil {
			bufWeights.Free()
			bufPacked.Free()
			t.Fatalf("Width %d repack dispatch failed: %v", w, err)
		}

		gpuWords, err := bufPacked.ReadUint32(numWords)
		if err != nil {
			bufWeights.Free()
			bufPacked.Free()
			t.Fatalf("ReadUint32 failed: %v", err)
		}
		bufWeights.Free()
		bufPacked.Free()

		tensorGPU := NewDynamicBitTensor([]int{count}, w)
		tensorGPU.SetWords(gpuWords, w)

		for i := 0; i < count; i++ {
			if tensorCPU.Get(i) != tensorGPU.Get(i) {
				t.Fatalf("Width %d element %d mismatch: CPU=%d, GPU=%d", w, i, tensorCPU.Get(i), tensorGPU.Get(i))
			}
		}
	}

	t.Log(">>> Randomized DBW Repack parity verified across all widths 2..64! <<<")
}

// TestTrainerConfigValidation verifies proper validation of training configurations.
func TestTrainerConfigValidation(t *testing.T) {
	mockModel := &singleTensorModel{
		param: NewDynamicBitTensor([]int{4}, 2),
		grad:  NewDynamicBitTensor([]int{4}, 2),
	}

	// 1. Invalid BatchSize <= 0
	cfg := DefaultTrainingConfig()
	cfg.BatchSize = 0
	_, err := NewTrainingPipeline(cfg, mockModel)
	if err == nil {
		t.Fatalf("Expected error for BatchSize=0, got nil")
	}

	cfg.BatchSize = -5
	_, err = NewTrainingPipeline(cfg, mockModel)
	if err == nil {
		t.Fatalf("Expected error for BatchSize=-5, got nil")
	}

	// 2. Invalid Epochs <= 0
	cfg = DefaultTrainingConfig()
	cfg.Epochs = 0
	_, err = NewTrainingPipeline(cfg, mockModel)
	if err == nil {
		t.Fatalf("Expected error for Epochs=0, got nil")
	}

	// 3. Nil model
	cfg = DefaultTrainingConfig()
	_, err = NewTrainingPipeline(cfg, nil)
	if err == nil {
		t.Fatalf("Expected error for nil model, got nil")
	}

	// 4. Invalid LR <= 0
	cfg = DefaultTrainingConfig()
	cfg.OptimizerConfig.LearningRateQ16 = 0
	_, err = NewTrainingPipeline(cfg, mockModel)
	if err == nil {
		t.Fatalf("Expected error for LR=0, got nil")
	}

	t.Log(">>> Trainer configuration validation verified! <<<")
}

// TestMatMulBackwardInputValidation verifies robust dimension and nil pointer checks.
func TestMatMulBackwardInputValidation(t *testing.T) {
	hw, err := NewHardwareContext()
	if err != nil {
		t.Skipf("Skipping Vulkan test (hardware initialization: %v)", err)
	}
	defer hw.Close()

	gpu, err := NewGpuEngine(hw)
	if err != nil {
		t.Fatalf("Failed to init GPU engine: %v", err)
	}
	defer gpu.Close()

	w := NewDynamicBitTensor([]int{3, 4}, 16)
	gradW := NewDynamicBitTensor([]int{3, 4}, 16)
	gradBias := NewDynamicBitTensor([]int{3}, 16)
	dOut := make([]int32, 3)
	x := make([]int32, 4)

	// 1. Nil w
	if _, err := gpu.MatMulBackward(dOut, x, nil, gradW, gradBias, 16); err == nil {
		t.Fatalf("Expected error for nil w, got nil")
	}

	// 2. Nil gradW
	if _, err := gpu.MatMulBackward(dOut, x, w, nil, gradBias, 16); err == nil {
		t.Fatalf("Expected error for nil gradW, got nil")
	}

	// 3. Mismatched dOut length
	if _, err := gpu.MatMulBackward(make([]int32, 5), x, w, gradW, gradBias, 16); err == nil {
		t.Fatalf("Expected error for mismatched dOut length, got nil")
	}

	// 4. Mismatched x length
	if _, err := gpu.MatMulBackward(dOut, make([]int32, 2), w, gradW, gradBias, 16); err == nil {
		t.Fatalf("Expected error for mismatched x length, got nil")
	}

	// 5. Mismatched gradW length
	wrongGradW := NewDynamicBitTensor([]int{2, 2}, 16)
	if _, err := gpu.MatMulBackward(dOut, x, w, wrongGradW, gradBias, 16); err == nil {
		t.Fatalf("Expected error for mismatched gradW length, got nil")
	}

	// 6. Mismatched gradBias length
	wrongGradBias := NewDynamicBitTensor([]int{5}, 16)
	if _, err := gpu.MatMulBackward(dOut, x, w, gradW, wrongGradBias, 16); err == nil {
		t.Fatalf("Expected error for mismatched gradBias length, got nil")
	}

	// 7. Non-2D weight tensor
	flatW := NewDynamicBitTensor([]int{12}, 16)
	if _, err := gpu.MatMulBackward(dOut, x, flatW, gradW, gradBias, 16); err == nil {
		t.Fatalf("Expected error for 1D weight tensor, got nil")
	}

	t.Log(">>> MatMulBackward input validation verified! <<<")
}
