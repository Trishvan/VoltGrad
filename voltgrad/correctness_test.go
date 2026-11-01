package voltgrad

import (
	"math"
	"math/rand"
	"reflect"
	"testing"
)

// TestShaderInt64Validation verifies that shaderInt64 is supported and enabled on the hardware device.
func TestShaderInt64Validation(t *testing.T) {
	hw, err := NewHardwareContext()
	if err != nil {
		t.Fatalf("Failed to initialize Vulkan hardware: %v", err)
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
		t.Fatalf("Failed to init hardware: %v", err)
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
		t.Fatalf("Failed to init hardware: %v", err)
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
		t.Fatalf("Failed to init hardware: %v", err)
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
		initBits := rng.Intn(40) + 2 // 2..41 bits
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
