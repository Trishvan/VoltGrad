package voltgrad

import (
	_ "embed"
	"fmt"
	"sync"
	"time"
	"unsafe"
)

//go:embed shaders/matmul_forward.spv
var matmulForwardSPV []byte

//go:embed shaders/matmul_backward_dx.spv
var matmulBackwardDxSPV []byte

//go:embed shaders/matmul_backward_dw.spv
var matmulBackwardDwSPV []byte

//go:embed shaders/matmul_batched_forward.spv
var matmulBatchedForwardSPV []byte

//go:embed shaders/matmul_batched_backward_dx.spv
var matmulBatchedBackwardDxSPV []byte

//go:embed shaders/matmul_batched_backward_dw.spv
var matmulBatchedBackwardDwSPV []byte

//go:embed shaders/relu_forward.spv
var reluForwardSPV []byte

//go:embed shaders/relu_backward.spv
var reluBackwardSPV []byte

//go:embed shaders/loss_mse.spv
var lossMseSPV []byte

//go:embed shaders/optimizer_step.spv
var optimizerStepSPV []byte

//go:embed shaders/repack_dbw.spv
var repackDbwSPV []byte

// ComputeKernel manages a Vulkan compute pipeline for a specific SPIR-V shader.
type ComputeKernel struct {
	hw           *HardwareContext
	shaderModule uint64
	descLayout   uint64
	pipeLayout   uint64
	pipeline     uint64
	descPool     uint64
	descSet      uint64
	cmdPool      uint64
	cmdBuf       uintptr
	fence        uint64
	queryPool    uint64
	numBindings  int
	pushSize     uint32
	inFlight     bool
	mu           sync.Mutex
}

// NewComputeKernel compiles SPIR-V bytecode into a Vulkan compute pipeline with specified storage bindings and push constant size.
func NewComputeKernel(hw *HardwareContext, spv []byte, numBindings int, pushSize uint32) (*ComputeKernel, error) {
	if len(spv) == 0 {
		return nil, fmt.Errorf("spv bytecode cannot be empty")
	}

	shaderCI := VkShaderModuleCreateInfo{
		SType:    VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		CodeSize: uintptr(len(spv)),
		PCode:    (*uint32)(unsafe.Pointer(&spv[0])),
	}
	var shaderModule uint64
	r := vkSyscall(hw.getProc("vkCreateShaderModule"), hw.device, uintptr(unsafe.Pointer(&shaderCI)), 0, uintptr(unsafe.Pointer(&shaderModule)))
	if r != 0 {
		return nil, fmt.Errorf("vkCreateShaderModule failed: %d", r)
	}

	// Storage buffer bindings
	bindings := make([]VkDescriptorSetLayoutBinding, numBindings)
	for i := 0; i < numBindings; i++ {
		bindings[i] = VkDescriptorSetLayoutBinding{
			Binding:         uint32(i),
			DescriptorType:  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			DescriptorCount: 1,
			StageFlags:      VK_SHADER_STAGE_COMPUTE_BIT,
		}
	}
	dslCI := VkDescriptorSetLayoutCreateInfo{
		SType:        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		BindingCount: uint32(numBindings),
		PBindings:    &bindings[0],
	}
	var descLayout uint64
	r = vkSyscall(hw.getProc("vkCreateDescriptorSetLayout"), hw.device, uintptr(unsafe.Pointer(&dslCI)), 0, uintptr(unsafe.Pointer(&descLayout)))
	if r != 0 {
		vkSyscall(hw.getProc("vkDestroyShaderModule"), hw.device, uintptr(shaderModule), 0)
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout failed: %d", r)
	}

	var pipeLayout uint64
	if pushSize > 0 {
		pushConst := VkPushConstantRange{
			StageFlags: VK_SHADER_STAGE_COMPUTE_BIT,
			Offset:     0,
			Size:       pushSize,
		}
		pipeLayoutCI := VkPipelineLayoutCreateInfo{
			SType:                  VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			SetLayoutCount:         1,
			PSetLayouts:            &descLayout,
			PushConstantRangeCount: 1,
			PPushConstantRanges:    &pushConst,
		}
		r = vkSyscall(hw.getProc("vkCreatePipelineLayout"), hw.device, uintptr(unsafe.Pointer(&pipeLayoutCI)), 0, uintptr(unsafe.Pointer(&pipeLayout)))
	} else {
		pipeLayoutCI := VkPipelineLayoutCreateInfo{
			SType:          VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			SetLayoutCount: 1,
			PSetLayouts:    &descLayout,
		}
		r = vkSyscall(hw.getProc("vkCreatePipelineLayout"), hw.device, uintptr(unsafe.Pointer(&pipeLayoutCI)), 0, uintptr(unsafe.Pointer(&pipeLayout)))
	}
	if r != 0 {
		vkSyscall(hw.getProc("vkDestroyDescriptorSetLayout"), hw.device, uintptr(descLayout), 0)
		vkSyscall(hw.getProc("vkDestroyShaderModule"), hw.device, uintptr(shaderModule), 0)
		return nil, fmt.Errorf("vkCreatePipelineLayout failed: %d", r)
	}

	mainEntry := append([]byte("main"), 0)
	pipeCI := VkComputePipelineCreateInfo{
		SType: VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		Stage: VkPipelineShaderStageCreateInfo{
			SType:  VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			Stage:  VK_SHADER_STAGE_COMPUTE_BIT,
			Module: shaderModule,
			PName:  &mainEntry[0],
		},
		Layout: pipeLayout,
	}
	var compPipeline uint64
	r = vkSyscall(hw.getProc("vkCreateComputePipelines"), hw.device, 0, 1, uintptr(unsafe.Pointer(&pipeCI)), 0, uintptr(unsafe.Pointer(&compPipeline)))
	if r != 0 {
		vkSyscall(hw.getProc("vkDestroyPipelineLayout"), hw.device, uintptr(pipeLayout), 0)
		vkSyscall(hw.getProc("vkDestroyDescriptorSetLayout"), hw.device, uintptr(descLayout), 0)
		vkSyscall(hw.getProc("vkDestroyShaderModule"), hw.device, uintptr(shaderModule), 0)
		return nil, fmt.Errorf("vkCreateComputePipelines failed: %d", r)
	}

	var descPool uint64
	var cmdPool uint64
	var fence uint64

	cleanup := func() {
		if fence != 0 {
			vkSyscall(hw.getProc("vkDestroyFence"), hw.device, uintptr(fence), 0)
		}
		if cmdPool != 0 {
			vkSyscall(hw.getProc("vkDestroyCommandPool"), hw.device, uintptr(cmdPool), 0)
		}
		if descPool != 0 {
			vkSyscall(hw.getProc("vkDestroyDescriptorPool"), hw.device, uintptr(descPool), 0)
		}
		if compPipeline != 0 {
			vkSyscall(hw.getProc("vkDestroyPipeline"), hw.device, uintptr(compPipeline), 0)
		}
		if pipeLayout != 0 {
			vkSyscall(hw.getProc("vkDestroyPipelineLayout"), hw.device, uintptr(pipeLayout), 0)
		}
		if descLayout != 0 {
			vkSyscall(hw.getProc("vkDestroyDescriptorSetLayout"), hw.device, uintptr(descLayout), 0)
		}
		if shaderModule != 0 {
			vkSyscall(hw.getProc("vkDestroyShaderModule"), hw.device, uintptr(shaderModule), 0)
		}
	}

	poolSize := VkDescriptorPoolSize{
		Type:            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		DescriptorCount: uint32(numBindings),
	}
	poolCI := VkDescriptorPoolCreateInfo{
		SType:         VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		MaxSets:       1,
		PoolSizeCount: 1,
		PPoolSizes:    &poolSize,
	}
	r = vkSyscall(hw.getProc("vkCreateDescriptorPool"), hw.device, uintptr(unsafe.Pointer(&poolCI)), 0, uintptr(unsafe.Pointer(&descPool)))
	if r != 0 {
		cleanup()
		return nil, fmt.Errorf("vkCreateDescriptorPool failed: %d", r)
	}

	allocSet := VkDescriptorSetAllocateInfo{
		SType:              VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		DescriptorPool:     descPool,
		DescriptorSetCount: 1,
		PSetLayouts:        &descLayout,
	}
	var descSet uint64
	r = vkSyscall(hw.getProc("vkAllocateDescriptorSets"), hw.device, uintptr(unsafe.Pointer(&allocSet)), uintptr(unsafe.Pointer(&descSet)))
	if r != 0 {
		cleanup()
		return nil, fmt.Errorf("vkAllocateDescriptorSets failed: %d", r)
	}

	cmdPoolCI := VkCommandPoolCreateInfo{
		SType:            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		Flags:            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		QueueFamilyIndex: hw.queueFam,
	}
	r = vkSyscall(hw.getProc("vkCreateCommandPool"), hw.device, uintptr(unsafe.Pointer(&cmdPoolCI)), 0, uintptr(unsafe.Pointer(&cmdPool)))
	if r != 0 {
		cleanup()
		return nil, fmt.Errorf("vkCreateCommandPool failed: %d", r)
	}

	cmdAlloc := VkCommandBufferAllocateInfo{
		SType:              VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		CommandPool:        cmdPool,
		Level:              VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		CommandBufferCount: 1,
	}
	var cmdBuf uintptr
	r = vkSyscall(hw.getProc("vkAllocateCommandBuffers"), hw.device, uintptr(unsafe.Pointer(&cmdAlloc)), uintptr(unsafe.Pointer(&cmdBuf)))
	if r != 0 {
		cleanup()
		return nil, fmt.Errorf("vkAllocateCommandBuffers failed: %d", r)
	}

	fenceCI := VkFenceCreateInfo{SType: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}
	r = vkSyscall(hw.getProc("vkCreateFence"), hw.device, uintptr(unsafe.Pointer(&fenceCI)), 0, uintptr(unsafe.Pointer(&fence)))
	if r != 0 {
		cleanup()
		return nil, fmt.Errorf("vkCreateFence failed: %d", r)
	}

	var queryPool uint64
	createQP := hw.getProc("vkCreateQueryPool")
	if createQP != 0 {
		qpCI := struct {
			SType              uint32
			PNext              uintptr
			Flags              uint32
			QueryType          uint32
			QueryCount         uint32
			PipelineStatistics uint32
		}{
			SType:      12, // VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO
			QueryType:  2,  // VK_QUERY_TYPE_TIMESTAMP
			QueryCount: 2,
		}
		vkSyscall(createQP, hw.device, uintptr(unsafe.Pointer(&qpCI)), 0, uintptr(unsafe.Pointer(&queryPool)))
	}

	return &ComputeKernel{
		hw:           hw,
		shaderModule: shaderModule,
		descLayout:   descLayout,
		pipeLayout:   pipeLayout,
		pipeline:     compPipeline,
		descPool:     descPool,
		descSet:      descSet,
		cmdPool:      cmdPool,
		cmdBuf:       cmdBuf,
		fence:        fence,
		queryPool:    queryPool,
		numBindings:  numBindings,
		pushSize:     pushSize,
	}, nil
}

// ExecutionToken represents an in-flight asynchronous GPU compute submission.
type ExecutionToken struct {
	kernel    *ComputeKernel
	completed bool
	mu        sync.Mutex
}

// Wait blocks until the asynchronous GPU compute operation completes or the timeout expires.
// Pass timeoutNs in nanoseconds (e.g. 2_000_000_000 for 2 seconds).
func (t *ExecutionToken) Wait(timeoutNs uint64) error {
	if t == nil || t.kernel == nil {
		return nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.completed {
		return nil
	}

	r := vkSyscall(t.kernel.hw.getProc("vkWaitForFences"), t.kernel.hw.device, 1, uintptr(unsafe.Pointer(&t.kernel.fence)), 1, uintptr(timeoutNs))
	if r != 0 {
		return fmt.Errorf("GPU compute execution failed or timed out: %d", r)
	}

	t.completed = true
	t.kernel.mu.Lock()
	t.kernel.inFlight = false
	t.kernel.mu.Unlock()
	return nil
}

// DispatchAsync records and submits the compute dispatch asynchronously to the GPU compute queue,
// returning an ExecutionToken without blocking the CPU thread.
func (k *ComputeKernel) DispatchAsync(buffers []*GpuBuffer, pushData unsafe.Pointer, pushSize uint32, workgroups uint32) (*ExecutionToken, error) {
	if len(buffers) != k.numBindings {
		return nil, fmt.Errorf("kernel requires %d buffers, got %d", k.numBindings, len(buffers))
	}
	if workgroups == 0 {
		workgroups = 1
	}

	k.mu.Lock()
	defer k.mu.Unlock()

	// If a previous asynchronous dispatch is still in-flight on this kernel, await completion first
	if k.inFlight {
		r := vkSyscall(k.hw.getProc("vkWaitForFences"), k.hw.device, 1, uintptr(unsafe.Pointer(&k.fence)), 1, uintptr(2000000000))
		if r != 0 {
			return nil, fmt.Errorf("previous asynchronous kernel execution still pending or timed out: %d", r)
		}
		k.inFlight = false
	}

	// Update descriptor sets
	writes := make([]VkWriteDescriptorSet, k.numBindings)
	bufInfos := make([]VkDescriptorBufferInfo, k.numBindings)
	for i, b := range buffers {
		bufInfos[i] = VkDescriptorBufferInfo{
			Buffer: b.buffer,
			Offset: 0,
			Range:  b.size,
		}
		writes[i] = VkWriteDescriptorSet{
			SType:           VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			DstSet:          k.descSet,
			DstBinding:      uint32(i),
			DescriptorCount: 1,
			DescriptorType:  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			PBufferInfo:     &bufInfos[i],
		}
	}
	vkSyscall(k.hw.getProc("vkUpdateDescriptorSets"), k.hw.device, uintptr(k.numBindings), uintptr(unsafe.Pointer(&writes[0])), 0, 0)

	// Reset fence
	r := vkSyscall(k.hw.getProc("vkResetFences"), k.hw.device, 1, uintptr(unsafe.Pointer(&k.fence)))
	if r != 0 {
		return nil, fmt.Errorf("vkResetFences failed: %d", r)
	}

	// Record command buffer
	beginInfo := VkCommandBufferBeginInfo{
		SType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		Flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	}
	r = vkSyscall(k.hw.getProc("vkBeginCommandBuffer"), k.cmdBuf, uintptr(unsafe.Pointer(&beginInfo)))
	if r != 0 {
		return nil, fmt.Errorf("vkBeginCommandBuffer failed: %d", r)
	}

	vkSyscall(k.hw.getProc("vkCmdBindPipeline"), k.cmdBuf, uintptr(VK_PIPELINE_BIND_POINT_COMPUTE), uintptr(k.pipeline))
	vkSyscall(k.hw.getProc("vkCmdBindDescriptorSets"), k.cmdBuf, uintptr(VK_PIPELINE_BIND_POINT_COMPUTE), uintptr(k.pipeLayout), 0, 1, uintptr(unsafe.Pointer(&k.descSet)), 0, 0)

	if pushData != nil && pushSize > 0 {
		vkSyscall(k.hw.getProc("vkCmdPushConstants"), k.cmdBuf, uintptr(k.pipeLayout), uintptr(VK_SHADER_STAGE_COMPUTE_BIT), 0, uintptr(pushSize), uintptr(pushData))
	}

	vkSyscall(k.hw.getProc("vkCmdDispatch"), k.cmdBuf, uintptr(workgroups), 1, 1)
	r = vkSyscall(k.hw.getProc("vkEndCommandBuffer"), k.cmdBuf)
	if r != 0 {
		return nil, fmt.Errorf("vkEndCommandBuffer failed: %d", r)
	}

	submitInfo := VkSubmitInfo{
		SType:              VK_STRUCTURE_TYPE_SUBMIT_INFO,
		CommandBufferCount: 1,
		PCommandBuffers:    &k.cmdBuf,
	}
	r = vkSyscall(k.hw.getProc("vkQueueSubmit"), k.hw.queue, 1, uintptr(unsafe.Pointer(&submitInfo)), uintptr(k.fence))
	if r != 0 {
		return nil, fmt.Errorf("vkQueueSubmit failed: %d", r)
	}

	k.inFlight = true
	return &ExecutionToken{kernel: k}, nil
}

// Dispatch synchronously executes the compute shader across the given storage buffers with push constants.
func (k *ComputeKernel) Dispatch(buffers []*GpuBuffer, pushData unsafe.Pointer, pushSize uint32, workgroups uint32) error {
	token, err := k.DispatchAsync(buffers, pushData, pushSize, workgroups)
	if err != nil {
		return err
	}
	return token.Wait(2000000000)
}

// DispatchProfiled executes the compute shader and returns the execution duration.
// It leverages Vulkan GPU query pools when supported, with fallback to high-precision timers.
func (k *ComputeKernel) DispatchProfiled(buffers []*GpuBuffer, pushData unsafe.Pointer, pushSize uint32, workgroups uint32) (time.Duration, error) {
	if k.queryPool != 0 {
		cmdResetQP := k.hw.getProc("vkCmdResetQueryPool")
		cmdWriteTS := k.hw.getProc("vkCmdWriteTimestamp")
		getQPResults := k.hw.getProc("vkGetQueryPoolResults")

		if cmdResetQP != 0 && cmdWriteTS != 0 && getQPResults != 0 {
			if len(buffers) != k.numBindings {
				return 0, fmt.Errorf("kernel requires %d buffers, got %d", k.numBindings, len(buffers))
			}
			if workgroups == 0 {
				workgroups = 1
			}

			writes := make([]VkWriteDescriptorSet, k.numBindings)
			bufInfos := make([]VkDescriptorBufferInfo, k.numBindings)
			for i, b := range buffers {
				bufInfos[i] = VkDescriptorBufferInfo{Buffer: b.buffer, Offset: 0, Range: b.size}
				writes[i] = VkWriteDescriptorSet{
					SType:           VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					DstSet:          k.descSet,
					DstBinding:      uint32(i),
					DescriptorCount: 1,
					DescriptorType:  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					PBufferInfo:     &bufInfos[i],
				}
			}
			vkSyscall(k.hw.getProc("vkUpdateDescriptorSets"), k.hw.device, uintptr(k.numBindings), uintptr(unsafe.Pointer(&writes[0])), 0, 0)

			r := vkSyscall(k.hw.getProc("vkResetFences"), k.hw.device, 1, uintptr(unsafe.Pointer(&k.fence)))
			if r != 0 {
				return 0, fmt.Errorf("vkResetFences failed: %d", r)
			}

			beginInfo := VkCommandBufferBeginInfo{
				SType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
				Flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
			}
			r = vkSyscall(k.hw.getProc("vkBeginCommandBuffer"), k.cmdBuf, uintptr(unsafe.Pointer(&beginInfo)))
			if r != 0 {
				return 0, fmt.Errorf("vkBeginCommandBuffer failed: %d", r)
			}

			vkSyscall(cmdResetQP, k.cmdBuf, uintptr(k.queryPool), 0, 2)
			vkSyscall(cmdWriteTS, k.cmdBuf, uintptr(1 /* TOP_OF_PIPE */), uintptr(k.queryPool), 0)

			vkSyscall(k.hw.getProc("vkCmdBindPipeline"), k.cmdBuf, uintptr(VK_PIPELINE_BIND_POINT_COMPUTE), uintptr(k.pipeline))
			vkSyscall(k.hw.getProc("vkCmdBindDescriptorSets"), k.cmdBuf, uintptr(VK_PIPELINE_BIND_POINT_COMPUTE), uintptr(k.pipeLayout), 0, 1, uintptr(unsafe.Pointer(&k.descSet)), 0, 0)
			if pushData != nil && pushSize > 0 {
				vkSyscall(k.hw.getProc("vkCmdPushConstants"), k.cmdBuf, uintptr(k.pipeLayout), uintptr(VK_SHADER_STAGE_COMPUTE_BIT), 0, uintptr(pushSize), uintptr(pushData))
			}
			vkSyscall(k.hw.getProc("vkCmdDispatch"), k.cmdBuf, uintptr(workgroups), 1, 1)

			vkSyscall(cmdWriteTS, k.cmdBuf, uintptr(0x2000 /* BOTTOM_OF_PIPE */), uintptr(k.queryPool), 1)

			r = vkSyscall(k.hw.getProc("vkEndCommandBuffer"), k.cmdBuf)
			if r != 0 {
				return 0, fmt.Errorf("vkEndCommandBuffer failed: %d", r)
			}

			submitInfo := VkSubmitInfo{
				SType:              VK_STRUCTURE_TYPE_SUBMIT_INFO,
				CommandBufferCount: 1,
				PCommandBuffers:    &k.cmdBuf,
			}
			r = vkSyscall(k.hw.getProc("vkQueueSubmit"), k.hw.queue, 1, uintptr(unsafe.Pointer(&submitInfo)), uintptr(k.fence))
			if r != 0 {
				return 0, fmt.Errorf("vkQueueSubmit failed: %d", r)
			}

			r = vkSyscall(k.hw.getProc("vkWaitForFences"), k.hw.device, 1, uintptr(unsafe.Pointer(&k.fence)), 1, uintptr(2000000000))
			if r != 0 {
				return 0, fmt.Errorf("GPU compute execution failed or timed out: %d", r)
			}

			var timestamps [2]uint64
			r = vkSyscall(getQPResults, k.hw.device, uintptr(k.queryPool), 0, 2, 16, uintptr(unsafe.Pointer(&timestamps[0])), 8, 1|2)
			if r == 0 && timestamps[1] >= timestamps[0] {
				ticks := timestamps[1] - timestamps[0]
				period := k.hw.TimestampPeriod()
				nanos := float64(ticks) * float64(period)
				return time.Duration(nanos), nil
			}
		}
	}

	start := time.Now()
	token, err := k.DispatchAsync(buffers, pushData, pushSize, workgroups)
	if err != nil {
		return 0, err
	}
	if err := token.Wait(2000000000); err != nil {
		return 0, err
	}
	return time.Since(start), nil
}

// Close releases all Vulkan resources associated with this kernel.
func (k *ComputeKernel) Close() {
	k.mu.Lock()
	defer k.mu.Unlock()

	if k.inFlight && k.fence != 0 {
		vkSyscall(k.hw.getProc("vkWaitForFences"), k.hw.device, 1, uintptr(unsafe.Pointer(&k.fence)), 1, uintptr(1000000000))
		k.inFlight = false
	}

	if k.queryPool != 0 {
		destroyQP := k.hw.getProc("vkDestroyQueryPool")
		if destroyQP != 0 {
			vkSyscall(destroyQP, k.hw.device, uintptr(k.queryPool), 0)
		}
		k.queryPool = 0
	}
	if k.fence != 0 {
		vkSyscall(k.hw.getProc("vkDestroyFence"), k.hw.device, uintptr(k.fence), 0)
		k.fence = 0
	}
	if k.cmdPool != 0 {
		vkSyscall(k.hw.getProc("vkDestroyCommandPool"), k.hw.device, uintptr(k.cmdPool), 0)
		k.cmdPool = 0
	}
	if k.descPool != 0 {
		vkSyscall(k.hw.getProc("vkDestroyDescriptorPool"), k.hw.device, uintptr(k.descPool), 0)
		k.descPool = 0
	}
	if k.pipeline != 0 {
		vkSyscall(k.hw.getProc("vkDestroyPipeline"), k.hw.device, uintptr(k.pipeline), 0)
		k.pipeline = 0
	}
	if k.pipeLayout != 0 {
		vkSyscall(k.hw.getProc("vkDestroyPipelineLayout"), k.hw.device, uintptr(k.pipeLayout), 0)
		k.pipeLayout = 0
	}
	if k.descLayout != 0 {
		vkSyscall(k.hw.getProc("vkDestroyDescriptorSetLayout"), k.hw.device, uintptr(k.descLayout), 0)
		k.descLayout = 0
	}
	if k.shaderModule != 0 {
		vkSyscall(k.hw.getProc("vkDestroyShaderModule"), k.hw.device, uintptr(k.shaderModule), 0)
		k.shaderModule = 0
	}
}
