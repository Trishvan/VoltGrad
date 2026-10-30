package voltgrad

import (
	"fmt"
	"syscall"
	"unsafe"
)

// Pipeline manages Vulkan compute shaders, descriptor sets, and command execution.
type Pipeline struct {
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

	// Pre-allocated IO buffers
	inputBuffer  *GpuBuffer
	outputBuffer *GpuBuffer
	bufferCap    int
}

// NewPipeline creates and compiles a Vulkan compute pipeline using embedded SPIR-V shader bytecode.
func NewPipeline(hw *HardwareContext, customShaderSPV []byte) (*Pipeline, error) {
	spv := customShaderSPV
	if len(spv) == 0 {
		spv = defaultShaderSPV
	}
	if len(spv) == 0 {
		return nil, fmt.Errorf("no SPIR-V bytecode provided")
	}

	shaderCI := VkShaderModuleCreateInfo{
		SType:    VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		CodeSize: uintptr(len(spv)),
		PCode:    (*uint32)(unsafe.Pointer(&spv[0])),
	}
	var shaderModule uint64
	r, _, _ := syscall.SyscallN(hw.getProc("vkCreateShaderModule"), hw.device, uintptr(unsafe.Pointer(&shaderCI)), 0, uintptr(unsafe.Pointer(&shaderModule)))
	if r != 0 {
		return nil, fmt.Errorf("vkCreateShaderModule failed: %d", r)
	}

	// 2 Storage Buffer Bindings: Binding 0 (Inputs), Binding 1 (Outputs/Gradients)
	bindings := [2]VkDescriptorSetLayoutBinding{
		{
			Binding:         0,
			DescriptorType:  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			DescriptorCount: 1,
			StageFlags:      VK_SHADER_STAGE_COMPUTE_BIT,
		},
		{
			Binding:         1,
			DescriptorType:  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			DescriptorCount: 1,
			StageFlags:      VK_SHADER_STAGE_COMPUTE_BIT,
		},
	}
	dslCI := VkDescriptorSetLayoutCreateInfo{
		SType:        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		BindingCount: 2,
		PBindings:    &bindings[0],
	}
	var descLayout uint64
	r, _, _ = syscall.SyscallN(hw.getProc("vkCreateDescriptorSetLayout"), hw.device, uintptr(unsafe.Pointer(&dslCI)), 0, uintptr(unsafe.Pointer(&descLayout)))
	if r != 0 {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout failed: %d", r)
	}

	// Push constant: 4-byte current_weight or operational parameter
	pushConst := VkPushConstantRange{
		StageFlags: VK_SHADER_STAGE_COMPUTE_BIT,
		Offset:     0,
		Size:       4,
	}
	pipeLayoutCI := VkPipelineLayoutCreateInfo{
		SType:                  VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		SetLayoutCount:         1,
		PSetLayouts:            &descLayout,
		PushConstantRangeCount: 1,
		PPushConstantRanges:    &pushConst,
	}
	var pipeLayout uint64
	r, _, _ = syscall.SyscallN(hw.getProc("vkCreatePipelineLayout"), hw.device, uintptr(unsafe.Pointer(&pipeLayoutCI)), 0, uintptr(unsafe.Pointer(&pipeLayout)))
	if r != 0 {
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
	r, _, _ = syscall.SyscallN(hw.getProc("vkCreateComputePipelines"), hw.device, 0, 1, uintptr(unsafe.Pointer(&pipeCI)), 0, uintptr(unsafe.Pointer(&compPipeline)))
	if r != 0 {
		return nil, fmt.Errorf("vkCreateComputePipelines failed: %d", r)
	}

	poolSize := VkDescriptorPoolSize{
		Type:            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		DescriptorCount: 2,
	}
	poolCI := VkDescriptorPoolCreateInfo{
		SType:         VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		MaxSets:       1,
		PoolSizeCount: 1,
		PPoolSizes:    &poolSize,
	}
	var descPool uint64
	r, _, _ = syscall.SyscallN(hw.getProc("vkCreateDescriptorPool"), hw.device, uintptr(unsafe.Pointer(&poolCI)), 0, uintptr(unsafe.Pointer(&descPool)))
	if r != 0 {
		return nil, fmt.Errorf("vkCreateDescriptorPool failed: %d", r)
	}

	allocSet := VkDescriptorSetAllocateInfo{
		SType:              VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		DescriptorPool:     descPool,
		DescriptorSetCount: 1,
		PSetLayouts:        &descLayout,
	}
	var descSet uint64
	r, _, _ = syscall.SyscallN(hw.getProc("vkAllocateDescriptorSets"), hw.device, uintptr(unsafe.Pointer(&allocSet)), uintptr(unsafe.Pointer(&descSet)))
	if r != 0 {
		return nil, fmt.Errorf("vkAllocateDescriptorSets failed: %d", r)
	}

	cmdPoolCI := VkCommandPoolCreateInfo{
		SType:            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		Flags:            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		QueueFamilyIndex: hw.queueFam,
	}
	var cmdPool uint64
	r, _, _ = syscall.SyscallN(hw.getProc("vkCreateCommandPool"), hw.device, uintptr(unsafe.Pointer(&cmdPoolCI)), 0, uintptr(unsafe.Pointer(&cmdPool)))
	if r != 0 {
		return nil, fmt.Errorf("vkCreateCommandPool failed: %d", r)
	}

	cmdAlloc := VkCommandBufferAllocateInfo{
		SType:              VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		CommandPool:        cmdPool,
		Level:              VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		CommandBufferCount: 1,
	}
	var cmdBuf uintptr
	r, _, _ = syscall.SyscallN(hw.getProc("vkAllocateCommandBuffers"), hw.device, uintptr(unsafe.Pointer(&cmdAlloc)), uintptr(unsafe.Pointer(&cmdBuf)))
	if r != 0 {
		return nil, fmt.Errorf("vkAllocateCommandBuffers failed: %d", r)
	}

	fenceCI := VkFenceCreateInfo{SType: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}
	var fence uint64
	r, _, _ = syscall.SyscallN(hw.getProc("vkCreateFence"), hw.device, uintptr(unsafe.Pointer(&fenceCI)), 0, uintptr(unsafe.Pointer(&fence)))
	if r != 0 {
		return nil, fmt.Errorf("vkCreateFence failed: %d", r)
	}

	return &Pipeline{
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
	}, nil
}

// EnsureBuffers ensures host-coherent input and output GPU storage buffers have capacity for count elements.
func (p *Pipeline) EnsureBuffers(count int) error {
	if p.bufferCap >= count && p.inputBuffer != nil && p.outputBuffer != nil {
		return nil
	}

	if p.inputBuffer != nil {
		p.inputBuffer.Free()
		p.inputBuffer = nil
	}
	if p.outputBuffer != nil {
		p.outputBuffer.Free()
		p.outputBuffer = nil
	}

	// Double word size: 8 bytes per DataPoint (x: 4 bytes, y_true: 4 bytes)
	inBytes := uint64(count * 8)
	outBytes := uint64(count * 4)

	var err error
	p.inputBuffer, err = p.hw.CreateHostCoherentBuffer(inBytes)
	if err != nil {
		return fmt.Errorf("failed allocating input buffer: %w", err)
	}

	p.outputBuffer, err = p.hw.CreateHostCoherentBuffer(outBytes)
	if err != nil {
		return fmt.Errorf("failed allocating output buffer: %w", err)
	}
	p.bufferCap = count

	// Bind to descriptor sets
	bufferInfos := [2]VkDescriptorBufferInfo{
		{Buffer: p.inputBuffer.buffer, Offset: 0, Range: inBytes},
		{Buffer: p.outputBuffer.buffer, Offset: 0, Range: outBytes},
	}
	writes := [2]VkWriteDescriptorSet{
		{
			SType:           VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			DstSet:          p.descSet,
			DstBinding:      0,
			DescriptorCount: 1,
			DescriptorType:  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			PBufferInfo:     &bufferInfos[0],
		},
		{
			SType:           VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			DstSet:          p.descSet,
			DstBinding:      1,
			DescriptorCount: 1,
			DescriptorType:  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			PBufferInfo:     &bufferInfos[1],
		},
	}
	syscall.SyscallN(p.hw.getProc("vkUpdateDescriptorSets"), p.hw.device, 2, uintptr(unsafe.Pointer(&writes[0])), 0, 0)
	return nil
}

// Dispatch executes the compute shader on the GPU across count elements with pushParam.
func (p *Pipeline) Dispatch(inputPairs []int32, pushParam int32) ([]int32, error) {
	count := len(inputPairs) / 2
	if count == 0 {
		return nil, fmt.Errorf("inputPairs must contain at least 1 pair (x, y)")
	}
	// Round up to multiple of 64
	paddedCount := ((count + 63) / 64) * 64

	if err := p.EnsureBuffers(paddedCount); err != nil {
		return nil, err
	}

	// Upload inputs
	p.inputBuffer.WriteInt32(inputPairs)

	// Reset fence
	syscall.SyscallN(p.hw.getProc("vkResetFences"), p.hw.device, 1, uintptr(unsafe.Pointer(&p.fence)))

	// Record command buffer
	beginInfo := VkCommandBufferBeginInfo{
		SType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		Flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	}
	syscall.SyscallN(p.hw.getProc("vkBeginCommandBuffer"), p.cmdBuf, uintptr(unsafe.Pointer(&beginInfo)))
	syscall.SyscallN(p.hw.getProc("vkCmdBindPipeline"), p.cmdBuf, uintptr(VK_PIPELINE_BIND_POINT_COMPUTE), uintptr(p.pipeline))
	syscall.SyscallN(p.hw.getProc("vkCmdBindDescriptorSets"), p.cmdBuf, uintptr(VK_PIPELINE_BIND_POINT_COMPUTE), uintptr(p.pipeLayout), 0, 1, uintptr(unsafe.Pointer(&p.descSet)), 0, 0)

	syscall.SyscallN(p.hw.getProc("vkCmdPushConstants"), p.cmdBuf, uintptr(p.pipeLayout), uintptr(VK_SHADER_STAGE_COMPUTE_BIT), 0, 4, uintptr(unsafe.Pointer(&pushParam)))

	workgroups := uint32(paddedCount / 64)
	syscall.SyscallN(p.hw.getProc("vkCmdDispatch"), p.cmdBuf, uintptr(workgroups), 1, 1)
	syscall.SyscallN(p.hw.getProc("vkEndCommandBuffer"), p.cmdBuf)

	submitInfo := VkSubmitInfo{
		SType:              VK_STRUCTURE_TYPE_SUBMIT_INFO,
		CommandBufferCount: 1,
		PCommandBuffers:    &p.cmdBuf,
	}
	syscall.SyscallN(p.hw.getProc("vkQueueSubmit"), p.hw.queue, 1, uintptr(unsafe.Pointer(&submitInfo)), uintptr(p.fence))

	// Wait for GPU fence (2-second timeout)
	r, _, _ := syscall.SyscallN(p.hw.getProc("vkWaitForFences"), p.hw.device, 1, uintptr(unsafe.Pointer(&p.fence)), 1, uintptr(2000000000))
	if r != 0 {
		return nil, fmt.Errorf("Vulkan GPU compute execution timed out or failed: %d", r)
	}

	// Read outputs directly from host-coherent mapped memory
	return p.outputBuffer.ReadInt32(count), nil
}

// Close releases all pipeline resources and descriptors.
func (p *Pipeline) Close() {
	if p.inputBuffer != nil {
		p.inputBuffer.Free()
		p.inputBuffer = nil
	}
	if p.outputBuffer != nil {
		p.outputBuffer.Free()
		p.outputBuffer = nil
	}
	if p.fence != 0 {
		syscall.SyscallN(p.hw.getProc("vkDestroyFence"), p.hw.device, uintptr(p.fence), 0)
		p.fence = 0
	}
	if p.cmdPool != 0 {
		syscall.SyscallN(p.hw.getProc("vkDestroyCommandPool"), p.hw.device, uintptr(p.cmdPool), 0)
		p.cmdPool = 0
	}
	if p.descPool != 0 {
		syscall.SyscallN(p.hw.getProc("vkDestroyDescriptorPool"), p.hw.device, uintptr(p.descPool), 0)
		p.descPool = 0
	}
	if p.pipeline != 0 {
		syscall.SyscallN(p.hw.getProc("vkDestroyPipeline"), p.hw.device, uintptr(p.pipeline), 0)
		p.pipeline = 0
	}
	if p.pipeLayout != 0 {
		syscall.SyscallN(p.hw.getProc("vkDestroyPipelineLayout"), p.hw.device, uintptr(p.pipeLayout), 0)
		p.pipeLayout = 0
	}
	if p.descLayout != 0 {
		syscall.SyscallN(p.hw.getProc("vkDestroyDescriptorSetLayout"), p.hw.device, uintptr(p.descLayout), 0)
		p.descLayout = 0
	}
	if p.shaderModule != 0 {
		syscall.SyscallN(p.hw.getProc("vkDestroyShaderModule"), p.hw.device, uintptr(p.shaderModule), 0)
		p.shaderModule = 0
	}
}
