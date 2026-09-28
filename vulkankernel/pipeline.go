package vulkankernel

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
