package voltgrad

import (
	_ "embed"
	"fmt"
	"syscall"
	"unsafe"
)

//go:embed shaders/matmul_forward.spv
var matmulForwardSPV []byte

//go:embed shaders/matmul_backward_dx.spv
var matmulBackwardDxSPV []byte

//go:embed shaders/matmul_backward_dw.spv
var matmulBackwardDwSPV []byte

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
	numBindings  int
	pushSize     uint32
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
	r, _, _ := syscall.SyscallN(hw.getProc("vkCreateShaderModule"), hw.device, uintptr(unsafe.Pointer(&shaderCI)), 0, uintptr(unsafe.Pointer(&shaderModule)))
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
	r, _, _ = syscall.SyscallN(hw.getProc("vkCreateDescriptorSetLayout"), hw.device, uintptr(unsafe.Pointer(&dslCI)), 0, uintptr(unsafe.Pointer(&descLayout)))
	if r != 0 {
		syscall.SyscallN(hw.getProc("vkDestroyShaderModule"), hw.device, uintptr(shaderModule), 0)
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
		r, _, _ = syscall.SyscallN(hw.getProc("vkCreatePipelineLayout"), hw.device, uintptr(unsafe.Pointer(&pipeLayoutCI)), 0, uintptr(unsafe.Pointer(&pipeLayout)))
	} else {
		pipeLayoutCI := VkPipelineLayoutCreateInfo{
			SType:          VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			SetLayoutCount: 1,
			PSetLayouts:    &descLayout,
		}
		r, _, _ = syscall.SyscallN(hw.getProc("vkCreatePipelineLayout"), hw.device, uintptr(unsafe.Pointer(&pipeLayoutCI)), 0, uintptr(unsafe.Pointer(&pipeLayout)))
	}
	if r != 0 {
		syscall.SyscallN(hw.getProc("vkDestroyDescriptorSetLayout"), hw.device, uintptr(descLayout), 0)
		syscall.SyscallN(hw.getProc("vkDestroyShaderModule"), hw.device, uintptr(shaderModule), 0)
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
		syscall.SyscallN(hw.getProc("vkDestroyPipelineLayout"), hw.device, uintptr(pipeLayout), 0)
		syscall.SyscallN(hw.getProc("vkDestroyDescriptorSetLayout"), hw.device, uintptr(descLayout), 0)
		syscall.SyscallN(hw.getProc("vkDestroyShaderModule"), hw.device, uintptr(shaderModule), 0)
		return nil, fmt.Errorf("vkCreateComputePipelines failed: %d", r)
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
		numBindings:  numBindings,
		pushSize:     pushSize,
	}, nil
}

// Dispatch executes the compute shader across the given storage buffers with push constants.
func (k *ComputeKernel) Dispatch(buffers []*GpuBuffer, pushData unsafe.Pointer, pushSize uint32, workgroups uint32) error {
	if len(buffers) != k.numBindings {
		return fmt.Errorf("kernel requires %d buffers, got %d", k.numBindings, len(buffers))
	}
	if workgroups == 0 {
		workgroups = 1
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
	syscall.SyscallN(k.hw.getProc("vkUpdateDescriptorSets"), k.hw.device, uintptr(k.numBindings), uintptr(unsafe.Pointer(&writes[0])), 0, 0)

	// Reset fence
	syscall.SyscallN(k.hw.getProc("vkResetFences"), k.hw.device, 1, uintptr(unsafe.Pointer(&k.fence)))

	// Record command buffer
	beginInfo := VkCommandBufferBeginInfo{
		SType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		Flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	}
	syscall.SyscallN(k.hw.getProc("vkBeginCommandBuffer"), k.cmdBuf, uintptr(unsafe.Pointer(&beginInfo)))
	syscall.SyscallN(k.hw.getProc("vkCmdBindPipeline"), k.cmdBuf, uintptr(VK_PIPELINE_BIND_POINT_COMPUTE), uintptr(k.pipeline))
	syscall.SyscallN(k.hw.getProc("vkCmdBindDescriptorSets"), k.cmdBuf, uintptr(VK_PIPELINE_BIND_POINT_COMPUTE), uintptr(k.pipeLayout), 0, 1, uintptr(unsafe.Pointer(&k.descSet)), 0, 0)

	if pushData != nil && pushSize > 0 {
		syscall.SyscallN(k.hw.getProc("vkCmdPushConstants"), k.cmdBuf, uintptr(k.pipeLayout), uintptr(VK_SHADER_STAGE_COMPUTE_BIT), 0, uintptr(pushSize), uintptr(pushData))
	}

	syscall.SyscallN(k.hw.getProc("vkCmdDispatch"), k.cmdBuf, uintptr(workgroups), 1, 1)
	syscall.SyscallN(k.hw.getProc("vkEndCommandBuffer"), k.cmdBuf)

	submitInfo := VkSubmitInfo{
		SType:              VK_STRUCTURE_TYPE_SUBMIT_INFO,
		CommandBufferCount: 1,
		PCommandBuffers:    &k.cmdBuf,
	}
	syscall.SyscallN(k.hw.getProc("vkQueueSubmit"), k.hw.queue, 1, uintptr(unsafe.Pointer(&submitInfo)), uintptr(k.fence))

	// Wait for GPU fence (2-second timeout)
	r, _, _ := syscall.SyscallN(k.hw.getProc("vkWaitForFences"), k.hw.device, 1, uintptr(unsafe.Pointer(&k.fence)), 1, uintptr(2000000000))
	if r != 0 {
		return fmt.Errorf("GPU compute execution failed or timed out: %d", r)
	}
	return nil
}

// Close releases all Vulkan resources associated with this kernel.
func (k *ComputeKernel) Close() {
	if k.fence != 0 {
		syscall.SyscallN(k.hw.getProc("vkDestroyFence"), k.hw.device, uintptr(k.fence), 0)
		k.fence = 0
	}
	if k.cmdPool != 0 {
		syscall.SyscallN(k.hw.getProc("vkDestroyCommandPool"), k.hw.device, uintptr(k.cmdPool), 0)
		k.cmdPool = 0
	}
	if k.descPool != 0 {
		syscall.SyscallN(k.hw.getProc("vkDestroyDescriptorPool"), k.hw.device, uintptr(k.descPool), 0)
		k.descPool = 0
	}
	if k.pipeline != 0 {
		syscall.SyscallN(k.hw.getProc("vkDestroyPipeline"), k.hw.device, uintptr(k.pipeline), 0)
		k.pipeline = 0
	}
	if k.pipeLayout != 0 {
		syscall.SyscallN(k.hw.getProc("vkDestroyPipelineLayout"), k.hw.device, uintptr(k.pipeLayout), 0)
		k.pipeLayout = 0
	}
	if k.descLayout != 0 {
		syscall.SyscallN(k.hw.getProc("vkDestroyDescriptorSetLayout"), k.hw.device, uintptr(k.descLayout), 0)
		k.descLayout = 0
	}
	if k.shaderModule != 0 {
		syscall.SyscallN(k.hw.getProc("vkDestroyShaderModule"), k.hw.device, uintptr(k.shaderModule), 0)
		k.shaderModule = 0
	}
}
