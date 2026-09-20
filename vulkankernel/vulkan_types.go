package vulkankernel

import "unsafe"

// Vulkan C-ABI mirrors (hand-ported subset of vulkan_core.h 1.3)
const (
	VK_STRUCTURE_TYPE_APPLICATION_INFO                  uint32 = 1
	VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO              uint32 = 10
	VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO          uint32 = 2
	VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO                uint32 = 3
	VK_STRUCTURE_TYPE_SUBMIT_INFO                       uint32 = 4
	VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO              uint32 = 5
	VK_STRUCTURE_TYPE_FENCE_CREATE_INFO                 uint32 = 8
	VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO                uint32 = 12
	VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO         uint32 = 16
	VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO uint32 = 18
	VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO      uint32 = 29
	VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO       uint32 = 30
	VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO uint32 = 32
	VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO       uint32 = 33
	VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO      uint32 = 34
	VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET              uint32 = 35
	VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO          uint32 = 39
	VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO      uint32 = 40
	VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO         uint32 = 42

	VK_QUEUE_COMPUTE_BIT uint32 = 0x00000002

	VK_BUFFER_USAGE_STORAGE_BUFFER_BIT uint32 = 0x00000020
	VK_SHARING_MODE_EXCLUSIVE          uint32 = 0

	VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT  uint32 = 0x00000002
	VK_MEMORY_PROPERTY_HOST_COHERENT_BIT uint32 = 0x00000004

	VK_DESCRIPTOR_TYPE_STORAGE_BUFFER uint32 = 7
	VK_SHADER_STAGE_COMPUTE_BIT       uint32 = 0x00000020

	VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT uint32 = 0x00000002
	VK_COMMAND_BUFFER_LEVEL_PRIMARY                 uint32 = 0
	VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT     uint32 = 0x00000001
	VK_PIPELINE_BIND_POINT_COMPUTE                  uint32 = 1

	VK_SUCCESS uint32 = 0
)

type VkApplicationInfo struct {
	SType              uint32
	_                  uint32
	PNext              unsafe.Pointer
	PApplicationName   *byte
	ApplicationVersion uint32
	_                  uint32
	PEngineName        *byte
	EngineVersion      uint32
	APIVersion         uint32
}

type VkInstanceCreateInfo struct {
	SType                   uint32
	_                       uint32
	PNext                   unsafe.Pointer
	Flags                   uint32
	_                       uint32
	PApplicationInfo        *VkApplicationInfo
	EnabledLayerCount       uint32
	_                       uint32
	PpEnabledLayerNames     *uintptr
	EnabledExtensionCount   uint32
	_                       uint32
	PpEnabledExtensionNames *uintptr
}

type VkQueueFamilyProperties struct {
	QueueFlags                  uint32
	QueueCount                  uint32
	TimestampValidBits          uint32
	MinImageTransferGranularity [3]uint32
}

type VkDeviceQueueCreateInfo struct {
	SType            uint32
	_                uint32
	PNext            unsafe.Pointer
	Flags            uint32
	QueueFamilyIndex uint32
	QueueCount       uint32
	_                uint32
	PQueuePriorities *float32
}

type VkDeviceCreateInfo struct {
	SType                   uint32
	_                       uint32
	PNext                   unsafe.Pointer
	Flags                   uint32
	QueueCreateInfoCount    uint32
	PQueueCreateInfos       *VkDeviceQueueCreateInfo
	EnabledLayerCount       uint32
	_                       uint32
	PpEnabledLayerNames     *uintptr
	EnabledExtensionCount   uint32
	_                       uint32
	PpEnabledExtensionNames *uintptr
	PEnabledFeatures        unsafe.Pointer
}

type VkMemoryType struct {
	PropertyFlags uint32
	HeapIndex     uint32
}

type VkMemoryHeap struct {
	Size  uint64
	Flags uint32
	_     uint32
}

type VkPhysicalDeviceMemoryProperties struct {
	MemoryTypeCount uint32
	MemoryTypes     [32]VkMemoryType
	MemoryHeapCount uint32
	MemoryHeaps     [16]VkMemoryHeap
}

type VkBufferCreateInfo struct {
	SType                 uint32
	_                     uint32
	PNext                 unsafe.Pointer
	Flags                 uint32
	_                     uint32
	Size                  uint64
	Usage                 uint32
	SharingMode           uint32
	QueueFamilyIndexCount uint32
	_                     uint32
	PQueueFamilyIndices   *uint32
}

type VkMemoryRequirements struct {
	Size           uint64
	Alignment      uint64
	MemoryTypeBits uint32
	_              uint32
}

type VkMemoryAllocateInfo struct {
	SType           uint32
	_               uint32
	PNext           unsafe.Pointer
	AllocationSize  uint64
	MemoryTypeIndex uint32
	_               uint32
}

type VkShaderModuleCreateInfo struct {
	SType    uint32
	_        uint32
	PNext    unsafe.Pointer
	Flags    uint32
	CodeSize uintptr
	PCode    *uint32
}

type VkDescriptorSetLayoutBinding struct {
	Binding            uint32
	DescriptorType     uint32
	DescriptorCount    uint32
	StageFlags         uint32
	PImmutableSamplers unsafe.Pointer
}

type VkDescriptorSetLayoutCreateInfo struct {
	SType        uint32
	_            uint32
	PNext        unsafe.Pointer
	Flags        uint32
	BindingCount uint32
	PBindings    *VkDescriptorSetLayoutBinding
}

type VkPushConstantRange struct {
	StageFlags uint32
	Offset     uint32
	Size       uint32
}

type VkPipelineLayoutCreateInfo struct {
	SType                  uint32
	_                      uint32
	PNext                  unsafe.Pointer
	Flags                  uint32
	SetLayoutCount         uint32
	PSetLayouts            *uint64
	PushConstantRangeCount uint32
	_                      uint32
	PPushConstantRanges    *VkPushConstantRange
}

type VkPipelineShaderStageCreateInfo struct {
	SType               uint32
	_                   uint32
	PNext               unsafe.Pointer
	Flags               uint32
	Stage               uint32
	Module              uint64
	PName               *byte
	PSpecializationInfo unsafe.Pointer
}

type VkComputePipelineCreateInfo struct {
	SType              uint32
	_                  uint32
	PNext              unsafe.Pointer
	Flags              uint32
	_                  uint32
	Stage              VkPipelineShaderStageCreateInfo
	Layout             uint64
	BasePipelineHandle uint64
	BasePipelineIndex  int32
	_                  uint32
}

type VkDescriptorPoolSize struct {
	Type            uint32
	DescriptorCount uint32
}

type VkDescriptorPoolCreateInfo struct {
	SType         uint32
	_             uint32
	PNext         unsafe.Pointer
	Flags         uint32
	MaxSets       uint32
	PoolSizeCount uint32
	_             uint32
	PPoolSizes    *VkDescriptorPoolSize
}

type VkDescriptorSetAllocateInfo struct {
	SType              uint32
	_                  uint32
	PNext              unsafe.Pointer
	DescriptorPool     uint64
	DescriptorSetCount uint32
	_                  uint32
	PSetLayouts        *uint64
}

type VkDescriptorBufferInfo struct {
	Buffer uint64
	Offset uint64
	Range  uint64
}

type VkWriteDescriptorSet struct {
	SType            uint32
	_                uint32
	PNext            unsafe.Pointer
	DstSet           uint64
	DstBinding       uint32
	DstArrayElement  uint32
	DescriptorCount  uint32
	DescriptorType   uint32
	PImageInfo       unsafe.Pointer
	PBufferInfo      *VkDescriptorBufferInfo
	PTexelBufferView unsafe.Pointer
}

type VkCommandPoolCreateInfo struct {
	SType            uint32
	_                uint32
	PNext            unsafe.Pointer
	Flags            uint32
	QueueFamilyIndex uint32
}

type VkCommandBufferAllocateInfo struct {
	SType              uint32
	_                  uint32
	PNext              unsafe.Pointer
	CommandPool        uint64
	Level              uint32
	CommandBufferCount uint32
}

type VkCommandBufferBeginInfo struct {
	SType            uint32
	_                uint32
	PNext            unsafe.Pointer
	Flags            uint32
	_                uint32
	PInheritanceInfo unsafe.Pointer
}

type VkSubmitInfo struct {
	SType                uint32
	_                    uint32
	PNext                unsafe.Pointer
	WaitSemaphoreCount   uint32
	_                    uint32
	PWaitSemaphores      unsafe.Pointer
	PWaitDstStageMask    unsafe.Pointer
	CommandBufferCount   uint32
	_                    uint32
	PCommandBuffers      *uintptr
	SignalSemaphoreCount uint32
	_                    uint32
	PSignalSemaphores    unsafe.Pointer
}

type VkFenceCreateInfo struct {
	SType uint32
	_     uint32
	PNext unsafe.Pointer
	Flags uint32
	_     uint32
}
