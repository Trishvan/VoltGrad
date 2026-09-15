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

