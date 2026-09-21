package vulkankernel

import (
	"errors"
	"fmt"
	"syscall"
	"unsafe"
)

// HardwareContext owns the Vulkan instance, logical device and compute queue.
type HardwareContext struct {
	dll                   *syscall.LazyDLL
	vkGetInstanceProcAddr *syscall.LazyProc

	instance uintptr
	physDev  uintptr
	device   uintptr
	queue    uintptr
	queueFam uint32
	memProps VkPhysicalDeviceMemoryProperties
}

// NewHardwareContext discovers and initializes the system's Vulkan compute device.
func NewHardwareContext() (*HardwareContext, error) {
	dll := syscall.NewLazyDLL("vulkan-1.dll")
	if err := dll.Load(); err != nil {
		return nil, fmt.Errorf("vulkan-1.dll dynamic load failure: %w", err)
	}

	procAddr := dll.NewProc("vkGetInstanceProcAddr")
	ctx := &HardwareContext{
		dll:                   dll,
		vkGetInstanceProcAddr: procAddr,
	}

	if err := ctx.initDevice(); err != nil {
		ctx.Close()
		return nil, err
	}
	return ctx, nil
}

func (c *HardwareContext) getProc(name string) uintptr {
	cName := append([]byte(name), 0)
	r, _, _ := c.vkGetInstanceProcAddr.Call(c.instance, uintptr(unsafe.Pointer(&cName[0])))
	return r
}

func (c *HardwareContext) initDevice() error {
	appInfo := VkApplicationInfo{
		SType:      VK_STRUCTURE_TYPE_APPLICATION_INFO,
		APIVersion: (1 << 22) | (0 << 12),
	}
	ci := VkInstanceCreateInfo{
		SType:            VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		PApplicationInfo: &appInfo,
	}

	createInstance := c.dll.NewProc("vkCreateInstance")
	r, _, _ := createInstance.Call(uintptr(unsafe.Pointer(&ci)), 0, uintptr(unsafe.Pointer(&c.instance)))
	if r != 0 {
		return fmt.Errorf("vkCreateInstance failed: %d", r)
	}

	enumPhys := c.dll.NewProc("vkEnumeratePhysicalDevices")
	var physCount uint32
	enumPhys.Call(c.instance, uintptr(unsafe.Pointer(&physCount)), 0)
	if physCount == 0 {
		return errors.New("no Vulkan physical devices detected")
	}

	physDevs := make([]uintptr, physCount)
	enumPhys.Call(c.instance, uintptr(unsafe.Pointer(&physCount)), uintptr(unsafe.Pointer(&physDevs[0])))
	c.physDev = physDevs[0]

	getQProps := c.dll.NewProc("vkGetPhysicalDeviceQueueFamilyProperties")
	var qCount uint32
	getQProps.Call(c.physDev, uintptr(unsafe.Pointer(&qCount)), 0)
	qProps := make([]VkQueueFamilyProperties, qCount)
	getQProps.Call(c.physDev, uintptr(unsafe.Pointer(&qCount)), uintptr(unsafe.Pointer(&qProps[0])))

	computeQueueFamily := -1
	for i, qf := range qProps {
		if qf.QueueFlags&VK_QUEUE_COMPUTE_BIT != 0 {
			computeQueueFamily = i
			break
		}
	}
	if computeQueueFamily == -1 {
		return errors.New("no compute queue found on GPU device")
	}
	c.queueFam = uint32(computeQueueFamily)

	priority := float32(1.0)
	queueCI := VkDeviceQueueCreateInfo{
		SType:            VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		QueueFamilyIndex: c.queueFam,
		QueueCount:       1,
		PQueuePriorities: &priority,
	}
	deviceCI := VkDeviceCreateInfo{
		SType:                VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		QueueCreateInfoCount: 1,
		PQueueCreateInfos:    &queueCI,
	}

	createDev := c.dll.NewProc("vkCreateDevice")
	r, _, _ = createDev.Call(c.physDev, uintptr(unsafe.Pointer(&deviceCI)), 0, uintptr(unsafe.Pointer(&c.device)))
	if r != 0 {
		return fmt.Errorf("vkCreateDevice failed: %d", r)
	}

	syscall.SyscallN(c.getProc("vkGetDeviceQueue"), c.device, uintptr(c.queueFam), 0, uintptr(unsafe.Pointer(&c.queue)))
	syscall.SyscallN(c.getProc("vkGetPhysicalDeviceMemoryProperties"), c.physDev, uintptr(unsafe.Pointer(&c.memProps)))

	return nil
}

func (c *HardwareContext) findMemoryType(typeBits uint32, properties uint32) int {
	for i := uint32(0); i < c.memProps.MemoryTypeCount; i++ {
		if (typeBits&(1<<i)) != 0 && (c.memProps.MemoryTypes[i].PropertyFlags&properties) == properties {
			return int(i)
		}
	}
	return -1
}

// Close releases the Vulkan instance and logical device.
func (c *HardwareContext) Close() {
	if c.device != 0 {
		destroyDevice := c.getProc("vkDestroyDevice")
		if destroyDevice != 0 {
			syscall.SyscallN(destroyDevice, c.device, 0)
		}
		c.device = 0
	}
	if c.instance != 0 {
		destroyInst := c.dll.NewProc("vkDestroyInstance")
		if destroyInst != nil {
			destroyInst.Call(c.instance, 0)
		}
		c.instance = 0
	}
}

