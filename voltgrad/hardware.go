package voltgrad

import (
	_ "embed"
	"encoding/binary"
	"errors"
	"fmt"
	"syscall"
	"unsafe"
)

//go:embed train.spv
var defaultShaderSPV []byte

// HardwareContext encapsulates low-level Vulkan instance, device, and compute queues.
type HardwareContext struct {
	dll                   *syscall.LazyDLL
	vkGetInstanceProcAddr *syscall.LazyProc

	instance uintptr
	physDev  uintptr
	device   uintptr
	queue    uintptr
	queueFam uint32
	memProps VkPhysicalDeviceMemoryProperties
	hasShaderInt64 bool
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

	getProps := c.dll.NewProc("vkGetPhysicalDeviceProperties")
	getQProps := c.dll.NewProc("vkGetPhysicalDeviceQueueFamilyProperties")
	getFeats := c.dll.NewProc("vkGetPhysicalDeviceFeatures")

	bestScore := -1
	bestDev := uintptr(0)
	bestQueueFam := -1

	for _, dev := range physDevs {
		// 1. Must support mandatory shaderInt64
		var feats VkPhysicalDeviceFeatures
		getFeats.Call(dev, uintptr(unsafe.Pointer(&feats)))
		if feats.ShaderInt64 == 0 {
			continue
		}

		// 2. Must support compute queue
		var qCount uint32
		getQProps.Call(dev, uintptr(unsafe.Pointer(&qCount)), 0)
		if qCount == 0 {
			continue
		}
		qProps := make([]VkQueueFamilyProperties, qCount)
		getQProps.Call(dev, uintptr(unsafe.Pointer(&qCount)), uintptr(unsafe.Pointer(&qProps[0])))

		computeFam := -1
		for i, qf := range qProps {
			if qf.QueueFlags&VK_QUEUE_COMPUTE_BIT != 0 {
				computeFam = i
				break
			}
		}
		if computeFam == -1 {
			continue
		}

		// 3. Score device: Discrete GPU (1000) > Integrated GPU (500) > Virtual (200) > CPU (100)
		propsBuf := make([]byte, 1024)
		getProps.Call(dev, uintptr(unsafe.Pointer(&propsBuf[0])))
		deviceType := binary.LittleEndian.Uint32(propsBuf[16:20])

		score := 10
		switch deviceType {
		case 2: // DISCRETE_GPU
			score = 1000
		case 1: // INTEGRATED_GPU
			score = 500
		case 3: // VIRTUAL_GPU
			score = 200
		case 4: // CPU
			score = 100
		}

		if score > bestScore {
			bestScore = score
			bestDev = dev
			bestQueueFam = computeFam
		}
	}

	if bestDev == 0 || bestQueueFam == -1 {
		return errors.New("no compatible Vulkan compute device with shaderInt64 found")
	}

	c.physDev = bestDev
	c.queueFam = uint32(bestQueueFam)
	c.hasShaderInt64 = true

	priority := float32(1.0)
	queueCI := VkDeviceQueueCreateInfo{
		SType:            VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		QueueFamilyIndex: c.queueFam,
		QueueCount:       1,
		PQueuePriorities: &priority,
	}

	enabledFeatures := VkPhysicalDeviceFeatures{
		ShaderInt64: 1,
	}

	deviceCI := VkDeviceCreateInfo{
		SType:                VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		QueueCreateInfoCount: 1,
		PQueueCreateInfos:    &queueCI,
		PEnabledFeatures:     unsafe.Pointer(&enabledFeatures),
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

// HasShaderInt64 returns true if 64-bit integer compute shader operations are enabled and validated on the hardware.
func (c *HardwareContext) HasShaderInt64() bool {
	return c.hasShaderInt64
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
		waitIdle := c.getProc("vkDeviceWaitIdle")
		if waitIdle != 0 {
			syscall.SyscallN(waitIdle, c.device)
		}
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

// GpuBuffer represents a host-coherent memory buffer directly accessible by both CPU and GPU.
type GpuBuffer struct {
	ctx    *HardwareContext
	buffer uint64
	memory uint64
	size   uint64
	mapped unsafe.Pointer
}

// CreateHostCoherentBuffer allocates a host-visible, host-coherent buffer (Zero Copy).
func (c *HardwareContext) CreateHostCoherentBuffer(size uint64) (*GpuBuffer, error) {
	bci := VkBufferCreateInfo{
		SType:       VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		Size:        size,
		Usage:       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		SharingMode: VK_SHARING_MODE_EXCLUSIVE,
	}
	var buf uint64
	r, _, _ := syscall.SyscallN(c.getProc("vkCreateBuffer"), c.device, uintptr(unsafe.Pointer(&bci)), 0, uintptr(unsafe.Pointer(&buf)))
	if r != 0 {
		return nil, fmt.Errorf("vkCreateBuffer error: %d", r)
	}

	var reqs VkMemoryRequirements
	syscall.SyscallN(c.getProc("vkGetBufferMemoryRequirements"), c.device, uintptr(buf), uintptr(unsafe.Pointer(&reqs)))

	memType := c.findMemoryType(reqs.MemoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
	if memType == -1 {
		syscall.SyscallN(c.getProc("vkDestroyBuffer"), c.device, uintptr(buf), 0)
		return nil, errors.New("host visible + coherent memory type unavailable")
	}

	allocInfo := VkMemoryAllocateInfo{
		SType:           VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		AllocationSize:  reqs.Size,
		MemoryTypeIndex: uint32(memType),
	}
	var mem uint64
	r, _, _ = syscall.SyscallN(c.getProc("vkAllocateMemory"), c.device, uintptr(unsafe.Pointer(&allocInfo)), 0, uintptr(unsafe.Pointer(&mem)))
	if r != 0 {
		syscall.SyscallN(c.getProc("vkDestroyBuffer"), c.device, uintptr(buf), 0)
		return nil, fmt.Errorf("vkAllocateMemory error: %d", r)
	}

	r, _, _ = syscall.SyscallN(c.getProc("vkBindBufferMemory"), c.device, uintptr(buf), uintptr(mem), 0)
	if r != 0 {
		syscall.SyscallN(c.getProc("vkFreeMemory"), c.device, uintptr(mem), 0)
		syscall.SyscallN(c.getProc("vkDestroyBuffer"), c.device, uintptr(buf), 0)
		return nil, fmt.Errorf("vkBindBufferMemory error: %d", r)
	}

	var mapped unsafe.Pointer
	r, _, _ = syscall.SyscallN(c.getProc("vkMapMemory"), c.device, uintptr(mem), 0, uintptr(size), 0, uintptr(unsafe.Pointer(&mapped)))
	if r != 0 {
		syscall.SyscallN(c.getProc("vkFreeMemory"), c.device, uintptr(mem), 0)
		syscall.SyscallN(c.getProc("vkDestroyBuffer"), c.device, uintptr(buf), 0)
		return nil, fmt.Errorf("vkMapMemory error: %d", r)
	}

	return &GpuBuffer{
		ctx:    c,
		buffer: buf,
		memory: mem,
		size:   size,
		mapped: mapped,
	}, nil
}

// WriteInt32 copies a slice of int32 directly into the mapped memory.
func (b *GpuBuffer) WriteInt32(data []int32) {
	if len(data) == 0 {
		return
	}
	byteLen := len(data) * 4
	if uint64(byteLen) > b.size {
		panic("buffer overflow on WriteInt32")
	}
	src := unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), byteLen)
	dst := unsafe.Slice((*byte)(b.mapped), byteLen)
	copy(dst, src)
}

// WriteUint32 copies a slice of uint32 directly into the mapped memory.
func (b *GpuBuffer) WriteUint32(data []uint32) {
	if len(data) == 0 {
		return
	}
	byteLen := len(data) * 4
	if uint64(byteLen) > b.size {
		panic("buffer overflow on WriteUint32")
	}
	src := unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), byteLen)
	dst := unsafe.Slice((*byte)(b.mapped), byteLen)
	copy(dst, src)
}

// WriteInt64 copies a slice of int64 directly into the mapped memory.
func (b *GpuBuffer) WriteInt64(data []int64) {
	if len(data) == 0 {
		return
	}
	byteLen := len(data) * 8
	if uint64(byteLen) > b.size {
		panic("buffer overflow on WriteInt64")
	}
	src := unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), byteLen)
	dst := unsafe.Slice((*byte)(b.mapped), byteLen)
	copy(dst, src)
}

// ReadInt32 reads back a copy of int32 elements into Go-owned memory.
func (b *GpuBuffer) ReadInt32(count int) []int32 {
	if count == 0 {
		return nil
	}
	src := unsafe.Slice((*int32)(b.mapped), count)
	out := make([]int32, count)
	copy(out, src)
	return out
}

// ReadUint32 reads back a copy of uint32 elements into Go-owned memory.
func (b *GpuBuffer) ReadUint32(count int) []uint32 {
	if count == 0 {
		return nil
	}
	src := unsafe.Slice((*uint32)(b.mapped), count)
	out := make([]uint32, count)
	copy(out, src)
	return out
}

// ReadInt64 reads back a copy of int64 elements into Go-owned memory.
func (b *GpuBuffer) ReadInt64(count int) []int64 {
	if count == 0 {
		return nil
	}
	src := unsafe.Slice((*int64)(b.mapped), count)
	out := make([]int64, count)
	copy(out, src)
	return out
}

// Clear zeroes out the mapped memory buffer.
func (b *GpuBuffer) Clear() {
	if b.mapped != nil && b.size > 0 {
		dst := unsafe.Slice((*byte)(b.mapped), b.size)
		for i := range dst {
			dst[i] = 0
		}
	}
}

// Free releases the buffer and unmaps memory.
func (b *GpuBuffer) Free() {
	if b.mapped != nil {
		syscall.SyscallN(b.ctx.getProc("vkUnmapMemory"), b.ctx.device, uintptr(b.memory))
		b.mapped = nil
	}
	if b.buffer != 0 {
		syscall.SyscallN(b.ctx.getProc("vkDestroyBuffer"), b.ctx.device, uintptr(b.buffer), 0)
		b.buffer = 0
	}
	if b.memory != 0 {
		syscall.SyscallN(b.ctx.getProc("vkFreeMemory"), b.ctx.device, uintptr(b.memory), 0)
		b.memory = 0
	}
}
