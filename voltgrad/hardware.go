package voltgrad

import (
	_ "embed"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
	"unsafe"
)

//go:embed train.spv
var defaultShaderSPV []byte

// DeviceReport captures structured telemetry, architecture info, and memory heaps for the selected Vulkan device.
type DeviceReport struct {
	DeviceName       string
	DeviceType       string
	VendorID         uint32
	VendorName       string
	DeviceID         uint32
	APIVersion       string
	DriverVersion    string
	HasShaderInt64   bool
	TimestampPeriod  float32
	MemoryHeaps      []MemoryHeapReport
	TotalDeviceLocal uint64
	TotalHostVisible uint64
}

// MemoryHeapReport summarizes a physical memory heap on the GPU.
type MemoryHeapReport struct {
	Index       int
	SizeBytes   uint64
	DeviceLocal bool
}

// String returns a formatted multiline summary of the device report.
func (r DeviceReport) String() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Vulkan Device: %s (%s)\n", r.DeviceName, r.DeviceType))
	sb.WriteString(fmt.Sprintf("Vendor: %s [0x%04X] | DeviceID: 0x%04X\n", r.VendorName, r.VendorID, r.DeviceID))
	sb.WriteString(fmt.Sprintf("Vulkan API: %s | Driver Version: %s\n", r.APIVersion, r.DriverVersion))
	sb.WriteString(fmt.Sprintf("Hardware 64-bit Shader Integers (shaderInt64): %v\n", r.HasShaderInt64))
	sb.WriteString(fmt.Sprintf("Timestamp Period: %.2f ns/tick\n", r.TimestampPeriod))
	sb.WriteString(fmt.Sprintf("Memory Heaps (%d total):\n", len(r.MemoryHeaps)))
	for _, h := range r.MemoryHeaps {
		kind := "Host Visible / Shared"
		if h.DeviceLocal {
			kind = "Device Local (VRAM)"
		}
		sb.WriteString(fmt.Sprintf("  - Heap %d: %d MB [%s]\n", h.Index, h.SizeBytes/(1024*1024), kind))
	}
	sb.WriteString(fmt.Sprintf("Total VRAM: %d MB | Total Host-Visible Memory: %d MB",
		r.TotalDeviceLocal/(1024*1024), r.TotalHostVisible/(1024*1024)))
	return sb.String()
}

func deviceTypeName(t uint32) string {
	switch t {
	case 1:
		return "Integrated GPU"
	case 2:
		return "Discrete GPU"
	case 3:
		return "Virtual GPU"
	case 4:
		return "CPU"
	default:
		return "Other Device"
	}
}

func vendorName(id uint32) string {
	switch id {
	case 0x10DE:
		return "NVIDIA"
	case 0x1002:
		return "AMD"
	case 0x8086:
		return "Intel"
	case 0x13B5:
		return "ARM"
	case 0x5143:
		return "Qualcomm"
	case 0x106B:
		return "Apple"
	default:
		return fmt.Sprintf("Vendor [0x%04X]", id)
	}
}

// HardwareContext encapsulates low-level Vulkan instance, device, and compute queues.
type HardwareContext struct {
	loader *platformLoader

	instance        uintptr
	physDev         uintptr
	device          uintptr
	queue           uintptr
	queueFam        uint32
	memProps        VkPhysicalDeviceMemoryProperties
	deviceReport    DeviceReport
	hasShaderInt64  bool
	timestampPeriod float32
	pool            *BufferPool
}

// NewHardwareContext discovers and initializes the system's Vulkan compute device using a cross-platform loader.
func NewHardwareContext() (*HardwareContext, error) {
	loader, err := newPlatformLoader()
	if err != nil {
		return nil, err
	}

	ctx := &HardwareContext{
		loader: loader,
	}

	if err := ctx.initDevice(); err != nil {
		ctx.Close()
		return nil, err
	}
	ctx.pool = NewBufferPool(ctx)
	return ctx, nil
}

func (c *HardwareContext) getProc(name string) uintptr {
	if c.loader == nil {
		return 0
	}
	return c.loader.getProc(c.instance, name)
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

	createInstance := c.getProc("vkCreateInstance")
	if createInstance == 0 {
		return errors.New("vkCreateInstance procedure not found")
	}
	r := vkSyscall(createInstance, uintptr(unsafe.Pointer(&ci)), 0, uintptr(unsafe.Pointer(&c.instance)))
	if r != 0 {
		return fmt.Errorf("vkCreateInstance failed: %d", r)
	}

	enumPhys := c.getProc("vkEnumeratePhysicalDevices")
	if enumPhys == 0 {
		return errors.New("vkEnumeratePhysicalDevices procedure not found")
	}
	var physCount uint32
	vkSyscall(enumPhys, c.instance, uintptr(unsafe.Pointer(&physCount)), 0)
	if physCount == 0 {
		return errors.New("no Vulkan physical devices detected")
	}

	physDevs := make([]uintptr, physCount)
	vkSyscall(enumPhys, c.instance, uintptr(unsafe.Pointer(&physCount)), uintptr(unsafe.Pointer(&physDevs[0])))

	getProps := c.getProc("vkGetPhysicalDeviceProperties")
	getQProps := c.getProc("vkGetPhysicalDeviceQueueFamilyProperties")
	getFeats := c.getProc("vkGetPhysicalDeviceFeatures")

	bestScore := -1
	bestDev := uintptr(0)
	bestQueueFam := -1
	var bestPropsBuf []byte

	for _, dev := range physDevs {
		// 1. Must support mandatory shaderInt64
		var feats VkPhysicalDeviceFeatures
		vkSyscall(getFeats, dev, uintptr(unsafe.Pointer(&feats)))
		if feats.ShaderInt64 == 0 {
			continue
		}

		// 2. Must support compute queue
		var qCount uint32
		vkSyscall(getQProps, dev, uintptr(unsafe.Pointer(&qCount)), 0)
		if qCount == 0 {
			continue
		}
		qProps := make([]VkQueueFamilyProperties, qCount)
		vkSyscall(getQProps, dev, uintptr(unsafe.Pointer(&qCount)), uintptr(unsafe.Pointer(&qProps[0])))

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
		vkSyscall(getProps, dev, uintptr(unsafe.Pointer(&propsBuf[0])))
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
			bestPropsBuf = propsBuf
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

	createDev := c.getProc("vkCreateDevice")
	if createDev == 0 {
		return errors.New("vkCreateDevice procedure not found")
	}
	r = vkSyscall(createDev, c.physDev, uintptr(unsafe.Pointer(&deviceCI)), 0, uintptr(unsafe.Pointer(&c.device)))
	if r != 0 {
		return fmt.Errorf("vkCreateDevice failed: %d", r)
	}

	vkSyscall(c.getProc("vkGetDeviceQueue"), c.device, uintptr(c.queueFam), 0, uintptr(unsafe.Pointer(&c.queue)))
	vkSyscall(c.getProc("vkGetPhysicalDeviceMemoryProperties"), c.physDev, uintptr(unsafe.Pointer(&c.memProps)))

	// Populate DeviceReport
	if len(bestPropsBuf) >= 276 {
		apiVer := binary.LittleEndian.Uint32(bestPropsBuf[0:4])
		drvVer := binary.LittleEndian.Uint32(bestPropsBuf[4:8])
		vendID := binary.LittleEndian.Uint32(bestPropsBuf[8:12])
		devID := binary.LittleEndian.Uint32(bestPropsBuf[12:16])
		devType := binary.LittleEndian.Uint32(bestPropsBuf[16:20])

		nameBytes := bestPropsBuf[20:276]
		n := 0
		for n < len(nameBytes) && nameBytes[n] != 0 {
			n++
		}
		devName := string(nameBytes[:n])

		period := float32(1.0)
		if len(bestPropsBuf) >= 724 {
			u := binary.LittleEndian.Uint32(bestPropsBuf[720:724])
			rawF := math.Float32frombits(u)
			if rawF > 0 && !math.IsNaN(float64(rawF)) && !math.IsInf(float64(rawF), 0) {
				period = rawF
			}
		}
		c.timestampPeriod = period

		c.deviceReport = DeviceReport{
			DeviceName:      devName,
			DeviceType:      deviceTypeName(devType),
			VendorID:        vendID,
			VendorName:      vendorName(vendID),
			DeviceID:        devID,
			APIVersion:      fmt.Sprintf("%d.%d.%d", apiVer>>22, (apiVer>>12)&0x3ff, apiVer&0xfff),
			DriverVersion:   fmt.Sprintf("%d.%d.%d", drvVer>>22, (drvVer>>12)&0x3ff, drvVer&0xfff),
			HasShaderInt64:  true,
			TimestampPeriod: period,
		}

		for i := uint32(0); i < c.memProps.MemoryHeapCount; i++ {
			heap := c.memProps.MemoryHeaps[i]
			isDeviceLocal := (heap.Flags & 1) != 0
			c.deviceReport.MemoryHeaps = append(c.deviceReport.MemoryHeaps, MemoryHeapReport{
				Index:       int(i),
				SizeBytes:   heap.Size,
				DeviceLocal: isDeviceLocal,
			})
			if isDeviceLocal {
				c.deviceReport.TotalDeviceLocal += heap.Size
			} else {
				c.deviceReport.TotalHostVisible += heap.Size
			}
		}
	}

	return nil
}

// DeviceReport returns structured information and telemetry about the active Vulkan GPU.
func (c *HardwareContext) DeviceReport() DeviceReport {
	return c.deviceReport
}

// TimestampPeriod returns the hardware query timestamp period in nanoseconds per tick.
func (c *HardwareContext) TimestampPeriod() float32 {
	if c.timestampPeriod <= 0 {
		return 1.0
	}
	return c.timestampPeriod
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
	if c.pool != nil {
		c.pool.Clear()
		c.pool = nil
	}
	if c.device != 0 {
		waitIdle := c.getProc("vkDeviceWaitIdle")
		if waitIdle != 0 {
			vkSyscall(waitIdle, c.device)
		}
		destroyDevice := c.getProc("vkDestroyDevice")
		if destroyDevice != 0 {
			vkSyscall(destroyDevice, c.device, 0)
		}
		c.device = 0
	}
	if c.instance != 0 {
		destroyInst := c.getProc("vkDestroyInstance")
		if destroyInst != 0 {
			vkSyscall(destroyInst, c.instance, 0)
		}
		c.instance = 0
	}
	if c.loader != nil {
		c.loader.close()
		c.loader = nil
	}
}

// BufferPool returns the memory buffer pool associated with this hardware context.
func (c *HardwareContext) BufferPool() *BufferPool {
	return c.pool
}

// GetPooledBuffer retrieves a reusable buffer from the pool or allocates a new one.
func (c *HardwareContext) GetPooledBuffer(size uint64) (*GpuBuffer, error) {
	if c.pool != nil {
		return c.pool.Get(size)
	}
	return c.CreateHostCoherentBuffer(size)
}

// GpuBuffer represents a host-coherent memory buffer directly accessible by both CPU and GPU.
type GpuBuffer struct {
	ctx      *HardwareContext
	buffer   uint64
	memory   uint64
	size     uint64 // currently active logical size requested by the caller
	capacity uint64 // total allocated physical capacity in bytes
	mapped   unsafe.Pointer
	pool     *BufferPool
	inUse    bool
	isFreed  bool
	mu       sync.Mutex
}

// InUse reports whether the buffer is currently checked out of the pool or active.
func (b *GpuBuffer) InUse() bool {
	if b == nil {
		return false
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.inUse
}

// IsFreed reports whether the buffer memory has been destroyed.
func (b *GpuBuffer) IsFreed() bool {
	if b == nil {
		return true
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.isFreed
}

// SetActiveSize updates the active logical size of the buffer within its capacity.
func (b *GpuBuffer) SetActiveSize(size uint64) error {
	if b == nil {
		return errors.New("cannot set active size on nil buffer")
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.isFreed || b.mapped == nil {
		return errors.New("gpu buffer memory is not mapped or has been freed")
	}
	if size > b.capacity {
		return fmt.Errorf("active size %d exceeds buffer capacity %d", size, b.capacity)
	}
	b.size = size
	return nil
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
	r := vkSyscall(c.getProc("vkCreateBuffer"), c.device, uintptr(unsafe.Pointer(&bci)), 0, uintptr(unsafe.Pointer(&buf)))
	if r != 0 {
		return nil, fmt.Errorf("vkCreateBuffer error: %d", r)
	}

	var reqs VkMemoryRequirements
	vkSyscall(c.getProc("vkGetBufferMemoryRequirements"), c.device, uintptr(buf), uintptr(unsafe.Pointer(&reqs)))

	memType := c.findMemoryType(reqs.MemoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
	if memType == -1 {
		vkSyscall(c.getProc("vkDestroyBuffer"), c.device, uintptr(buf), 0)
		return nil, errors.New("host visible + coherent memory type unavailable")
	}

	allocInfo := VkMemoryAllocateInfo{
		SType:           VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		AllocationSize:  reqs.Size,
		MemoryTypeIndex: uint32(memType),
	}
	var mem uint64
	r = vkSyscall(c.getProc("vkAllocateMemory"), c.device, uintptr(unsafe.Pointer(&allocInfo)), 0, uintptr(unsafe.Pointer(&mem)))
	if r != 0 {
		vkSyscall(c.getProc("vkDestroyBuffer"), c.device, uintptr(buf), 0)
		return nil, fmt.Errorf("vkAllocateMemory error: %d", r)
	}

	r = vkSyscall(c.getProc("vkBindBufferMemory"), c.device, uintptr(buf), uintptr(mem), 0)
	if r != 0 {
		vkSyscall(c.getProc("vkFreeMemory"), c.device, uintptr(mem), 0)
		vkSyscall(c.getProc("vkDestroyBuffer"), c.device, uintptr(buf), 0)
		return nil, fmt.Errorf("vkBindBufferMemory error: %d", r)
	}

	var mapped unsafe.Pointer
	r = vkSyscall(c.getProc("vkMapMemory"), c.device, uintptr(mem), 0, uintptr(size), 0, uintptr(unsafe.Pointer(&mapped)))
	if r != 0 {
		vkSyscall(c.getProc("vkFreeMemory"), c.device, uintptr(mem), 0)
		vkSyscall(c.getProc("vkDestroyBuffer"), c.device, uintptr(buf), 0)
		return nil, fmt.Errorf("vkMapMemory error: %d", r)
	}

	return &GpuBuffer{
		ctx:      c,
		buffer:   buf,
		memory:   mem,
		size:     size,
		capacity: size,
		mapped:   mapped,
		inUse:    true,
		isFreed:  false,
	}, nil
}

// WriteInt32 uploads an int32 slice directly into host-coherent GPU mapped memory with strict bounds checking.
func (b *GpuBuffer) WriteInt32(data []int32) error {
	if len(data) == 0 {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.isFreed || b.mapped == nil {
		return errors.New("gpu buffer memory is not mapped or has been freed")
	}
	reqBytes := uint64(len(data) * 4)
	if reqBytes > b.capacity {
		return fmt.Errorf("buffer overflow: writing %d int32 elements (%d bytes) exceeds buffer capacity of %d bytes", len(data), reqBytes, b.capacity)
	}
	if reqBytes > b.size {
		b.size = reqBytes
	}
	dst := unsafe.Slice((*int32)(b.mapped), len(data))
	copy(dst, data)
	return nil
}

// WriteUint32 uploads a uint32 slice directly into host-coherent GPU mapped memory with strict bounds checking.
func (b *GpuBuffer) WriteUint32(data []uint32) error {
	if len(data) == 0 {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.isFreed || b.mapped == nil {
		return errors.New("gpu buffer memory is not mapped or has been freed")
	}
	reqBytes := uint64(len(data) * 4)
	if reqBytes > b.capacity {
		return fmt.Errorf("buffer overflow: writing %d uint32 elements (%d bytes) exceeds buffer capacity of %d bytes", len(data), reqBytes, b.capacity)
	}
	if reqBytes > b.size {
		b.size = reqBytes
	}
	dst := unsafe.Slice((*uint32)(b.mapped), len(data))
	copy(dst, data)
	return nil
}

// WriteInt64 uploads an int64 slice directly into host-coherent GPU mapped memory with strict bounds checking.
func (b *GpuBuffer) WriteInt64(data []int64) error {
	if len(data) == 0 {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.isFreed || b.mapped == nil {
		return errors.New("gpu buffer memory is not mapped or has been freed")
	}
	reqBytes := uint64(len(data) * 8)
	if reqBytes > b.capacity {
		return fmt.Errorf("buffer overflow: writing %d int64 elements (%d bytes) exceeds buffer capacity of %d bytes", len(data), reqBytes, b.capacity)
	}
	if reqBytes > b.size {
		b.size = reqBytes
	}
	dst := unsafe.Slice((*int64)(b.mapped), len(data))
	copy(dst, data)
	return nil
}

// ReadInt32 reads back a copy of int32 elements into Go-owned memory with strict bounds checking.
func (b *GpuBuffer) ReadInt32(count int) ([]int32, error) {
	if count < 0 {
		return nil, fmt.Errorf("invalid read count: %d (must be non-negative)", count)
	}
	if count == 0 {
		return nil, nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.isFreed || b.mapped == nil {
		return nil, errors.New("gpu buffer memory is not mapped or has been freed")
	}
	reqBytes := uint64(count * 4)
	if reqBytes > b.capacity {
		return nil, fmt.Errorf("buffer read bounds error: reading %d int32 elements (%d bytes) exceeds buffer capacity of %d bytes", count, reqBytes, b.capacity)
	}
	if reqBytes > b.size {
		return nil, fmt.Errorf("buffer read bounds error: reading %d int32 elements (%d bytes) exceeds active buffer size of %d bytes", count, reqBytes, b.size)
	}
	src := unsafe.Slice((*int32)(b.mapped), count)
	out := make([]int32, count)
	copy(out, src)
	return out, nil
}

// ReadUint32 reads back a copy of uint32 elements into Go-owned memory with strict bounds checking.
func (b *GpuBuffer) ReadUint32(count int) ([]uint32, error) {
	if count < 0 {
		return nil, fmt.Errorf("invalid read count: %d (must be non-negative)", count)
	}
	if count == 0 {
		return nil, nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.isFreed || b.mapped == nil {
		return nil, errors.New("gpu buffer memory is not mapped or has been freed")
	}
	reqBytes := uint64(count * 4)
	if reqBytes > b.capacity {
		return nil, fmt.Errorf("buffer read bounds error: reading %d uint32 elements (%d bytes) exceeds buffer capacity of %d bytes", count, reqBytes, b.capacity)
	}
	if reqBytes > b.size {
		return nil, fmt.Errorf("buffer read bounds error: reading %d uint32 elements (%d bytes) exceeds active buffer size of %d bytes", count, reqBytes, b.size)
	}
	src := unsafe.Slice((*uint32)(b.mapped), count)
	out := make([]uint32, count)
	copy(out, src)
	return out, nil
}

// ReadInt64 reads back a copy of int64 elements into Go-owned memory with strict bounds checking.
func (b *GpuBuffer) ReadInt64(count int) ([]int64, error) {
	if count < 0 {
		return nil, fmt.Errorf("invalid read count: %d (must be non-negative)", count)
	}
	if count == 0 {
		return nil, nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.isFreed || b.mapped == nil {
		return nil, errors.New("gpu buffer memory is not mapped or has been freed")
	}
	reqBytes := uint64(count * 8)
	if reqBytes > b.capacity {
		return nil, fmt.Errorf("buffer read bounds error: reading %d int64 elements (%d bytes) exceeds buffer capacity of %d bytes", count, reqBytes, b.capacity)
	}
	if reqBytes > b.size {
		return nil, fmt.Errorf("buffer read bounds error: reading %d int64 elements (%d bytes) exceeds active buffer size of %d bytes", count, reqBytes, b.size)
	}
	src := unsafe.Slice((*int64)(b.mapped), count)
	out := make([]int64, count)
	copy(out, src)
	return out, nil
}

// Clear zeroes out the active buffer memory.
func (b *GpuBuffer) Clear() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.mapped != nil && b.capacity > 0 {
		dst := unsafe.Slice((*byte)(b.mapped), b.capacity)
		for i := range dst {
			dst[i] = 0
		}
	}
}

// Free releases the buffer and unmaps memory with double-free protection.
// If the buffer was tracked by a BufferPool, it is atomically evicted so subsequent pool acquisitions never reuse freed memory.
func (b *GpuBuffer) Free() {
	if b == nil {
		return
	}
	b.mu.Lock()
	if b.isFreed {
		b.mu.Unlock()
		return
	}
	b.isFreed = true
	b.inUse = false
	b.size = 0
	pool := b.pool
	b.pool = nil
	if b.mapped != nil {
		vkSyscall(b.ctx.getProc("vkUnmapMemory"), b.ctx.device, uintptr(b.memory))
		b.mapped = nil
	}
	if b.buffer != 0 {
		vkSyscall(b.ctx.getProc("vkDestroyBuffer"), b.ctx.device, uintptr(b.buffer), 0)
		b.buffer = 0
	}
	if b.memory != 0 {
		vkSyscall(b.ctx.getProc("vkFreeMemory"), b.ctx.device, uintptr(b.memory), 0)
		b.memory = 0
	}
	b.mu.Unlock()

	if pool != nil {
		pool.removeBuffer(b)
	}
}
