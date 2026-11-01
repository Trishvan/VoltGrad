package voltgrad

import (
	"fmt"
	"sync"
)

// BufferPool manages reusable host-coherent GPU memory buffers to eliminate repeated Vulkan allocations.
type BufferPool struct {
	hw    *HardwareContext
	mu    sync.Mutex
	pools map[uint64][]*GpuBuffer // Keyed by power-of-two capacity
}

// NewBufferPool creates an empty buffer pool for the given hardware context.
func NewBufferPool(hw *HardwareContext) *BufferPool {
	return &BufferPool{
		hw:    hw,
		pools: make(map[uint64][]*GpuBuffer),
	}
}

func nextPowerOf2(n uint64) uint64 {
	if n <= 256 {
		return 256
	}
	n--
	n |= n >> 1
	n |= n >> 2
	n |= n >> 4
	n |= n >> 8
	n |= n >> 16
	n |= n >> 32
	n++
	return n
}

// Get retrieves a pooled buffer with at least the requested size, or allocates a new one.
func (p *BufferPool) Get(size uint64) (*GpuBuffer, error) {
	if size == 0 {
		size = 4
	}
	capSize := nextPowerOf2(size)

	p.mu.Lock()
	list := p.pools[capSize]
	if len(list) > 0 {
		buf := list[len(list)-1]
		p.pools[capSize] = list[:len(list)-1]
		p.mu.Unlock()
		buf.size = size
		return buf, nil
	}
	p.mu.Unlock()

	// Allocate new buffer with capSize
	buf, err := p.hw.CreateHostCoherentBuffer(capSize)
	if err != nil {
		return nil, fmt.Errorf("buffer pool allocation failed: %w", err)
	}
	buf.pool = p
	buf.capacity = capSize
	buf.size = size
	return buf, nil
}

// Put returns a buffer back to the pool for reuse.
func (p *BufferPool) Put(buf *GpuBuffer) {
	if buf == nil || buf.mapped == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.pools[buf.capacity] = append(p.pools[buf.capacity], buf)
}

// Clear releases all pooled GPU buffers from memory.
func (p *BufferPool) Clear() {
	p.mu.Lock()
	defer p.mu.Unlock()

	for capSize, list := range p.pools {
		for _, b := range list {
			b.pool = nil
			b.Free()
		}
		delete(p.pools, capSize)
	}
}

// Release returns the buffer to its pool if pooled, or frees it immediately.
func (b *GpuBuffer) Release() {
	if b == nil {
		return
	}
	if b.pool != nil {
		b.pool.Put(b)
	} else {
		b.Free()
	}
}

// Capacity returns the total allocated capacity of the buffer.
func (b *GpuBuffer) Capacity() uint64 {
	return b.capacity
}

// Size returns the currently active size of the buffer.
func (b *GpuBuffer) Size() uint64 {
	return b.size
}
