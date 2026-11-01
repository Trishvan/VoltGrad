package voltgrad

import (
	"fmt"
	"sync"
)

// PersistentGpuTensor maintains tensor data persistently allocated in Vulkan GPU memory across training iterations.
// It holds both packed DBW words, 64-bit parameter representations, gradients, and velocities in host-coherent GPU memory,
// completely eliminating repetitive host-to-device uploads and device-to-host readbacks on every training step.
type PersistentGpuTensor struct {
	hw        *HardwareContext
	tensor    *DynamicBitTensor
	bufPacked *GpuBuffer
	bufInt64  *GpuBuffer
	bufGrad   *GpuBuffer
	bufVel    *GpuBuffer
	isDirty   bool
	mu        sync.RWMutex
}

// NewPersistentGpuTensor creates a GPU-resident persistent tensor mirroring a DynamicBitTensor.
func NewPersistentGpuTensor(hw *HardwareContext, t *DynamicBitTensor) (*PersistentGpuTensor, error) {
	if hw == nil {
		return nil, fmt.Errorf("hardware context cannot be nil")
	}
	if t == nil {
		return nil, fmt.Errorf("tensor cannot be nil")
	}

	pt := &PersistentGpuTensor{
		hw:     hw,
		tensor: t,
	}

	if err := pt.Allocate(); err != nil {
		return nil, err
	}
	if err := pt.SyncToGPU(); err != nil {
		pt.Free()
		return nil, err
	}

	return pt, nil
}

// Allocate provisions the persistent host-coherent storage buffers on the GPU device.
func (pt *PersistentGpuTensor) Allocate() error {
	pt.mu.Lock()
	defer pt.mu.Unlock()

	// Ensure capacity to hold up to 64-bit packed words or 64-bit uncompressed elements
	neededBytes := uint64(pt.tensor.Len() * 8)
	if neededBytes < 64 {
		neededBytes = 64
	}

	var err error
	pt.bufPacked, err = pt.hw.CreateHostCoherentBuffer(neededBytes)
	if err != nil {
		return fmt.Errorf("failed allocating persistent packed buffer: %w", err)
	}

	pt.bufInt64, err = pt.hw.CreateHostCoherentBuffer(neededBytes)
	if err != nil {
		pt.bufPacked.Free()
		pt.bufPacked = nil
		return fmt.Errorf("failed allocating persistent int64 buffer: %w", err)
	}

	pt.bufGrad, err = pt.hw.CreateHostCoherentBuffer(neededBytes)
	if err != nil {
		pt.bufPacked.Free()
		pt.bufInt64.Free()
		pt.bufPacked = nil
		pt.bufInt64 = nil
		return fmt.Errorf("failed allocating persistent grad buffer: %w", err)
	}

	pt.bufVel, err = pt.hw.CreateHostCoherentBuffer(neededBytes)
	if err != nil {
		pt.bufPacked.Free()
		pt.bufInt64.Free()
		pt.bufGrad.Free()
		pt.bufPacked = nil
		pt.bufInt64 = nil
		pt.bufGrad = nil
		return fmt.Errorf("failed allocating persistent vel buffer: %w", err)
	}

	// Zero-initialize buffers
	pt.bufPacked.Clear()
	pt.bufInt64.Clear()
	pt.bufGrad.Clear()
	pt.bufVel.Clear()

	return nil
}

// SyncToGPU writes the CPU tensor's current data into the resident GPU buffers.
func (pt *PersistentGpuTensor) SyncToGPU() error {
	pt.mu.Lock()
	defer pt.mu.Unlock()

	if pt.bufPacked == nil || pt.bufInt64 == nil {
		return fmt.Errorf("persistent GPU buffers are not allocated")
	}

	words := pt.tensor.Words()
	if err := pt.bufPacked.SetActiveSize(uint64(len(words) * 4)); err != nil {
		return fmt.Errorf("failed to set active size on packed buffer: %w", err)
	}
	if err := pt.bufPacked.WriteUint32(words); err != nil {
		return fmt.Errorf("failed to sync packed words to GPU: %w", err)
	}

	int64s := pt.tensor.ToInt64Slice()
	if err := pt.bufInt64.SetActiveSize(uint64(len(int64s) * 8)); err != nil {
		return fmt.Errorf("failed to set active size on int64 buffer: %w", err)
	}
	if err := pt.bufInt64.WriteInt64(int64s); err != nil {
		return fmt.Errorf("failed to sync int64 elements to GPU: %w", err)
	}

	pt.isDirty = false
	return nil
}

// SyncToHost reads back the GPU buffer's contents into the CPU DynamicBitTensor.
func (pt *PersistentGpuTensor) SyncToHost() error {
	pt.mu.Lock()
	defer pt.mu.Unlock()

	if pt.bufInt64 == nil {
		return fmt.Errorf("persistent GPU buffer is not allocated")
	}

	numElements := pt.tensor.Len()
	int64s, err := pt.bufInt64.ReadInt64(numElements)
	if err != nil {
		return fmt.Errorf("failed to read back persistent GPU int64 buffer: %w", err)
	}
	pt.tensor.FromInt64Slice(int64s)

	if pt.bufPacked != nil {
		numWords := len(pt.tensor.Words())
		if words, err := pt.bufPacked.ReadUint32(numWords); err == nil {
			pt.tensor.SetWords(words, pt.tensor.BitWidth())
		}
	}

	pt.isDirty = false
	return nil
}

// Buffer returns the primary packed GPU buffer for compute kernel execution.
func (pt *PersistentGpuTensor) Buffer() *GpuBuffer {
	pt.mu.RLock()
	defer pt.mu.RUnlock()
	return pt.bufPacked
}

// BufferPacked returns the packed DBW words GPU buffer.
func (pt *PersistentGpuTensor) BufferPacked() *GpuBuffer {
	pt.mu.RLock()
	defer pt.mu.RUnlock()
	return pt.bufPacked
}

// BufferInt64 returns the 64-bit element GPU buffer used by the optimizer.
func (pt *PersistentGpuTensor) BufferInt64() *GpuBuffer {
	pt.mu.RLock()
	defer pt.mu.RUnlock()
	return pt.bufInt64
}

// BufferGrad returns the 64-bit gradient accumulation GPU buffer.
func (pt *PersistentGpuTensor) BufferGrad() *GpuBuffer {
	pt.mu.RLock()
	defer pt.mu.RUnlock()
	return pt.bufGrad
}

// BufferVel returns the 64-bit optimizer velocity GPU buffer.
func (pt *PersistentGpuTensor) BufferVel() *GpuBuffer {
	pt.mu.RLock()
	defer pt.mu.RUnlock()
	return pt.bufVel
}

// IsAllocated reports whether the GPU buffers are allocated and ready.
func (pt *PersistentGpuTensor) IsAllocated() bool {
	pt.mu.RLock()
	defer pt.mu.RUnlock()
	return pt.bufPacked != nil && pt.bufInt64 != nil
}

// ClearGrad zeroes out the resident GPU gradient buffer.
func (pt *PersistentGpuTensor) ClearGrad() {
	pt.mu.Lock()
	defer pt.mu.Unlock()
	if pt.bufGrad != nil {
		pt.bufGrad.Clear()
	}
}

// Tensor returns the underlying DynamicBitTensor.
func (pt *PersistentGpuTensor) Tensor() *DynamicBitTensor {
	return pt.tensor
}

// MarkDirty marks that the GPU buffer has been mutated by an in-place GPU kernel.
func (pt *PersistentGpuTensor) MarkDirty() {
	pt.mu.Lock()
	pt.isDirty = true
	pt.mu.Unlock()
}

// IsDirty returns true if the GPU buffer has updates not yet synchronized to CPU.
func (pt *PersistentGpuTensor) IsDirty() bool {
	pt.mu.RLock()
	defer pt.mu.RUnlock()
	return pt.isDirty
}

// Free releases all persistent GPU buffers.
func (pt *PersistentGpuTensor) Free() {
	pt.mu.Lock()
	defer pt.mu.Unlock()
	if pt.bufPacked != nil {
		pt.bufPacked.Free()
		pt.bufPacked = nil
	}
	if pt.bufInt64 != nil {
		pt.bufInt64.Free()
		pt.bufInt64 = nil
	}
	if pt.bufGrad != nil {
		pt.bufGrad.Free()
		pt.bufGrad = nil
	}
	if pt.bufVel != nil {
		pt.bufVel.Free()
		pt.bufVel = nil
	}
}
