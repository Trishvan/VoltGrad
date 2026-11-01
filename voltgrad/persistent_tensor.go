package voltgrad

import (
	"fmt"
	"sync"
)

// PersistentGpuTensor maintains tensor data persistently allocated in Vulkan GPU memory across training iterations.
// It eliminates repetitive host-to-device uploads and device-to-host readbacks on every forward/backward/optimizer step.
type PersistentGpuTensor struct {
	hw      *HardwareContext
	tensor  *DynamicBitTensor
	gpuBuf  *GpuBuffer
	isDirty bool
	mu      sync.RWMutex
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

// Allocate provisions the persistent host-coherent storage buffer on the GPU device.
func (pt *PersistentGpuTensor) Allocate() error {
	pt.mu.Lock()
	defer pt.mu.Unlock()

	// Ensure capacity to hold either packed words (uint32) or 64-bit uncompressed elements (int64)
	neededBytes := uint64(pt.tensor.Len() * 8)
	wordsBytes := uint64(len(pt.tensor.Words()) * 4)
	if wordsBytes > neededBytes {
		neededBytes = wordsBytes
	}
	if neededBytes < 64 {
		neededBytes = 64
	}

	buf, err := pt.hw.CreateHostCoherentBuffer(neededBytes)
	if err != nil {
		return fmt.Errorf("failed to allocate persistent GPU buffer: %w", err)
	}
	pt.gpuBuf = buf
	return nil
}

// SyncToGPU writes the CPU tensor's current data into the resident GPU buffer.
func (pt *PersistentGpuTensor) SyncToGPU() error {
	pt.mu.Lock()
	defer pt.mu.Unlock()

	if pt.gpuBuf == nil {
		return fmt.Errorf("persistent GPU buffer is not allocated")
	}

	words := pt.tensor.Words()
	if err := pt.gpuBuf.WriteUint32(words); err != nil {
		return fmt.Errorf("failed to sync packed words to GPU: %w", err)
	}
	pt.isDirty = false
	return nil
}

// SyncToHost reads back the GPU buffer's contents into the CPU DynamicBitTensor.
func (pt *PersistentGpuTensor) SyncToHost() error {
	pt.mu.Lock()
	defer pt.mu.Unlock()

	if pt.gpuBuf == nil {
		return fmt.Errorf("persistent GPU buffer is not allocated")
	}

	numWords := len(pt.tensor.Words())
	words, err := pt.gpuBuf.ReadUint32(numWords)
	if err != nil {
		return fmt.Errorf("failed to read back persistent GPU buffer: %w", err)
	}
	pt.tensor.SetWords(words, pt.tensor.BitWidth())
	pt.isDirty = false
	return nil
}

// Buffer returns the underlying active persistent GpuBuffer.
func (pt *PersistentGpuTensor) Buffer() *GpuBuffer {
	pt.mu.RLock()
	defer pt.mu.RUnlock()
	return pt.gpuBuf
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

// Free releases the persistent GPU buffer.
func (pt *PersistentGpuTensor) Free() {
	pt.mu.Lock()
	defer pt.mu.Unlock()
	if pt.gpuBuf != nil {
		pt.gpuBuf.Free()
		pt.gpuBuf = nil
	}
}
