package voltgrad

import (
	"testing"
)

func TestBufferPoolReuseAndClear(t *testing.T) {
	hw, err := NewHardwareContext()
	if err != nil {
		t.Skipf("Skipping Vulkan test (hardware initialization: %v)", err)
	}
	defer hw.Close()

	pool := hw.BufferPool()
	if pool == nil {
		t.Fatalf("Expected non-nil buffer pool on hardware context")
	}

	// 1. Request buffer of size 100 -> should be allocated with capacity 256
	buf1, err := hw.GetPooledBuffer(100)
	if err != nil {
		t.Fatalf("Failed to get pooled buffer: %v", err)
	}
	if buf1.Capacity() != 256 {
		t.Fatalf("Expected capacity 256, got %d", buf1.Capacity())
	}
	if buf1.Size() != 100 {
		t.Fatalf("Expected size 100, got %d", buf1.Size())
	}
	addr1 := buf1.mapped

	// Write data to verify valid mapping
	testData := []int32{1, 2, 3, 4}
	if err := buf1.WriteInt32(testData); err != nil {
		t.Fatalf("WriteInt32 failed: %v", err)
	}
	readData, err := buf1.ReadInt32(4)
	if err != nil {
		t.Fatalf("ReadInt32 failed: %v", err)
	}
	if readData[0] != 1 || readData[3] != 4 {
		t.Fatalf("Data mismatch in pooled buffer: %v", readData)
	}

	// 2. Release buffer back to pool
	buf1.Release()

	// 3. Request buffer of size 150 -> should reuse the exact same buffer!
	buf2, err := hw.GetPooledBuffer(150)
	if err != nil {
		t.Fatalf("Failed to get second pooled buffer: %v", err)
	}
	defer buf2.Release()

	if buf2.Capacity() != 256 {
		t.Fatalf("Expected capacity 256, got %d", buf2.Capacity())
	}
	if buf2.Size() != 150 {
		t.Fatalf("Expected size 150, got %d", buf2.Size())
	}
	if buf2.mapped != addr1 {
		t.Fatalf("Expected buffer memory address to be reused from pool (%p vs %p)", buf2.mapped, addr1)
	}

	// 4. Request larger buffer (e.g. 500 -> capacity 512)
	buf3, err := hw.GetPooledBuffer(500)
	if err != nil {
		t.Fatalf("Failed to get 500-byte pooled buffer: %v", err)
	}
	if buf3.Capacity() != 512 {
		t.Fatalf("Expected capacity 512, got %d", buf3.Capacity())
	}
	buf3.Release()

	// 5. Clear pool
	pool.Clear()
	if len(pool.pools[256]) != 0 || len(pool.pools[512]) != 0 {
		t.Fatalf("Expected empty pool buckets after Clear()")
	}

	t.Log(">>> BufferPool reuse, capacity bucketing, and clearing verified! <<<")
}

func TestBufferDoubleReleaseAndLifecycleSafety(t *testing.T) {
	hw, err := NewHardwareContext()
	if err != nil {
		t.Skipf("Skipping Vulkan test: %v", err)
	}
	defer hw.Close()

	pool := hw.BufferPool()

	// 1. Double Release Protection
	buf, err := hw.GetPooledBuffer(64)
	if err != nil {
		t.Fatalf("Failed to get pooled buffer: %v", err)
	}
	if !buf.InUse() {
		t.Fatalf("Expected buffer to be in-use after Get")
	}

	// First release
	buf.Release()
	if buf.InUse() {
		t.Fatalf("Expected buffer to not be in-use after Release")
	}

	bucketLen := len(pool.pools[buf.Capacity()])
	if bucketLen != 1 {
		t.Fatalf("Expected bucket length 1, got %d", bucketLen)
	}

	// Second release (must be safe no-op and NOT duplicate buffer in pool)
	buf.Release()
	if len(pool.pools[buf.Capacity()]) != 1 {
		t.Fatalf("Double release duplicated buffer in pool bucket!")
	}

	// 2. Double Free Protection on standalone buffer
	rawBuf, err := hw.CreateHostCoherentBuffer(128)
	if err != nil {
		t.Fatalf("CreateHostCoherentBuffer failed: %v", err)
	}
	if rawBuf.IsFreed() {
		t.Fatalf("New buffer marked as freed")
	}

	rawBuf.Free()
	if !rawBuf.IsFreed() {
		t.Fatalf("Buffer not marked as freed after Free()")
	}

	// Second Free call must not crash or error
	rawBuf.Free()

	// Release on freed buffer must be a safe no-op
	rawBuf.Release()

	// Write on freed buffer must return an error
	if err := rawBuf.WriteInt32([]int32{1, 2, 3}); err == nil {
		t.Fatalf("Expected error writing to freed buffer, got nil")
	}

	// Read on freed buffer must return an error
	if _, err := rawBuf.ReadInt32(2); err == nil {
		t.Fatalf("Expected error reading from freed buffer, got nil")
	}

	t.Log(">>> Buffer Double-Release and Lifecycle Safety VERIFIED! <<<")
}

func TestBufferActiveSizeBounds(t *testing.T) {
	hw, err := NewHardwareContext()
	if err != nil {
		t.Skipf("Skipping Vulkan test: %v", err)
	}
	defer hw.Close()

	// Allocate with active size 32 bytes (which will have capacity 256)
	buf, err := hw.GetPooledBuffer(32)
	if err != nil {
		t.Fatalf("GetPooledBuffer failed: %v", err)
	}
	defer buf.Release()

	if buf.Size() != 32 {
		t.Fatalf("Expected active size 32, got %d", buf.Size())
	}
	if buf.Capacity() != 256 {
		t.Fatalf("Expected capacity 256, got %d", buf.Capacity())
	}

	// Writing 8 int32 elements (32 bytes) matches active size
	data := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	if err := buf.WriteInt32(data); err != nil {
		t.Fatalf("WriteInt32 failed: %v", err)
	}

	// Reading 8 elements (32 bytes) matches active size
	readOut, err := buf.ReadInt32(8)
	if err != nil {
		t.Fatalf("ReadInt32 failed: %v", err)
	}
	if len(readOut) != 8 || readOut[7] != 8 {
		t.Fatalf("ReadInt32 content mismatch: %v", readOut)
	}

	// Reading 9 elements (36 bytes) must fail because 36 > active size (32),
	// even though 36 is within capacity (256)
	if _, err := buf.ReadInt32(9); err == nil {
		t.Fatalf("Expected error reading 9 elements exceeding active size 32, got nil")
	}

	// Writing 10 elements (40 bytes) expands active size up to 40 bytes
	moreData := make([]int32, 10)
	if err := buf.WriteInt32(moreData); err != nil {
		t.Fatalf("WriteInt32 expanding size failed: %v", err)
	}
	if buf.Size() != 40 {
		t.Fatalf("Expected expanded size 40, got %d", buf.Size())
	}

	// Now reading 10 elements succeeds!
	if _, err := buf.ReadInt32(10); err != nil {
		t.Fatalf("ReadInt32 after expansion failed: %v", err)
	}

	// Setting active size within capacity succeeds
	if err := buf.SetActiveSize(64); err != nil {
		t.Fatalf("SetActiveSize(64) failed: %v", err)
	}
	if buf.Size() != 64 {
		t.Fatalf("Expected size 64, got %d", buf.Size())
	}

	// Setting active size > capacity must fail
	if err := buf.SetActiveSize(300); err == nil {
		t.Fatalf("Expected error setting active size > capacity, got nil")
	}

	t.Log(">>> Buffer Active-Size Bounds Safety VERIFIED! <<<")
}
