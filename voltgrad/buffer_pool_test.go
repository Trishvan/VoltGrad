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
