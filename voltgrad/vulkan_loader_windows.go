//go:build windows

package voltgrad

import (
	"fmt"
	"syscall"
	"unsafe"
)

type platformLoader struct {
	handle                syscall.Handle
	vkGetInstanceProcAddr uintptr
}

func newPlatformLoader() (*platformLoader, error) {
	handle, err := syscall.LoadLibrary("vulkan-1.dll")
	if err != nil {
		return nil, fmt.Errorf("vulkan-1.dll dynamic load failure: %w", err)
	}
	proc, err := syscall.GetProcAddress(handle, "vkGetInstanceProcAddr")
	if err != nil {
		syscall.FreeLibrary(handle)
		return nil, fmt.Errorf("failed locating vkGetInstanceProcAddr: %w", err)
	}
	return &platformLoader{
		handle:                handle,
		vkGetInstanceProcAddr: proc,
	}, nil
}

func (l *platformLoader) getProc(instance uintptr, name string) uintptr {
	cName := append([]byte(name), 0)
	return vkSyscall(l.vkGetInstanceProcAddr, instance, uintptr(unsafe.Pointer(&cName[0])))
}

func (l *platformLoader) close() {
	if l.handle != 0 {
		syscall.FreeLibrary(l.handle)
		l.handle = 0
	}
}
