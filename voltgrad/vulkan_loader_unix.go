//go:build !windows

package voltgrad

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/ebitengine/purego"
)

type platformLoader struct {
	handle                uintptr
	vkGetInstanceProcAddr uintptr
}

func newPlatformLoader() (*platformLoader, error) {
	var candidates []string
	if runtime.GOOS == "darwin" {
		candidates = []string{
			"libvulkan.1.dylib",
			"libMoltenVK.dylib",
			"/usr/local/lib/libvulkan.dylib",
			"/opt/homebrew/lib/libvulkan.dylib",
		}
	} else {
		candidates = []string{
			"libvulkan.so.1",
			"libvulkan.so",
		}
	}

	var handle uintptr
	var lastErr error
	for _, cand := range candidates {
		h, err := purego.Dlopen(cand, purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err == nil && h != 0 {
			handle = h
			break
		}
		lastErr = err
	}
	if handle == 0 {
		return nil, fmt.Errorf("failed loading Vulkan shared library (%v): %w", candidates, lastErr)
	}

	proc, err := purego.Dlsym(handle, "vkGetInstanceProcAddr")
	if err != nil {
		purego.Dlclose(handle)
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
		purego.Dlclose(l.handle)
		l.handle = 0
	}
}
