//go:build !windows

package voltgrad

import (
	"github.com/ebitengine/purego"
)

// vkSyscall invokes a native C function pointer across Vulkan drivers on Unix/Linux/macOS without CGO.
func vkSyscall(fn uintptr, args ...uintptr) uintptr {
	if fn == 0 {
		return 0
	}
	r, _, _ := purego.SyscallN(fn, args...)
	return r
}
