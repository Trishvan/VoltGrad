//go:build windows

package core

import (
	"syscall"
)

// vkSyscall invokes a native C function pointer across Vulkan drivers on Windows.
func vkSyscall(fn uintptr, args ...uintptr) uintptr {
	if fn == 0 {
		return 0
	}
	r, _, _ := syscall.SyscallN(fn, args...)
	return r
}
