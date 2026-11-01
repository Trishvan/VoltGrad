package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
)

var shaderFiles = []string{
	"matmul_forward",
	"matmul_backward_dx",
	"matmul_backward_dw",
	"matmul_batched_forward",
	"matmul_batched_backward_dx",
	"matmul_batched_backward_dw",
	"relu_forward",
	"relu_backward",
	"loss_mse",
	"optimizer_step",
	"repack_dbw",
}

func findCompiler() (string, string) {
	// Look for glslc in PATH first
	if p, err := exec.LookPath("glslc"); err == nil {
		return "glslc", p
	}
	// Look for glslangValidator in PATH
	if p, err := exec.LookPath("glslangValidator"); err == nil {
		return "glslangValidator", p
	}

	// Check VULKAN_SDK environment variable
	if sdk := os.Getenv("VULKAN_SDK"); sdk != "" {
		ext := ""
		if runtime.GOOS == "windows" {
			ext = ".exe"
		}
		candidates := []struct {
			cType string
			path  string
		}{
			{"glslc", filepath.Join(sdk, "bin", "glslc"+ext)},
			{"glslc", filepath.Join(sdk, "Bin", "glslc"+ext)},
			{"glslangValidator", filepath.Join(sdk, "bin", "glslangValidator"+ext)},
			{"glslangValidator", filepath.Join(sdk, "Bin", "glslangValidator"+ext)},
		}
		for _, c := range candidates {
			if _, err := os.Stat(c.path); err == nil {
				return c.cType, c.path
			}
		}
	}

	return "", ""
}

func main() {
	compilerType, compilerPath := findCompiler()
	if compilerPath == "" {
		fmt.Fprintln(os.Stderr, "Error: No native SPIR-V shader compiler (glslc or glslangValidator) found.")
		fmt.Fprintln(os.Stderr)
		fmt.Fprintln(os.Stderr, "To compile GLSL compute shaders without Node.js or JavaScript:")
		fmt.Fprintln(os.Stderr, "  - Windows / Linux / macOS: Install Vulkan SDK from https://vulkan.lunarg.com/")
		fmt.Fprintln(os.Stderr, "  - macOS (Homebrew):       brew install shaderc")
		fmt.Fprintln(os.Stderr, "  - Ubuntu / Debian:         sudo apt install glslc")
		fmt.Fprintln(os.Stderr)
		fmt.Fprintln(os.Stderr, "Note: Precompiled SPIR-V (.spv) bytecode is already included and embedded")
		fmt.Fprintln(os.Stderr, "in VoltGrad, so recompilation is only required when modifying .comp source files.")
		os.Exit(1)
	}

	fmt.Printf("Using native shader compiler: %s (%s)\n", compilerType, compilerPath)

	shaderDir := filepath.Join("voltgrad", "shaders")
	if _, err := os.Stat(shaderDir); os.IsNotExist(err) {
		if _, err := os.Stat("matmul_forward.comp"); err == nil {
			shaderDir = "."
		} else {
			fmt.Fprintf(os.Stderr, "Error: Cannot find shaders directory (%s)\n", shaderDir)
			os.Exit(1)
		}
	}

	for _, name := range shaderFiles {
		compFile := filepath.Join(shaderDir, name+".comp")
		spvFile := filepath.Join(shaderDir, name+".spv")

		var cmd *exec.Cmd
		if compilerType == "glslc" {
			cmd = exec.Command(compilerPath, "-fshader-stage=compute", compFile, "-o", spvFile)
		} else {
			cmd = exec.Command(compilerPath, "-V", compFile, "-o", spvFile)
		}

		out, err := cmd.CombinedOutput()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Compilation error in %s:\n%s\n", compFile, string(out))
			os.Exit(1)
		}

		info, _ := os.Stat(spvFile)
		size := int64(0)
		if info != nil {
			size = info.Size()
		}
		fmt.Printf("Compiled %s -> %s (%d bytes)\n", compFile, spvFile, size)
	}

	fmt.Printf("All %d compute shaders compiled successfully to SPIR-V!\n", len(shaderFiles))
}
