package main

import (
	"fmt"

	"gotrain/vulkankernel"
)

func main() {
	min, max := vulkankernel.MinMax64(2)
	fmt.Println("q2 range:", min, max)

t := vulkankernel.NewDynamicBitTensor([]int{4}, 2)
	for i := 0; i < 4; i++ {
		val := int64(1) << uint(2+i)
		newv := t.Add(0, val)
		fmt.Printf("add %2d -> value %6d, width now %d\n", val, newv, t.BitWidth())
	}

t0 := vulkankernel.NewDynamicBitTensor([]int{8}, 2)
	t0.SetInt32(0, 70000)
	fmt.Println("overflow set -> width:", t0.BitWidth())
}
