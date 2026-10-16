package main

import (
	"syscall"
	"unsafe"
)

// Go's runtime clock on Windows (time.Now / nanotime) ticks at the system timer
// resolution (~0.5-15 ms), far too coarse for the single-station workloads, so the
// harness reads QueryPerformanceCounter directly (sub-100 ns tick, monotonic).

var (
	kernel32 = syscall.NewLazyDLL("kernel32.dll")
	qpc      = kernel32.NewProc("QueryPerformanceCounter")
	qpf      = kernel32.NewProc("QueryPerformanceFrequency")
	qpcFreq  = func() int64 {
		var f int64
		qpf.Call(uintptr(unsafe.Pointer(&f)))
		return f
	}()
)

const clockName = "QPC"

// nowNS returns a monotonic timestamp in nanoseconds.
func nowNS() int64 {
	var c int64
	qpc.Call(uintptr(unsafe.Pointer(&c)))
	// split to avoid overflow of c*1e9
	return c/qpcFreq*1_000_000_000 + c%qpcFreq*1_000_000_000/qpcFreq
}
