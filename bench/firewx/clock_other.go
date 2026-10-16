//go:build !windows

package main

import "time"

var epoch = time.Now()

const clockName = "time.Now"

func nowNS() int64 { return int64(time.Since(epoch)) }
