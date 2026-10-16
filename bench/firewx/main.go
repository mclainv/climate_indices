// Benchmark harness for alpineworks.io/firewx KBDI. Follows ../PROTOCOL.md.
//
// The firewx package is imported unmodified from the clone via the go.mod
// replace directive; nothing of its code is copied here.
//
// Variants:
//
//	kernel        simple.NewKBDIState + (*KBDIState).Step per day, typed
//	              []Fahrenheit / []Inches inputs (imperial, already converted).
//	kernel_mt     same loop, goroutines over contiguous cell ranges
//	              (GOMAXPROCS = all cores; harness-level parallelism).
//	api           simple.KBDIState.StepObs(firewx.Obs) per day: the documented
//	              end-user entry point (Obs carries Celsius and mm; the call does
//	              the unit conversion, Opt unwrapping and the Step).
//	nfdrs_driver  nfdrs.Driver.Update(Obs) once per day (observation placed at
//	              RegObsHour), which also runs 4 Nelson sticks, 2 GSI models and
//	              the four NFDRS indices on every update; KBDI is read back with
//	              Driver.State().KBDIValue.
package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	firewx "alpineworks.io/firewx"
	"alpineworks.io/firewx/nfdrs"
	"alpineworks.io/firewx/simple"
)

func readF64(path string) []float64 {
	b, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	out := make([]float64, len(b)/8)
	for i := range out {
		out[i] = math.Float64frombits(uint64(b[8*i]) | uint64(b[8*i+1])<<8 | uint64(b[8*i+2])<<16 | uint64(b[8*i+3])<<24 |
			uint64(b[8*i+4])<<32 | uint64(b[8*i+5])<<40 | uint64(b[8*i+6])<<48 | uint64(b[8*i+7])<<56)
	}
	return out
}

func metaInt(meta, key string) int {
	i := strings.Index(meta, "\""+key+"\"")
	rest := meta[i+len(key)+2:]
	rest = strings.TrimLeft(rest[strings.Index(rest, ":")+1:], " ")
	j := 0
	for j < len(rest) && rest[j] >= '0' && rest[j] <= '9' {
		j++
	}
	n, _ := strconv.Atoi(rest[:j])
	return n
}

// obsBytes is sizeof(firewx.Obs) on 64-bit; used only to size chunks.
const obsBytes = 184

// maxObsBytes: above this, Obs are built chunk by chunk (build untimed, step
// timed) so the 2500x3650 grid does not need ~1.7 GB of resident Obs.
const maxObsBytes = 600 << 20

func main() {
	var workload, variant, timings, values string
	args := os.Args[1:]
	for i := 0; i+1 < len(args); i += 2 {
		switch args[i] {
		case "--workload":
			workload = args[i+1]
		case "--variant":
			variant = args[i+1]
		case "--timings":
			timings = args[i+1]
		case "--values":
			values = args[i+1]
		default:
			panic("unknown flag " + args[i])
		}
	}
	if variant == "" {
		variant = "kernel"
	}
	if timings == "" {
		timings = "out/timings.jsonl"
	}
	meta, err := os.ReadFile(filepath.Join(workload, "meta.json"))
	if err != nil {
		panic(err)
	}
	cells, days := metaInt(string(meta), "cells"), metaInt(string(meta), "days")
	name := filepath.Base(workload)
	threads := 1
	if variant == "kernel_mt" {
		threads = runtime.NumCPU()
		if threads > cells {
			threads = cells
		}
	}
	runtime.GOMAXPROCS(threads)

	// ---- inputs, prepared before any timing ----
	mapIn := readF64(filepath.Join(workload, "map_in.bin"))
	var (
		precipIn []firewx.Inches
		tmaxF    []firewx.Fahrenheit
		precipMM []float64
		tmaxC    []float64
	)
	obsMode := variant == "api" || variant == "nfdrs_driver"
	if obsMode {
		precipMM = readF64(filepath.Join(workload, "precip_mm.bin"))
		tmaxC = readF64(filepath.Join(workload, "tmax_c.bin"))
	} else {
		pi, tf := readF64(filepath.Join(workload, "precip_in.bin")), readF64(filepath.Join(workload, "tmax_f.bin"))
		precipIn = make([]firewx.Inches, len(pi))
		tmaxF = make([]firewx.Fahrenheit, len(tf))
		for i := range pi {
			precipIn[i], tmaxF[i] = firewx.Inches(pi[i]), firewx.Fahrenheit(tf[i])
		}
	}

	// Obs builder for one cell (untimed). One observation per day at 12:00 UTC.
	base := time.Date(1991, 1, 1, 12, 0, 0, 0, time.UTC)
	buildObs := func(cell int, dst []firewx.Obs) {
		for d := 0; d < days; d++ {
			o := firewx.Obs{
				Time:          base.AddDate(0, 0, d),
				Temperature:   firewx.Some(firewx.Celsius(tmaxC[cell*days+d])),
				Precipitation: firewx.Some(firewx.Millimeters(precipMM[cell*days+d])),
			}
			if variant == "nfdrs_driver" {
				o.RelativeHumidity = firewx.Some(firewx.Percent(40))
				o.SolarRadiation = firewx.Some(firewx.WattsPerSquareMeter(0))
			}
			dst[d] = o
		}
	}
	chunkCells := cells
	if obsMode && cells*days*obsBytes > maxObsBytes {
		chunkCells = maxObsBytes / (days * obsBytes)
	}
	var obsBuf []firewx.Obs // chunkCells*days; built for all cells up front when chunkCells==cells
	if obsMode {
		obsBuf = make([]firewx.Obs, chunkCells*days)
		if chunkCells == cells {
			for c := 0; c < cells; c++ {
				buildObs(c, obsBuf[c*days:(c+1)*days])
			}
		}
	}

	driverCfg := func(c int) nfdrs.Config {
		return nfdrs.Config{
			FuelModel: nfdrs.FuelModelY, Latitude: 36.8, SlopeClass: 1, KBDIThreshold: 800,
			MeanAnnualPrecip: firewx.Inches(mapIn[c]), AnnualHerb: true, RegObsHour: 12,
		}
	}

	// runCells: one full recurrence over cells [a,b) into res (hundredths of an inch).
	// For Obs variants obsBuf holds cells a..b-1 at offset (c-a)*days.
	runCells := func(a, b int, res []float64) {
		switch variant {
		case "kernel", "kernel_mt":
			for c := a; c < b; c++ {
				s := simple.NewKBDIState(firewx.Inches(mapIn[c]))
				p, t, o := precipIn[c*days:(c+1)*days], tmaxF[c*days:(c+1)*days], res[c*days:(c+1)*days]
				for d := range p {
					s.Step(t[d], p[d])
					o[d] = float64(s.Index)
				}
			}
		case "api":
			for c := a; c < b; c++ {
				s := simple.NewKBDIState(firewx.Inches(mapIn[c]))
				ob, o := obsBuf[(c-a)*days:(c-a+1)*days], res[c*days:(c+1)*days]
				for d := range ob {
					s.StepObs(ob[d])
					o[d] = float64(s.Index)
				}
			}
		case "nfdrs_driver":
			for c := a; c < b; c++ {
				dr := nfdrs.NewDriver(driverCfg(c))
				ob, o := obsBuf[(c-a)*days:(c-a+1)*days], res[c*days:(c+1)*days]
				for d := range ob {
					dr.Update(ob[d])
					o[d] = dr.State().KBDIValue
				}
			}
		default:
			panic("unknown variant " + variant)
		}
	}

	// call: one complete recurrence, allocating the output; returns elapsed ns.
	var res []float64
	call := func() int64 {
		if obsMode && chunkCells < cells {
			var elapsed int64
			t0 := nowNS()
			res = make([]float64, cells*days)
			elapsed += nowNS() - t0
			for a := 0; a < cells; a += chunkCells {
				b := a + chunkCells
				if b > cells {
					b = cells
				}
				for c := a; c < b; c++ { // untimed
					buildObs(c, obsBuf[(c-a)*days:(c-a+1)*days])
				}
				t1 := nowNS()
				runCells(a, b, res)
				elapsed += nowNS() - t1
			}
			return elapsed
		}
		t0 := nowNS()
		res = make([]float64, cells*days)
		if threads == 1 {
			runCells(0, cells, res)
		} else {
			var wg sync.WaitGroup
			for k := 0; k < threads; k++ {
				wg.Add(1)
				go func(a, b int) { defer wg.Done(); runCells(a, b, res) }(k*cells/threads, (k+1)*cells/threads)
			}
			wg.Wait()
		}
		return nowNS() - t0
	}
	checksum := func() float64 {
		s := 0.0
		for _, v := range res {
			s += v * 0.254
		}
		return s
	}

	firstCall := call()
	sum := checksum()
	warm := time.Now()
	for n := 1; n < 5 || time.Since(warm) < 500*time.Millisecond; n++ {
		call()
	}
	minSamples := 15
	if cells == 1 {
		minSamples = 200
	}
	var samples []int64
	started := time.Now()
	for {
		ns := call()
		samples = append(samples, ns)
		sum = checksum()
		el := time.Since(started)
		if (len(samples) >= minSamples && el >= 3*time.Second) || (el >= 60*time.Second && len(samples) >= 3) {
			break
		}
	}

	if values != "" {
		if err := os.MkdirAll(filepath.Dir(values), 0o755); err != nil {
			panic(err)
		}
		b := make([]byte, 8*len(res))
		for i, v := range res {
			u := math.Float64bits(v * 0.254)
			for k := 0; k < 8; k++ {
				b[8*i+k] = byte(u >> (8 * k))
			}
		}
		if err := os.WriteFile(values, b, 0o644); err != nil {
			panic(err)
		}
	}

	notes := map[string]string{
		"kernel":       "simple.KBDIState.Step, typed Inches/Fahrenheit slices, float64, GOMAXPROCS=1",
		"kernel_mt":    "simple.KBDIState.Step over cell ranges on goroutines (harness-level parallelism)",
		"api":          "simple.KBDIState.StepObs(firewx.Obs), Celsius/mm Obs prebuilt, float64, GOMAXPROCS=1",
		"nfdrs_driver": "nfdrs.Driver.Update (1 obs/day at RegObsHour; also 4 Nelson sticks, 2 GSI, NFDRS indices) + State().KBDIValue, GOMAXPROCS=1",
	}[variant]
	if obsMode && chunkCells < cells {
		notes += fmt.Sprintf("; Obs built per %d-cell chunk outside the timer (timed = sum of chunk step segments)", chunkCells)
	}
	notes += "; clock=" + clockName + "; " + runtime.Version() + " " + runtime.GOOS + "/" + runtime.GOARCH
	strs := make([]string, len(samples))
	for i, s := range samples {
		strs[i] = strconv.FormatInt(s, 10)
	}
	line := fmt.Sprintf("{\"impl\": \"firewx\", \"variant\": \"%s\", \"workload\": \"%s\", \"cells\": %d, \"days\": %d, \"threads\": %d, \"samples_ns\": [%s], \"first_call_ns\": %d, \"checksum\": %s, \"notes\": \"%s\"}\n",
		variant, name, cells, days, threads, strings.Join(strs, ","), firstCall, strconv.FormatFloat(sum, 'g', -1, 64), notes)
	if err := os.MkdirAll(filepath.Dir(timings), 0o755); err != nil {
		panic(err)
	}
	f, err := os.OpenFile(timings, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		panic(err)
	}
	f.WriteString(line)
	f.Close()
	sorted := append([]int64(nil), samples...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	fmt.Fprintf(os.Stderr, "%s %s %d: median %.4f ms, min %.4f ms, n=%d, checksum %v\n", variant, name, threads,
		float64(sorted[len(sorted)/2])/1e6, float64(sorted[0])/1e6, len(samples), sum)
}
