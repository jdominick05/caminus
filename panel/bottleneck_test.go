// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"strings"
	"testing"

	"github.com/charmbracelet/x/ansi"
)

func TestBottleneck(t *testing.T) {
	base := sysState{OK: true, VramGB: 21, VramTot: 24, Pid: 1, Spill: 0.35, ProcSwap: -1, GPUUtil: 95, Cores: 0.4}
	with := func(f func(*sysState)) sysState { s := base; f(&s); return s }
	cases := []struct {
		name string
		busy bool
		ss   sysState
		want string
	}{
		{"healthy", true, base, "GPU-bound"},
		{"idle", false, base, "3.0 GB VRAM headroom"},
		{"spill busy", true, with(func(s *sysState) { s.Spill = 1.2; s.VramGB = 23.6 }), "VRAM full: 1.2 GB spilled"},
		{"spill idle", false, with(func(s *sysState) { s.Spill = 0.9; s.VramGB = 23.6 }), "VRAM full"},
		{"spill with VRAM to spare (not a fit problem)", true, with(func(s *sysState) { s.Spill = 1.2 }), "GPU-bound"},
		{"iGPU: shared memory is its memory", true, with(func(s *sysState) { s.Spill = 14; s.VramGB = 0.5; s.VramTot = 0.5 }), "GPU-bound"},
		{"paging beats spill", true, with(func(s *sysState) { s.Spill = 2; s.VramGB = 23.6; s.PageIn = 5000; s.RamTot = 32; s.RamFree = 0.8 }), "paging: 20 MB/s"},
		{"swap, nothing read back", false, with(func(s *sysState) { s.ProcSwap = 0.5 }), "0.5 GB of caminus is in swap (not read back"},
		{"swap read back", true, with(func(s *sysState) { s.ProcSwap = 0.5; s.PageIn = 400 }), "paging: 0.5 GB of caminus is in swap"},
		{"swap-ins count with RAM free", true, with(func(s *sysState) { s.ProcSwap = 0; s.PageIn = 900; s.RamTot = 64; s.RamFree = 30 }), "paging: 4 MB/s"},
		{"swap-ins count without the process", true, with(func(s *sysState) { s.Pid, s.ProcSwap, s.PageIn = 0, -1, 900 }), "paging: 4 MB/s"},
		{"cpu-bound", true, with(func(s *sysState) { s.GPUUtil = 12; s.Cores = 7.5 }), "CPU-bound: 7.5 cores, GPU 12%"},
		{"busy-wait idle", false, with(func(s *sysState) { s.Cores = 1.9 }), "burning 1.9 cores"},
		{"no gpu counters", true, with(func(s *sysState) { s.GPUUtil = -1; s.Cores = 6 }), "GPU load unknown"},
		{"no counters", true, sysState{}, ""},
		{"low gpu, low cpu (waiting on the client)", true, with(func(s *sysState) { s.GPUUtil = 10; s.Cores = 0.2 }), "busy · no fallbacks"},
	}
	for _, c := range cases {
		got, _ := bottleneck(c.busy, c.ss, c.ss.VramTot)
		if c.want == "" && got != "" || !strings.Contains(got, c.want) {
			t.Errorf("%s: %q, want %q", c.name, got, c.want)
		}
	}
}

func TestFallbackRows(t *testing.T) {
	ss := sysState{OK: true, VramGB: 23.76, VramTot: 23.98, RamFree: 14.9, RamTot: 31, Pid: 18184, ProcVram: 21.48, Spill: 0.338,
		ProcSwap: -1, GPUUtil: 93, SwapGB: 0.983, Cores: 0.3}
	rows := fallbackRows(true, "", ss, 24, 16)
	j := ansi.Strip(strings.Join(rows, "\n"))
	for _, want := range []string{"RAM 0.34 GB", "swap 1.0 GB @ 0.0 MB/s", "GPU 93%", "CPU 0.3 of 16 cores", "caminus", "▸ GPU-bound"} {
		if !strings.Contains(j, want) {
			t.Errorf("missing %q in\n%s", want, j)
		}
	}
	ss.Pid = 0
	if j := ansi.Strip(strings.Join(fallbackRows(false, "", ss, 24, 16), "\n")); !strings.Contains(j, "no caminus process found") {
		t.Errorf("no process:\n%s", j)
	}
	if fallbackRows(true, "", sysState{}, 0, 4) != nil {
		t.Error("rows without counters")
	}
}
