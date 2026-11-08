// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

// Bottleneck: one line that says what limits the server right now, from the caminus process's fallbacks.
//
// In order: paging (swap read back from disk), VRAM spilled into system RAM (GTT: GPU memory the driver keeps in
// system RAM, read at PCIe speed), the CPU doing the work while the GPU waits, and otherwise GPU-bound, which is
// what a healthy server looks like. Idle, it flags a server that burns CPU without work (busy-wait polling).
package main

import (
	"fmt"
	"image/color"
	"strings"
)

const (
	spillWarnGB  = 0.75 // the process's GTT from which it counts as spill
	vramFullFrac = 0.9  // spill only counts while VRAM is this full
	dgpuMinGB    = 2    // below this the GPU is integrated: system RAM is its normal memory
	swapReadBack = 50   // pages/s of swap read back that make swapped-out memory a problem
	pageInWarn   = 500  // pages/s read back from disk (2 MB/s)
	cpuBoundCore = 1.5  // cores while the GPU is below gpuBusyPct
	gpuBusyPct   = 50
	idleBurnCore = 1.0
)

func bottleneck(busy bool, ss sysState, vramTot float64) (string, color.Color) {
	if !ss.OK {
		return "", cMuted
	}
	switch {
	case ss.PageIn >= pageInWarn:
		return fmt.Sprintf("paging: %.0f MB/s read back from disk (RAM short)", ss.PageIn*4096/1e6), cRed
	case ss.ProcSwap >= 0.1 && ss.PageIn >= swapReadBack:
		return fmt.Sprintf("paging: %.1f GB of %s is in swap", ss.ProcSwap, procName), cRed
	case ss.ProcSwap >= 0.1 && !busy:
		return fmt.Sprintf("%.1f GB of %s is in swap (not read back now)", ss.ProcSwap, procName), cOrange
	case overflow(ss, vramTot):
		c := cOrange
		if busy {
			c = cRed
		}
		return fmt.Sprintf("VRAM full: %.1f GB spilled to system RAM (slow)", ss.Spill), c
	case !busy && ss.Cores >= idleBurnCore:
		return fmt.Sprintf("idle but burning %.1f cores (busy-wait?)", ss.Cores), cYellow
	case !busy:
		if vramTot > 0 && ss.VramGB > 0 {
			return fmt.Sprintf("idle · no fallbacks · %.1f GB VRAM headroom", max(0, vramTot-ss.VramGB)), cMuted
		}
		return "idle · no fallbacks", cMuted
	case ss.GPUUtil >= 0 && ss.GPUUtil < gpuBusyPct && ss.Cores >= cpuBoundCore:
		return fmt.Sprintf("CPU-bound: %.1f cores, GPU %.0f%%", ss.Cores, ss.GPUUtil), cOrange
	case ss.GPUUtil >= gpuBusyPct:
		return fmt.Sprintf("GPU-bound (normal): GPU %.0f%%, no fallbacks", ss.GPUUtil), cGreen
	case ss.Cores >= cpuBoundCore && ss.GPUUtil < 0:
		return fmt.Sprintf("busy on %.1f cores, GPU load unknown", ss.Cores), cYellow
	}
	return "busy · no fallbacks", cGreen
}

// overflow says whether the process's GTT means the model did not fit. With the model on several cards each is
// judged alone (one card can overflow while the other has room, and what the runtime keeps in GTT is per card).
func overflow(ss sysState, vramTot float64) bool {
	if ss.modelCards() < 2 {
		return ss.Spill >= spillWarnGB && vramFull(ss, vramTot)
	}
	for _, c := range ss.GPUs {
		if c.Model && c.Shared >= spillWarnGB && c.Used >= vramFullFrac*c.Tot {
			return true
		}
	}
	return false
}

// vramFull says whether spill means "did not fit": VRAM is (nearly) full. Integrated GPUs use system RAM by design,
// so they never count as full.
func vramFull(ss sysState, vramTot float64) bool {
	if vramTot <= 0 || ss.VramGB <= 0 {
		return true // unknown: trust the spill
	}
	return vramTot >= dgpuMinGB && ss.VramGB >= vramFullFrac*vramTot
}

// fallbackRows are the THROUGHPUT lines for spill, paging and load, plus the verdict. off (serverState.offline)
// replaces the verdict: there is no model to judge.
func fallbackRows(busy bool, off string, ss sysState, vramTot float64, ncpu int) []string {
	if !ss.OK {
		return nil
	}
	gb := func(x float64, warn bool) string {
		if x < 0 {
			return fg(cMuted).Render("n/a")
		}
		s := fmt.Sprintf("%.2f GB", x)
		if warn {
			return fg(cOrange).Render(s)
		}
		return s
	}
	spill := "RAM " + gb(ss.Spill, overflow(ss, vramTot))
	rate := fmt.Sprintf("@ %.1f MB/s", ss.PageIn*4096/1e6) // read back from disk
	if ss.PageIn >= pageInWarn {
		rate = fg(cRed).Render(rate)
	}
	page := fmt.Sprintf("swap %.1f GB %s", ss.SwapGB, rate)
	if ss.ProcSwap >= 0 { // what of the process is in swap
		page = "swap " + gb(ss.ProcSwap, ss.ProcSwap >= 0.1) + " " + rate
	}
	rows := []string{fmt.Sprintf("%-6s%s · %s", "spill", spill, page)}
	if ss.Pid == 0 {
		rows = append(rows, fmt.Sprintf("%-6s%s", "load", fg(cMuted).Render("no "+procName+" process found")))
	} else {
		gpu := fg(cMuted).Render("n/a")
		if ss.GPUUtil >= 0 {
			gpu = fmt.Sprintf("%.0f%%", ss.GPUUtil)
		}
		cpu := "…"
		if ss.Cores >= 0 {
			cpu = fmt.Sprintf("%.1f", ss.Cores)
		}
		rows = append(rows, fmt.Sprintf("%-6sGPU %s · CPU %s of %d cores · %s", "load", gpu, cpu, ncpu, fg(cMuted).Render(procName)))
	}
	if off != "" {
		v := "▸ " + off
		if vramTot > 0 && ss.VramGB > 0 {
			v += fmt.Sprintf(" · %.1f GB VRAM free", max(0, vramTot-ss.VramGB))
		}
		return append(rows, fg(cMuted).Render(v))
	}
	if v, c := bottleneck(busy, ss, vramTot); v != "" {
		rows = append(rows, fg(c).Render("▸ "+strings.TrimSpace(v)))
	}
	return rows
}
