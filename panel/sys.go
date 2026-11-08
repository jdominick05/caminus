// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

// Machine counters from /proc and sysfs: VRAM, RAM, and the Caminus process's fallbacks (VRAM spilled to system RAM
// as amdgpu GTT, swap, the CPU doing the work) for the bottleneck line.
//
// Every amdgpu card comes from sysfs (VRAM, GTT, busy %), the process's share of each from its DRM fdinfo. The
// process is the one named caminus (the server sets its comm with prctl), the most resident one if several run. The
// probe reads under a /proc and a /sys root, so tests run on fixture trees.
package main

import (
	"encoding/hex"
	"net"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

type sysState struct {
	OK              bool
	VramGB, VramTot float64 // VramTot 0 = unknown
	RamFree, RamTot float64
	Hold            string

	// the caminus process; Pid 0 = not found
	Pid      int
	SrvWS    float64 // resident memory without the page cache (anonymous + shmem), GB
	CPUSec   float64 // total CPU time, s
	ProcVram float64 // GB, -1 = unknown
	Spill    float64 // GB of GPU memory in system RAM (amdgpu GTT), -1 = unknown
	ProcSwap float64 // GB of the process in swap, -1 = unknown
	GPUUtil  float64 // %, -1 = unknown
	SwapGB   float64 // swap in use, GB
	PageIn   float64 // pages swapped in per second (rates turns the cumulative pswpin into it)

	Cores     float64  // CPU cores the process used since the last probe, -1 = unknown
	Listen    []string // addresses the process listens on ("127.0.0.1:8092"): finds a server on another port
	pageInRaw float64  // pswpin at this probe

	GPUs []gpuCard // every amdgpu card, biggest first; nil = no per-card counters
}

// gpuCard is one GPU. Caminus' two-card engines hold a part of the model on each.
type gpuCard struct {
	Name         string
	Used, Tot    float64 // VRAM, GB
	Proc, Shared float64 // the caminus process on this card: VRAM and GTT, GB
	Util         float64 // the card's busy %, -1 = unknown
	Model        bool    // the process holds at least modelCardGB here
}

// modelCardGB of the process's VRAM make a card one the model runs on; the runtime keeps a few MB (or a few
// hundred, on the iGPU) on every other GPU.
const modelCardGB = 1

const gib = 1 << 30

// procName is the server's comm (caminus serve names its process before any thread starts).
const procName = "caminus"

// useCards replaces the single-card numbers (the biggest card's) with those of the cards the process holds the model
// on: VRAM, the process's VRAM and spill summed, GPU load summed up to 100 %. Without such a card (no server, or no
// model loaded yet) VRAM is the biggest card's.
func (ss *sysState) useCards(cards []gpuCard) {
	if len(cards) == 0 {
		return
	}
	sort.SliceStable(cards, func(i, j int) bool { return cards[i].Tot > cards[j].Tot })
	var used, tot, proc, shared, util float64
	n := 0
	for i := range cards {
		c := &cards[i]
		c.Model = ss.Pid > 0 && c.Proc >= modelCardGB
		if c.Model {
			n++
			used, tot, proc, shared = used+c.Used, tot+c.Tot, proc+c.Proc, shared+c.Shared
			util += max(0, c.Util)
		}
	}
	ss.GPUs = cards
	if n == 0 {
		ss.VramGB, ss.VramTot = cards[0].Used, cards[0].Tot
		return
	}
	ss.VramGB, ss.VramTot, ss.ProcVram, ss.Spill = used, tot, proc, shared
	if ss.GPUUtil >= 0 {
		ss.GPUUtil = min(100, util)
	}
}

// modelCards counts the cards the model runs on.
func (ss sysState) modelCards() int {
	n := 0
	for _, c := range ss.GPUs {
		if c.Model {
			n++
		}
	}
	return n
}

func probeLinux(procRoot, sysRoot string) sysState {
	b, err := os.ReadFile(filepath.Join(procRoot, "meminfo"))
	if err != nil {
		return sysState{}
	}
	mi := parseKV(string(b))
	if mi["MemTotal"] <= 0 {
		return sysState{}
	}
	ss := sysState{OK: true, RamTot: mi["MemTotal"] / (1 << 20), RamFree: mi["MemAvailable"] / (1 << 20),
		SwapGB: (mi["SwapTotal"] - mi["SwapFree"]) / (1 << 20), ProcVram: -1, Spill: -1, ProcSwap: -1, GPUUtil: -1, Cores: -1}
	if vs, err := os.ReadFile(filepath.Join(procRoot, "vmstat")); err == nil {
		for _, l := range strings.Split(string(vs), "\n") {
			if f := strings.Fields(l); len(f) == 2 && f[0] == "pswpin" {
				ss.PageIn, _ = strconv.ParseFloat(f[1], 64)
			}
		}
	}
	if pid := findProc(procRoot); pid > 0 {
		d := filepath.Join(procRoot, strconv.Itoa(pid))
		ss.Pid, ss.Listen = pid, listenAddrs(procRoot, pid)
		if st, err := os.ReadFile(filepath.Join(d, "status")); err == nil {
			kv := parseKV(string(st))
			// VmRSS counts the mmap'd model file, which is page cache (reclaimable, "free"): only the process's own pages
			ss.SrvWS = kv["VmRSS"] / (1 << 20)
			if a, ok := kv["RssAnon"]; ok {
				ss.SrvWS = (a + kv["RssShmem"]) / (1 << 20)
			}
			if sw, ok := kv["VmSwap"]; ok {
				ss.ProcSwap = sw / (1 << 20)
			}
		}
		if st, err := os.ReadFile(filepath.Join(d, "stat")); err == nil {
			ss.CPUSec = statCPU(string(st))
		}
	}
	// every card, and the process's share of each from its DRM fdinfo. Without that share (another user's process)
	// the biggest card counts, its card-wide GTT standing in for spill.
	if amd := amdGPUs(sysRoot); len(amd) > 0 {
		var mem map[string][2]float64
		if ss.Pid > 0 {
			mem = drmMem(filepath.Join(procRoot, strconv.Itoa(ss.Pid), "fdinfo"))
		}
		cards := make([]gpuCard, len(amd))
		var proc, shared float64
		for i, g := range amd {
			m := mem[g.pdev]
			cards[i] = gpuCard{Name: g.name, Used: g.used / gib, Tot: g.tot / gib, Proc: m[0] / gib, Shared: m[1] / gib, Util: g.busy}
			proc, shared = proc+m[0]/gib, shared+m[1]/gib
		}
		ss.GPUUtil = amd[0].busy
		switch {
		case mem != nil:
			ss.ProcVram, ss.Spill = proc, shared
		case ss.Pid > 0 && amd[0].gtt >= 0:
			ss.Spill = amd[0].gtt / gib
		}
		ss.useCards(cards)
	}
	return ss
}

// listenAddrs are the TCP sockets pid listens on: /proc/net/tcp{,6} rows in state 0A (LISTEN) whose inode is one of
// the process's fds.
func listenAddrs(procRoot string, pid int) []string {
	inodes := map[string]bool{}
	fds, _ := filepath.Glob(filepath.Join(procRoot, strconv.Itoa(pid), "fd", "*"))
	for _, fd := range fds {
		if l, err := os.Readlink(fd); err == nil && strings.HasPrefix(l, "socket:[") {
			inodes[strings.TrimSuffix(strings.TrimPrefix(l, "socket:["), "]")] = true
		}
	}
	var out []string
	for _, f := range []string{"tcp", "tcp6"} {
		b, err := os.ReadFile(filepath.Join(procRoot, "net", f))
		if err != nil {
			continue
		}
		for _, l := range strings.Split(string(b), "\n")[1:] {
			x := strings.Fields(l)
			if len(x) < 10 || x[3] != "0A" || !inodes[x[9]] {
				continue
			}
			if a := hexAddr(x[1]); a != "" {
				out = append(out, a)
			}
		}
	}
	return out
}

// hexAddr turns /proc/net/tcp's "0100007F:1F9C" (little-endian IPv4, or 32 hex digits of IPv6) into "127.0.0.1:8092".
func hexAddr(s string) string {
	h, p, ok := strings.Cut(s, ":")
	port, err := strconv.ParseUint(p, 16, 16)
	if !ok || err != nil {
		return ""
	}
	b, err := hex.DecodeString(h)
	if err != nil || len(b) != 4 && len(b) != 16 {
		return ""
	}
	for i := 0; i+4 <= len(b); i += 4 { // each 32-bit word is little-endian
		b[i], b[i+1], b[i+2], b[i+3] = b[i+3], b[i+2], b[i+1], b[i]
	}
	return net.JoinHostPort(net.IP(b).String(), strconv.Itoa(int(port)))
}

// parseKV reads "Key:   123 kB" lines (/proc/meminfo, /proc/<pid>/status) into kB values.
func parseKV(s string) map[string]float64 {
	m := map[string]float64{}
	for _, l := range strings.Split(s, "\n") {
		k, v, ok := strings.Cut(l, ":")
		if !ok {
			continue
		}
		if f := strings.Fields(v); len(f) > 0 {
			if x, err := strconv.ParseFloat(f[0], 64); err == nil {
				m[strings.TrimSpace(k)] = x
			}
		}
	}
	return m
}

// findProc returns the pid of the process named caminus (its comm), the most resident one if several run; 0 = none.
// procPort: the configured server's port (top_main.go); "" = none configured
var procPort string

// findProc picks the caminus process: the one listening on procPort when one does (two servers, one card each, each
// with its own monitor), else the largest by resident memory
func findProc(procRoot string) int {
	dirs, _ := filepath.Glob(filepath.Join(procRoot, "[0-9]*"))
	sort.Strings(dirs)
	best, bestRSS := 0, -1.0
	for _, d := range dirs {
		c, err := os.ReadFile(filepath.Join(d, "comm"))
		if err != nil || strings.TrimSpace(string(c)) != procName {
			continue
		}
		if procPort != "" {
			pid, _ := strconv.Atoi(filepath.Base(d))
			for _, a := range listenAddrs(procRoot, pid) {
				if strings.HasSuffix(a, ":"+procPort) {
					return pid
				}
			}
		}
		st, _ := os.ReadFile(filepath.Join(d, "status"))
		if rss := parseKV(string(st))["VmRSS"]; rss > bestRSS {
			best, _ = strconv.Atoi(filepath.Base(d))
			bestRSS = rss
		}
	}
	return best
}

// statCPU returns utime + stime in seconds from /proc/<pid>/stat (USER_HZ is 100 on Linux).
func statCPU(s string) float64 {
	i := strings.LastIndexByte(s, ')') // comm may contain spaces
	if i < 0 {
		return 0
	}
	f := strings.Fields(s[i+1:])
	if len(f) < 13 { // fields 3.. after comm; utime and stime are 14 and 15
		return 0
	}
	u, _ := strconv.ParseFloat(f[11], 64)
	st, _ := strconv.ParseFloat(f[12], 64)
	return (u + st) / 100
}

type amdCard struct {
	name, pdev           string // pdev: PCI address, as DRM fdinfo names it
	used, tot, gtt, busy float64
}

// amdGPUs lists the amdgpu cards from sysfs (bytes, busy in %), biggest first.
func amdGPUs(sysRoot string) []amdCard {
	var out []amdCard
	files, _ := filepath.Glob(filepath.Join(sysRoot, "class", "drm", "card*", "device", "mem_info_vram_total"))
	for _, f := range files {
		d := filepath.Dir(f)
		t, u := readNum(f), readNum(filepath.Join(d, "mem_info_vram_used"))
		if t <= 0 || u < 0 {
			continue
		}
		g := amdCard{name: filepath.Base(filepath.Dir(d)), used: u, tot: t, gtt: readNum(filepath.Join(d, "mem_info_gtt_used")), busy: readNum(filepath.Join(d, "gpu_busy_percent"))}
		if b, err := os.ReadFile(filepath.Join(d, "product_name")); err == nil && strings.TrimSpace(string(b)) != "" {
			g.name = strings.TrimSpace(string(b))
		}
		if b, err := os.ReadFile(filepath.Join(d, "uevent")); err == nil {
			for _, l := range strings.Split(string(b), "\n") {
				if v, ok := strings.CutPrefix(strings.TrimSpace(l), "PCI_SLOT_NAME="); ok {
					g.pdev = v
				}
			}
		}
		out = append(out, g)
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].tot > out[j].tot })
	return out
}

// drmMem sums a process's GPU memory per PCI address from its DRM fdinfo, in bytes: [0] VRAM, [1] GTT. Every fd of
// one DRM client repeats that client's numbers, so each drm-client-id counts once. nil = fdinfo unreadable (another
// user's process).
func drmMem(dir string) map[string][2]float64 {
	fds, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	m, seen := map[string][2]float64{}, map[string]bool{}
	for _, fd := range fds {
		b, err := os.ReadFile(filepath.Join(dir, fd.Name()))
		if err != nil {
			continue
		}
		kv := map[string]string{}
		for _, l := range strings.Split(string(b), "\n") {
			if k, v, ok := strings.Cut(l, ":"); ok && strings.HasPrefix(k, "drm-") {
				kv[k] = strings.TrimSpace(v)
			}
		}
		pdev := kv["drm-pdev"]
		id := pdev + " " + kv["drm-client-id"]
		if pdev == "" || seen[id] {
			continue
		}
		seen[id] = true
		x := m[pdev]
		x[0] += drmBytes(kv, "vram")
		x[1] += drmBytes(kv, "gtt")
		m[pdev] = x
	}
	return m
}

// drmBytes reads drm-memory-<region> (amdgpu), else drm-resident-<region>: "123 KiB", MiB, GiB or plain bytes.
func drmBytes(kv map[string]string, region string) float64 {
	v, ok := kv["drm-memory-"+region]
	if !ok {
		v = kv["drm-resident-"+region]
	}
	f := strings.Fields(v)
	if len(f) == 0 {
		return 0
	}
	x, err := strconv.ParseFloat(f[0], 64)
	if err != nil {
		return 0
	}
	if u, ok := map[string]float64{"KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30}[f[len(f)-1]]; ok {
		x *= u
	}
	return x
}

func readNum(p string) float64 {
	b, err := os.ReadFile(p)
	if err != nil {
		return -1
	}
	x, err := strconv.ParseFloat(strings.TrimSpace(string(b)), 64)
	if err != nil {
		return -1
	}
	return x
}

// ---------- poller ----------

func (st *state) pollSys(hold string) {
	var prev sysState
	var prevT time.Time
	for {
		ss := probeLinux("/proc", "/sys")
		now := time.Now()
		ss = rates(ss, prev, now.Sub(prevT).Seconds())
		prev, prevT = ss, now
		ss.Hold = readHold(hold)
		st.mu.Lock()
		st.sys, st.sysTried = ss, true
		st.mu.Unlock()
		time.Sleep(5 * time.Second)
	}
}

// rates turns counters into rates: CPU seconds into cores, the cumulative pswpin into pages per second. prev keeps
// the raw counters in its fields, so the returned state carries them for the next call.
func rates(ss, prev sysState, dt float64) sysState {
	ss.Cores = -1
	if ss.Pid != 0 && ss.Pid == prev.Pid && dt > 0 && ss.CPUSec >= prev.CPUSec {
		ss.Cores = (ss.CPUSec - prev.CPUSec) / dt
	}
	raw := ss.PageIn
	ss.PageIn = 0
	if prev.OK && dt > 0 && raw >= prev.pageInRaw {
		ss.PageIn = (raw - prev.pageInRaw) / dt
	}
	ss.pageInRaw = raw
	return ss
}

// readHold returns the hold file's first line ("GPU job" if empty), or "" when there is no hold.
func readHold(p string) string {
	if p == "" {
		return ""
	}
	b, err := os.ReadFile(p)
	if err != nil {
		return ""
	}
	if h := strings.TrimSpace(strings.SplitN(string(b), "\n", 2)[0]); h != "" {
		return h
	}
	return "GPU job"
}
