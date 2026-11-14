// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func near(a, b float64) bool { return math.Abs(a-b) < 0.01 }

// fakeLinux builds a /proc and /sys tree with a caminus process (pid 4242).
func fakeLinux(t *testing.T, amd bool) (proc, sys string) {
	d := t.TempDir()
	proc, sys = filepath.Join(d, "proc"), filepath.Join(d, "sys")
	write(t, filepath.Join(proc, "meminfo"), "MemTotal:       65536000 kB\nMemFree:         1000000 kB\nMemAvailable:   32768000 kB\n"+
		"SwapTotal:       8388608 kB\nSwapFree:        7340032 kB\n")
	write(t, filepath.Join(proc, "vmstat"), "nr_free_pages 1\npswpin 1200\npswpout 5000\n")
	write(t, filepath.Join(proc, "1", "comm"), "systemd\n")
	write(t, filepath.Join(proc, "1", "status"), "Name:\tsystemd\nVmRSS:\t   12000 kB\n")
	write(t, filepath.Join(proc, "4242", "comm"), "caminus\n")
	write(t, filepath.Join(proc, "4242", "status"), "Name:\tcaminus\nVmRSS:\t 9437184 kB\nRssAnon:\t 2080768 kB\nRssFile:\t 7340032 kB\nRssShmem:\t   16384 kB\nVmSwap:\t  524288 kB\n")
	write(t, filepath.Join(proc, "4242", "stat"), "4242 (caminus) S 1 4242 4242 0 -1 4194560 100 0 0 0 1500 250 0 0 20 0 9 0 100 0 0\n")
	write(t, filepath.Join(proc, "self", "comm"), "llm-top\n")
	if amd {
		// iGPU with 512 MB and a 24 GB card: the big one counts
		card := func(n, tot, used, gtt, busy string) {
			dev := filepath.Join(sys, "class", "drm", n, "device")
			write(t, filepath.Join(dev, "mem_info_vram_total"), tot)
			write(t, filepath.Join(dev, "mem_info_vram_used"), used)
			write(t, filepath.Join(dev, "mem_info_gtt_used"), gtt)
			write(t, filepath.Join(dev, "gpu_busy_percent"), busy)
		}
		card("card0", "536870912\n", "100000000\n", "50000000\n", "3\n")
		card("card1", "25753026560\n", "21474836480\n", "1073741824\n", "97\n")
	}
	return proc, sys
}

func TestProbeLinuxAMD(t *testing.T) {
	proc, sys := fakeLinux(t, true)
	ss := probeLinux(proc, sys)
	if !ss.OK || !near(ss.RamTot, 62.5) || !near(ss.RamFree, 31.25) || !near(ss.SrvWS, 2) || !near(ss.VramGB, 20) || !near(ss.VramTot, 23.98) {
		t.Errorf("%+v", ss)
	}
	if ss.Pid != 4242 || !near(ss.Spill, 1) || ss.GPUUtil != 97 || !near(ss.ProcSwap, 0.5) || !near(ss.SwapGB, 1) || !near(ss.CPUSec, 17.5) || ss.PageIn != 1200 {
		t.Errorf("fallbacks: %+v", ss)
	}
}

// The XTX and the XT holding Caminus' two-card split, and the Ryzen iGPU: the process's share of each card comes from
// its DRM fdinfo, one count per client however many fds repeat it.
func TestProbeLinuxSplitModel(t *testing.T) {
	proc, sys := fakeLinux(t, false)
	card := func(n, pdev, tot, used, gtt, busy string) {
		dev := filepath.Join(sys, "class", "drm", n, "device")
		write(t, filepath.Join(dev, "mem_info_vram_total"), tot)
		write(t, filepath.Join(dev, "mem_info_vram_used"), used)
		write(t, filepath.Join(dev, "mem_info_gtt_used"), gtt)
		write(t, filepath.Join(dev, "gpu_busy_percent"), busy)
		write(t, filepath.Join(dev, "uevent"), "DRIVER=amdgpu\nPCI_ID=1002:744C\nPCI_SLOT_NAME="+pdev+"\n")
	}
	card("card1", "0000:03:00.0", "25753026560\n", "20053274624\n", "900000000\n", "40\n")
	card("card2", "0000:06:00.0", "21458059264\n", "15406784512\n", "100000000\n", "35\n")
	card("card3", "0000:76:00.0", "536870912\n", "63123456\n", "300000000\n", "2\n")
	fdinfo := func(fd, pdev, id, vram, gtt string) {
		write(t, filepath.Join(proc, "4242", "fdinfo", fd), "pos:\t0\nflags:\t02100002\ndrm-driver:\tamdgpu\ndrm-client-id:\t"+id+
			"\ndrm-pdev:\t"+pdev+"\ndrm-memory-vram:\t"+vram+"\ndrm-memory-gtt: \t"+gtt+"\ndrm-memory-cpu: \t0 KiB\n")
	}
	write(t, filepath.Join(proc, "4242", "fdinfo", "0"), "pos:\t0\nflags:\t0100002\nmnt_id:\t25\n")
	fdinfo("5", "0000:03:00.0", "415", "18558284 KiB", "26708 KiB")
	fdinfo("6", "0000:03:00.0", "415", "18558284 KiB", "26708 KiB") // a dup of fd 5
	fdinfo("7", "0000:06:00.0", "416", "14939528 KiB", "524288 KiB")
	fdinfo("8", "0000:76:00.0", "417", "88 KiB", "26720 KiB")
	ss := probeLinux(proc, sys)
	if ss.modelCards() != 2 || len(ss.GPUs) != 3 || ss.GPUs[0].Name != "card1" || !ss.GPUs[1].Model || ss.GPUs[2].Model {
		t.Fatalf("cards: %+v", ss.GPUs)
	}
	if ss.Pid != 4242 || !near(ss.VramGB, 33.02) || !near(ss.VramTot, 43.97) || !near(ss.ProcVram, 31.95) || ss.GPUUtil != 75 {
		t.Errorf("%+v", ss)
	}
	if !near(ss.Spill, 0.52) || !near(ss.GPUs[1].Shared, 0.5) || !near(ss.GPUs[0].Proc, 17.7) {
		t.Errorf("spill %.2f, cards %+v", ss.Spill, ss.GPUs)
	}
	// no server: the desktop's GTT is not spill
	if err := os.RemoveAll(filepath.Join(proc, "4242")); err != nil {
		t.Fatal(err)
	}
	if ss := probeLinux(proc, sys); ss.Pid != 0 || ss.Spill != -1 || !near(ss.VramTot, 23.98) || ss.GPUUtil != 40 {
		t.Errorf("no process: %+v", ss)
	}
}

func TestProbeLinuxNoGPU(t *testing.T) {
	proc, sys := fakeLinux(t, false)
	if err := os.RemoveAll(filepath.Join(proc, "4242")); err != nil {
		t.Fatal(err)
	}
	ss := probeLinux(proc, sys)
	if !ss.OK || ss.VramTot != 0 || ss.VramGB != 0 || ss.SrvWS != 0 || ss.Pid != 0 || ss.GPUUtil != -1 {
		t.Errorf("%+v", ss)
	}
	if probeLinux(filepath.Join(t.TempDir(), "none"), sys).OK {
		t.Error("OK without /proc/meminfo")
	}
}

// Caminus names its process "caminus" (prctl): found by comm, the most resident one when two run (a brain / hands
// pair), never another server.
func TestFindProc(t *testing.T) {
	proc, _ := fakeLinux(t, false) // caminus is 4242, 9 GB resident
	write(t, filepath.Join(proc, "77", "comm"), "llama-server\n")
	write(t, filepath.Join(proc, "77", "status"), "VmRSS:\t 99000000 kB\n")
	write(t, filepath.Join(proc, "3100", "comm"), "caminus\n")
	write(t, filepath.Join(proc, "3100", "status"), "VmRSS:\t 400000 kB\n")
	if pid := findProc(proc); pid != 4242 {
		t.Errorf("two caminus processes: %d, want the bigger", pid)
	}
	if err := os.RemoveAll(filepath.Join(proc, "4242")); err != nil {
		t.Fatal(err)
	}
	if pid := findProc(proc); pid != 3100 {
		t.Errorf("caminus alone: %d", pid)
	}
	if err := os.RemoveAll(filepath.Join(proc, "3100")); err != nil {
		t.Fatal(err)
	}
	if pid := findProc(proc); pid != 0 {
		t.Errorf("no caminus: %d", pid)
	}
}

func TestStatCPU(t *testing.T) {
	if got := statCPU("12 (my (odd) name) R 1 1 1 0 -1 0 0 0 0 0 250 50 0 0"); !near(got, 3) {
		t.Errorf("%v", got)
	}
	if statCPU("garbage") != 0 {
		t.Error("garbage")
	}
}

func TestRates(t *testing.T) {
	prev := sysState{OK: true, Pid: 9, CPUSec: 100}
	prev = rates(prev, sysState{}, 0)
	prev.pageInRaw = 1000
	ss := rates(sysState{OK: true, Pid: 9, CPUSec: 110, PageIn: 6000}, prev, 5)
	if !near(ss.Cores, 2) || !near(ss.PageIn, 1000) || ss.pageInRaw != 6000 {
		t.Errorf("%+v", ss)
	}
	if ss := rates(sysState{OK: true, Pid: 10, CPUSec: 5}, prev, 5); ss.Cores != -1 {
		t.Error("cores across a pid change")
	}
	if ss := rates(sysState{OK: true, Pid: 9, CPUSec: 120, PageIn: 10}, ss, 5); ss.PageIn != 0 {
		t.Error("a counter that went down (a reboot between probes) is not a rate")
	}
}

func TestReadHold(t *testing.T) {
	d := t.TempDir()
	if readHold(filepath.Join(d, "none")) != "" || readHold("") != "" {
		t.Error("hold without file")
	}
	write(t, filepath.Join(d, "h"), "benchmark run\nmore")
	if got := readHold(filepath.Join(d, "h")); got != "benchmark run" {
		t.Errorf("%q", got)
	}
	write(t, filepath.Join(d, "e"), "\n")
	if got := readHold(filepath.Join(d, "e")); got != "GPU job" {
		t.Errorf("%q", got)
	}
}

func TestProbeLiveTmp(t *testing.T) {
	if os.Getenv("LLMTOP_LIVE") == "" {
		t.Skip("live probe: set LLMTOP_LIVE=1")
	}
	ss := probeLinux("/proc", "/sys")
	t.Logf("pid %d procVram %.2f spill %.2f gpu %.0f cards %d (model on %d)", ss.Pid, ss.ProcVram, ss.Spill, ss.GPUUtil, len(ss.GPUs), ss.modelCards())
}

// The probe reports where the server process listens, so a caminus serve on another port is found without config.
func TestListenAddrs(t *testing.T) {
	for in, want := range map[string]string{"0100007F:1F9B": "127.0.0.1:8091", "00000000:1F90": "0.0.0.0:8080",
		"00000000000000000000000001000000:1F9B": "[::1]:8091", "zz:1": "", "0100007F": ""} {
		if got := hexAddr(in); got != want {
			t.Errorf("hexAddr(%q) = %q, want %q", in, got, want)
		}
	}
	proc, _ := fakeLinux(t, false)
	fd := filepath.Join(proc, "4242", "fd")
	os.MkdirAll(fd, 0o755)
	if err := os.Symlink("socket:[55501]", filepath.Join(fd, "3")); err != nil {
		t.Skip("no symlinks here:", err)
	}
	write(t, filepath.Join(proc, "net", "tcp"), "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"+
		"   0: 0100007F:1F9C 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 55501 1 0\n"+
		"   1: 0100007F:2328 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 99999 1 0\n")
	if got := listenAddrs(proc, 4242); len(got) != 1 || got[0] != "127.0.0.1:8092" {
		t.Errorf("linux listen: %v", got)
	}
	if ss := probeLinux(proc, filepath.Join(t.TempDir(), "sys")); len(ss.Listen) != 1 || ss.Listen[0] != "127.0.0.1:8092" {
		t.Errorf("probe listen: %v", ss.Listen)
	}
}

// card is a gpuCard from byte counts, the way sysfs and fdinfo give them.
func card(name string, tot, used, proc, shared, util float64) gpuCard {
	return gpuCard{Name: name, Tot: tot / gib, Used: used / gib, Proc: proc / gib, Shared: shared / gib, Util: util}
}

// Two discrete cards (RX 7900 XTX 24 GB, RX 7900 XT 20 GB) and the iGPU. iGPU and idle card always carry a little of
// the process's memory.
func dualCards(xtx, xt [4]float64) []gpuCard {
	return []gpuCard{card("AMD Radeon(TM) Graphics", 536870912, 408150016, 149258240, 388784128, 0),
		card("AMD Radeon RX 7900 XTX", 25753026560, xtx[0], xtx[1], xtx[2], xtx[3]),
		card("AMD Radeon RX 7900 XT", 21458059264, xt[0], xt[1], xt[2], xt[3])}
}

// useCards: the numbers are those of the cards holding the model, and spill is judged per card.
func TestUseCards(t *testing.T) {
	with := func(cards []gpuCard) sysState {
		ss := sysState{OK: true, Pid: 25640, GPUUtil: 0, Spill: -1, ProcVram: -1}
		ss.useCards(cards)
		return ss
	}
	// all on the XTX, as with one GPU
	ss := with(dualCards([4]float64{23886606336, 22135398400, 325439488, 60}, [4]float64{6430720, 3211264, 316837888, 0}))
	if len(ss.GPUs) != 3 || ss.GPUs[0].Name != "AMD Radeon RX 7900 XTX" || !ss.GPUs[0].Model || ss.GPUs[1].Model || ss.GPUs[2].Model || ss.modelCards() != 1 {
		t.Fatalf("cards: %+v", ss.GPUs)
	}
	if !near(ss.VramGB, 22.25) || !near(ss.VramTot, 23.98) || !near(ss.ProcVram, 20.62) || !near(ss.Spill, 0.303) || ss.GPUUtil != 60 {
		t.Errorf("XTX only: %+v", ss)
	}
	// all on the XT: the XTX's numbers must not hide the XT filling up
	ss = with(dualCards([4]float64{1900000000, 0, 316837888, 0}, [4]float64{20401094656, 19327352832, 1073741824, 95}))
	if !near(ss.VramGB, 19) || !near(ss.VramTot, 19.98) || !near(ss.Spill, 1) || ss.GPUUtil != 95 {
		t.Errorf("XT only: %+v", ss)
	}
	if v, _ := bottleneck(true, ss, ss.VramTot); !strings.Contains(v, "VRAM full") {
		t.Errorf("XT only verdict %q", v)
	}
	// split: sums, load summed up to 100 %
	ss = with(dualCards([4]float64{15032385536, 13958643712, 343932928, 35}, [4]float64{12884901888, 11811160064, 322122547, 30}))
	if ss.modelCards() != 2 || !near(ss.VramGB, 26) || !near(ss.VramTot, 43.97) || !near(ss.ProcVram, 24) || !near(ss.Spill, 0.62) || ss.GPUUtil != 65 {
		t.Errorf("split: %+v", ss)
	}
	if v, _ := bottleneck(true, ss, ss.VramTot); !strings.Contains(v, "GPU-bound") {
		t.Errorf("split verdict %q", v)
	}
	if ss := with(dualCards([4]float64{20e9, 19e9, 0, 97}, [4]float64{17e9, 16e9, 0, 95})); ss.GPUUtil != 100 {
		t.Errorf("both cards busy: %v %%", ss.GPUUtil)
	}
	// split, the XT full and spilling while the XTX has room: 39 of 44 GB is not 90%, the card is
	ss = with(dualCards([4]float64{19327352832, 18253611008, 343932928, 50}, [4]float64{21152923648, 20401094656, 1610612736, 50}))
	if v, _ := bottleneck(true, ss, ss.VramTot); !strings.Contains(v, "VRAM full") {
		t.Errorf("one card overflowing: %q (%+v)", v, ss)
	}
	// split, both full but only a little GTT per card: 0.5 + 0.5 GB is over the one-card threshold
	ss = with(dualCards([4]float64{24696061952, 23622320128, 536870912, 50}, [4]float64{20937965568, 19864223744, 536870912, 50}))
	if v, _ := bottleneck(true, ss, ss.VramTot); strings.Contains(v, "VRAM full") || overflow(ss, ss.VramTot) {
		t.Errorf("split at rest flagged: %q", v)
	}
	// no server process: VRAM is the biggest card's, not the fullest one's
	ss = sysState{OK: true, GPUUtil: -1}
	ss.useCards([]gpuCard{card("AMD Radeon RX 7900 XT", 21458059264, 9000000000, 0, 0, -1), card("AMD Radeon RX 7900 XTX", 25753026560, 2000000000, 0, 0, -1)})
	if !near(ss.VramGB, 1.86) || !near(ss.VramTot, 23.98) || ss.modelCards() != 0 || ss.GPUUtil != -1 {
		t.Errorf("no process: %+v", ss)
	}
}

func TestCardName(t *testing.T) {
	for in, want := range map[string]string{"AMD Radeon RX 7900 XTX": "RX 7900 XTX", "AMD Radeon(TM) Graphics": "Radeon(TM) Graphics", "": "GPU"} {
		if got := cardName(in); got != want {
			t.Errorf("%q: %q, want %q", in, got, want)
		}
	}
}

// two Caminus servers (brain / hands): each monitor follows the process listening on its own port
func TestFindProcByPort(t *testing.T) {
	proc := t.TempDir()
	mk := func(pid int, rssKB int, port uint16, inode string) {
		d := filepath.Join(proc, strconv.Itoa(pid))
		_ = os.MkdirAll(filepath.Join(d, "fd"), 0o755)
		_ = os.WriteFile(filepath.Join(d, "comm"), []byte("caminus\n"), 0o644)
		_ = os.WriteFile(filepath.Join(d, "status"), []byte(fmt.Sprintf("VmRSS:\t%d kB\n", rssKB)), 0o644)
		_ = os.Symlink("socket:["+inode+"]", filepath.Join(d, "fd", "3"))
		_ = port
	}
	mk(100, 9000000, 8092, "111") // the bigger one (the brain, :8092)
	mk(200, 4000000, 8093, "222") // the hands, :8093
	tcp := "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n" +
		"   0: 0100007F:1F9C 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 111 1\n" +
		"   1: 0100007F:1F9D 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 222 1\n"
	_ = os.MkdirAll(filepath.Join(proc, "net"), 0o755)
	_ = os.WriteFile(filepath.Join(proc, "net", "tcp"), []byte(tcp), 0o644)
	defer func(p string) { procPort = p }(procPort)
	for _, c := range []struct {
		port string
		want int
	}{{"", 100}, {"8092", 100}, {"8093", 200}, {"9999", 100}} {
		procPort = c.port
		if got := findProc(proc); got != c.want {
			t.Errorf("port %q: pid %d, want %d", c.port, got, c.want)
		}
	}
}

// each monitor of a brain / hands pair lists only its own server's agent sessions
func TestMineOnly(t *testing.T) {
	ss := []*session{{Path: "a", model: "arex-2"}, {Path: "b", model: "qwen3.6"}, {Path: "c", model: "qwen3.6-nothink"}, {Path: "d"},
		{Path: "e", model: "k3-coder-xhigh"}, {Path: "f", model: "qwen3.60"}, {Path: "g", model: "other-qwen3.6"}}
	names := func(l []*session) (o []string) {
		for _, s := range l {
			o = append(o, s.Path)
		}
		return
	}
	for _, c := range []struct{ alias, want string }{{"qwen3.6", "b c d"}, {"arex-2", "a d"}, {"k3-coder", "d e"}, {"", "a b c d e f g"}} {
		if got := strings.Join(names(mineOnly(ss, c.alias)), " "); got != c.want {
			t.Errorf("alias %q: %q, want %q", c.alias, got, c.want)
		}
	}
	if len(ss) != 7 {
		t.Errorf("mineOnly changed its input")
	}
}
