// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestWithdrawnRequestRejectsCachedApprovalAndAnswer(t *testing.T) {
	bridge := t.TempDir()
	writeReq(t, bridge, "a", "first", 1)
	reqs, _ := loadRequests(bridge)
	r := reqs[0]
	if err := writeBridgeJSON(filepath.Join(bridge, "res", "a.json"), []byte(`{"id":"a","decision":"withdrawn"}`)); err != nil {
		t.Fatal(err)
	}
	if _, err := claimRoot(bridge, "human", r, "approved", r.Cmd); err == nil {
		t.Fatal("withdrawn approval accepted")
	}
	if err := writeAnswer(bridge, "human", r, "denied", "", nil, "", "", ""); err == nil {
		t.Fatal("withdrawal overwritten")
	}
	reqs, _ = loadRequests(bridge)
	if len(reqs) != 0 {
		t.Fatal("withdrawn request remains pending")
	}
}

func TestEditedRequestIdentitySurvivesQueueRefresh(t *testing.T) {
	bridge := t.TempDir()
	writeReq(t, bridge, "a", "first", 1)
	writeReq(t, bridge, "b", "second", 2)
	m := emberModel{bridge: bridge, seen: map[string]bool{}, mode: "normal", auditPath: filepath.Join(t.TempDir(), "audit")}
	m.refreshLocal()
	mm, _ := m.onKey("e")
	m = mm.(emberModel)
	m.edit = "human edited first command"
	writeBridgeJSON(filepath.Join(bridge, "res", "a.json"), []byte(`{"id":"a","decision":"withdrawn"}`))
	m.refreshLocal()
	if m.reqs[0].ID != "b" {
		t.Fatal("test did not move selected queue row")
	}
	mm, command := m.submitInput()
	m = mm.(emberModel)
	if command != nil {
		t.Fatal("stale edited command would execute")
	}
	if _, err := os.Stat(filepath.Join(bridge, "res", "b.json")); !os.IsNotExist(err) {
		t.Fatal("edit retargeted to second request")
	}
	if !strings.Contains(m.msg, "withdrawn") {
		t.Fatal(m.msg)
	}
}

func TestChangedRequestRequiresAnotherHumanReview(t *testing.T) {
	bridge := t.TempDir()
	writeReq(t, bridge, "a", "first", 1)
	reqs, _ := loadRequests(bridge)
	writeReq(t, bridge, "a", "changed", 1)
	if _, err := claimRoot(bridge, "human", reqs[0], "edited", "first"); err == nil {
		t.Fatal("changed snapshot approved")
	}
}

func TestRootClaimCompletesOnlyOnceWithMatchingToken(t *testing.T) {
	bridge := t.TempDir()
	writeReq(t, bridge, "a", "first", 1)
	reqs, _ := loadRequests(bridge)
	r := reqs[0]
	token, err := claimRoot(bridge, "human", r, "approved", r.Cmd)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := claimRoot(bridge, "human", r, "approved", r.Cmd); err == nil {
		t.Fatal("duplicate execution claimed")
	}
	if err := finishRoot(bridge, "a", "wrong", 0, "bad"); err == nil {
		t.Fatal("wrong claim token accepted")
	}
	if err := finishRoot(bridge, "a", token, 0, "verified"); err != nil {
		t.Fatal(err)
	}
	raw, _ := os.ReadFile(filepath.Join(bridge, "res", "a.json"))
	var result map[string]any
	json.Unmarshal(raw, &result)
	if result["decision"] != "approved" || result["out"] != "verified" {
		t.Fatal(result)
	}
	if err := finishRoot(bridge, "a", token, 0, "again"); err == nil {
		t.Fatal("completed claim overwritten")
	}
}

func TestBridgeLockIsSharedWithPython(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("Python needed for cross-language lock check")
	}
	bridge := t.TempDir()
	writeReq(t, bridge, "a", "first", 1)
	err = withBridgeLock(bridge, "a", func() error {
		script := "import fcntl,sys\nf=open(sys.argv[1],'r+')\ntry:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError:sys.exit(0)\nsys.exit(1)\n"
		return exec.Command(python, "-I", "-c", script, filepath.Join(bridge, "req", "a.lock")).Run()
	})
	if err != nil {
		t.Fatalf("Python did not observe Go lock: %v", err)
	}
}

func TestBridgeLockGivesUpQuicklyInsteadOfFreezingThePanel(t *testing.T) {
	bridge := t.TempDir()
	dir := filepath.Join(bridge, "req")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	id := "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6"
	holder, err := os.OpenFile(filepath.Join(dir, id+".lock"), os.O_RDWR|os.O_CREATE, 0o660)
	if err != nil {
		t.Fatal(err)
	}
	defer holder.Close()
	if err := syscall.Flock(int(holder.Fd()), syscall.LOCK_EX); err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	err = withBridgeLock(bridge, id, func() error {
		t.Fatal("the body must not run while another writer holds the lock")
		return nil
	})
	elapsed := time.Since(start)
	if err == nil || !strings.Contains(err.Error(), "busy") {
		t.Fatalf("a held lock must report busy, got %v", err)
	}
	if elapsed > 2*time.Second {
		t.Fatalf("waited %v inside the update loop; the panel would freeze", elapsed)
	}
}
