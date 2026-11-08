// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

// caminus-panel: Caminus's terminal programs in one binary (the Python CLI runs it: `caminus panel`, `caminus top`).
//
//	caminus-panel [-advanced] [-caminus CMD]   the control panel (control.go)
//	caminus-panel top [llm-top's flags]        the monitor: llm-top, baked into Caminus on 11-07 (top_main.go and the
//	                                           files beside it; its own -config / -once / -web flags)
//	caminus-panel -version
package main

import (
	"fmt"
	"os"
)

func main() {
	args := os.Args[1:]
	if len(args) > 0 {
		switch args[0] {
		case "top", "monitor":
			topMain(args[1:])
			return
		case "-version", "--version", "version":
			fmt.Println("caminus-panel", version)
			return
		case "panel":
			args = args[1:]
		}
	}
	ctlMain(args)
}
