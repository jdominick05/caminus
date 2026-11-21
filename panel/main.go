// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

// caminus-panel: Caminus's terminal programs in one binary (the Python CLI runs it: `caminus panel`, `caminus top`).
//
//	caminus-panel [-caminus CMD]               the control panel (control.go, panelview.go, panellive.go)
//	caminus-panel top [llm-top's flags]        the monitor: llm-top, baked into Caminus on 11-07 (top_main.go and the
//	                                           files beside it; its own -config / -once / -web flags)
//	caminus-panel ember [--user U --bridge D --audit F --key K]   the agent environment's TUI (emberui.go; `caminus ember
//	                                           ui` execs it as `caminus ember`, the settings from CAMINUS_ENV_* / CAMINUS_EMBER_CLI)
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
		case "ember", "env": // (env: the old name)
			emberMain(args[1:])
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
