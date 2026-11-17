#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 jdominick05
# A stand-in caminus CLI for the panel's tests and lab runs: it records every call and simulates one running server
# (no real server, no GPU). Run the panel with -caminus pointing at this file and STUB_STATE set to a scratch folder:
#   running     the managed server's name (caminus stop NAME takes it away; caminus start writes it)
#   ext_pid     an outside server's pid (the test's own sleep): listed as external while that process lives
#   hold        how many ps reads after a stop still show the card's memory in use (the wait for VRAM)
#   fail_stop   caminus stop fails the way manage.py's stop_one does when the guard outlives its timeout
#   fail_start  caminus start fails the way a startup failure does
#   calls.log   every command line, one per line
S=${STUB_STATE:?}
echo "$*" >> "$S/calls.log"
running=$(cat "$S/running" 2>/dev/null)
servers() {
  sep=""
  if [ -n "$running" ]; then
    printf '{"name":"%s","model":"%s","state":"ready","url":"http://127.0.0.1:59991","port":59991,"cards":["XTX"],"pid":0,"max_ctx":40960,"uptime_s":600,"external":false}' "$running" "$running"
    sep=","
  fi
  if [ -f "$S/ext_pid" ]; then
    p=$(cat "$S/ext_pid")
    if kill -0 "$p" 2>/dev/null; then
      printf '%s{"name":"observed-59992","model":"alpha","state":"ready","url":"http://127.0.0.1:59992","port":59992,"cards":[],"pid":%s,"external":true}' "$sep" "$p"
    fi
  fi
}
ext=""
if [ -f "$S/ext_pid" ] && kill -0 "$(cat "$S/ext_pid")" 2>/dev/null; then ext=1; fi
used=1500
if [ -n "$running" ] || [ -n "$ext" ]; then used=21000; fi
case "$1" in
models)
  echo '{"models":[{"name":"alpha","title":"Alpha 30B (stub)","cards":1,"size_gb":19,"present":true,"path":"/stub/alpha.gguf","serve_modes":[{"cards":1,"drafts":["off"]}]},{"name":"beta","title":"Beta 27B (stub)","cards":1,"size_gb":16,"present":true,"path":"/stub/beta.gguf","serve_modes":[{"cards":1,"drafts":["off"]}]}]}' ;;
ps)
  if [ -z "$running" ] && [ -f "$S/hold" ] && [ "$(cat "$S/hold")" -gt 0 ]; then
    used=19800
    echo $(( $(cat "$S/hold") - 1 )) > "$S/hold"
  fi
  printf '{"servers":[%s],"cards":[{"index":0,"name":"XTX","pci":"0000:03:00.0","used_mib":%s,"line_mib":23800,"total_mib":24560,"link":"x16","link_ok":true,"status":"supported"}],"problems":[]}\n' "$(servers)" "$used" ;;
qwen-code) echo '{"installed":false,"npm":true}' ;;
ember) echo '{"ok":false}' ;;
serve) echo '[]' ;;
fit) echo '{"ok":true,"max_ctx":{"f16":40960,"q8":null},"kv_formats":["f16"]}' ;;
stop)
  sleep "${STUB_DELAY:-1}"
  if [ -f "$S/fail_stop" ]; then
    echo '{"ok":false,"note":"","error":"'"$2"' still running after 170 s (guard pid 4242): caminus logs '"$2"'"}'
    exit 1
  fi
  if [ "$2" != "$running" ]; then
    echo '{"ok":false,"note":"","error":"no server named '"'$2'"' (caminus ps)"}'
    exit 1
  fi
  rm -f "$S/running"
  echo "${STUB_HOLD:-2}" > "$S/hold"
  echo '{"ok":true,"note":"stopped '"$2"'","error":""}' ;;
start)
  sleep "${STUB_DELAY:-1}"
  if [ -n "$running" ]; then
    echo '{"ok":false,"note":"","error":"every card is held by a Caminus server (caminus ps; caminus stop NAME)"}'
    exit 1
  fi
  if [ -n "$ext" ] || { [ -f "$S/hold" ] && [ "$(cat "$S/hold")" -gt 0 ]; }; then
    echo '{"ok":false,"note":"","error":"VRAM in use on XTX 19,800 MiB by a process Caminus did not start: stop it first"}'
    exit 1
  fi
  if [ -f "$S/fail_start" ]; then
    echo '{"ok":false,"note":"","error":"'"$2"' exited during startup: out of memory"}'
    exit 1
  fi
  echo "$2" > "$S/running"
  echo '{"ok":true,"note":"'"$2"' on XTX at http://127.0.0.1:59991","error":""}' ;;
*)
  echo '{"ok":false,"note":"","error":"stub: unknown command"}'
  exit 1 ;;
esac
