#!/usr/bin/env bash
# guard.sh LOG SECS CMD...: every GPU command runs under it (CLAUDE.md, GPU safety; docs/dev/packaging.md, "Guard"). Shipped in
# the package (caminus/share/guard.sh; `caminus serve` runs itself under it; tools/guard.sh is a link to this file).
# guard.sh --lines [ROOT]: print the watched cards and their lines (from ROOT's sysfs, default /) and exit; nothing runs.
#
# Runs `timeout SECS [gpu-high] CMD` in a session of its own, logs a 1 Hz line to LOG.mon (time, used MiB per watched card,
# host MemAvailable MiB), and kills the run when a watched card's sysfs mem_info_vram_used passes its line, host MemAvailable
# falls below 1500 MiB, or the kernel log shows an amdgpu page fault / GPU reset / wedge / timeout / an unresponsive MES (exit 99, and the STOP file;
# the display core's REG_WAIT timeouts from a monitor sleeping or changing mode are noted, not counted).
# Every limit comes from the machine, never weaker than before:
#   * watched: every amdgpu card whose KFD node is gfx1100 (gfx_target_version 110000, the target Caminus ships), in DRM order;
#     an iGPU / other target is not watched (its small VRAM carve-out holds the desktop: a 0.97 line would kill for nothing)
#   * line: floor(floor(total_MiB * 97 / 100) / 50) * 50 (0.97 of the card, rounded down to 50 MiB): here the XTX's 24,560 MiB
#     -> 23,800 and the XT's 20,464 -> 19,850, the old fixed lines. caminus.device.machine computes the same for caminus.device.capacity
#   * the display-driving card (a connected connector or boot_vga = 1) is flagged: caminus.device.capacity keeps its 512 MiB reserve
#     below the line (the desktop's own VRAM grows); the guard kills at the line as for every card
#   * the fault watch reads `journalctl -k` when this user can (the systemd-journal / adm groups), else `dmesg` (when
#     kernel.dmesg_restrict = 0), else it says at the start, in the log and on stdout, that the watch is OFF and how to enable it
#   * clocks: CAMINUS_GPU_HIGH, else gpu-high on PATH or in ~/.local/bin (DPM high + runtime PM on for the run); without one the
#     command runs at the current clocks and the guard says so
# GPU fault stop: a fault during a run kills it and records the kernel-log lines in STOP; every later guarded run refuses
# (exit 98) until someone has looked at the fault and removed STOP by hand. A faulting experiment is never re-run.
set -u

ROOT=
gfx_of_minor() {   # the gfx_target_version of the KFD node with this render minor (empty if none)
    local p
    for p in "$ROOT"/sys/class/kfd/kfd/topology/nodes/*/properties; do
        [ -r "$p" ] || continue
        if awk -v m="$1" '$1 == "drm_render_minor" && $2 == m {f = 1} END {exit !f}' "$p"; then
            awk '$1 == "gfx_target_version" {print $2}' "$p"; return
        fi
    done
}
discover() {   # fills CARDS / DEVS / LABELS / TOTALS / LINES / DISPLAY for the watched cards
    CARDS=(); DEVS=(); LABELS=(); TOTALS=(); LINES=(); DISPLAY=()
    local c n dev minor r gfx total did lab disp s
    for c in "$ROOT"/sys/class/drm/card*; do
        n=${c##*/}
        [[ $n =~ ^card[0-9]+$ ]] || continue
        dev=$c/device
        [ -r "$dev/mem_info_vram_total" ] || continue
        if [ -r "$dev/vendor" ] && [ "$(cat "$dev/vendor")" != 0x1002 ]; then continue; fi
        minor=
        for r in "$dev"/drm/renderD*; do [ -e "$r" ] && minor=${r##*renderD}; done
        [ -n "$minor" ] || continue
        gfx=$(gfx_of_minor "$minor")
        [ "$gfx" = 110000 ] || continue
        total=$(( $(cat "$dev/mem_info_vram_total") >> 20 ))
        did=$(cat "$dev/device" 2>/dev/null)
        case "$did:$(( total / 1024 ))" in
            0x744c:23|0x744c:24) lab=XTX ;;
            0x744c:19|0x744c:20) lab=XT ;;
            *) lab=$n ;;
        esac
        disp=0
        for s in "$ROOT"/sys/class/drm/"$n"-*/status; do [ -r "$s" ] && [ "$(cat "$s")" = connected ] && disp=1; done
        [ "$(cat "$dev/boot_vga" 2>/dev/null)" = 1 ] && disp=1
        CARDS+=("$n"); DEVS+=("$dev"); LABELS+=("$lab"); TOTALS+=("$total")
        LINES+=($(( (total * 97 / 100) / 50 * 50 ))); DISPLAY+=("$disp")
    done
    # (in DRM order: the glob sorts card1 < card10 < card2; sort by number)
    local i j t
    for ((i = 0; i < ${#CARDS[@]}; i++)); do for ((j = i + 1; j < ${#CARDS[@]}; j++)); do
        if (( ${CARDS[j]#card} < ${CARDS[i]#card} )); then
            for a in CARDS DEVS LABELS TOTALS LINES DISPLAY; do
                eval "t=\${$a[i]}; $a[i]=\${$a[j]}; $a[j]=\$t"
            done
        fi
    done; done
}

if [ "${1:-}" = --lines ]; then
    ROOT=${2:-}; ROOT=${ROOT%/}
    discover
    for ((i = 0; i < ${#CARDS[@]}; i++)); do
        echo "${CARDS[i]} ${LABELS[i]} total ${TOTALS[i]} line ${LINES[i]} display ${DISPLAY[i]} reserve $(( DISPLAY[i] ? 512 : 0 ))"
    done
    exit 0
fi

if [ $# -lt 3 ]; then
    echo "usage: guard.sh LOG SECS CMD...  (SECS 0: no time limit) | guard.sh --lines [ROOT]" >&2; exit 2
fi
LOG=$1; SECS=$2; shift 2
STOP=${K3_GUARD_STOP:-$HOME/.local/state/k3/GPU_FAULT_STOP}; mkdir -p "$(dirname "$STOP")" "$(dirname "$LOG")"
# (the MES lines: the card's scheduler firmware stopped taking work. Seen 11-09 on kernel 7.0.0-42 after the RX 7900 XT woke from
# runtime suspend: "MES failed to respond to msg=MISC (WAIT_REG_MEM)", then "MES ring buffer is full" every 2 s, fence
# fallbacks on every card and the desktop lagging; only a reboot brought the card back. The DMUB firmware's "auto-load failed"
# on such a wake is not a fault by itself: on kernel 7.0.0-38 it recovers)
FAULTS='amdgpu.*(page fault|GPU reset|wedged|timeout|MES failed to respond|MES ring buffer is full)'
# The display core's register waits (its REG_WAIT macro: "REG_WAIT timeout <n>us * <n> tries - <function> line:<n>") time out
# when a monitor sleeps, wakes or changes mode. They come from the display engine, not from a compute queue, so they are
# logged as a note and never count as a fault (a monitor sleeping overnight used to record a fault and refuse every later
# start). Page faults, resets, wedges and ring timeouts on the same lines' cards stay faults.
DISPLAY_NOISE='REG_WAIT timeout [0-9]+us \* [0-9]+ tries - '
if [ -e "$STOP" ]; then echo "GUARD REFUSED: a GPU fault was recorded ($STOP). Stop all GPU work and report to the lead." | tee "$LOG"; exit 98; fi
discover
if [ ${#CARDS[@]} -eq 0 ]; then
    echo "GUARD REFUSED: no gfx1100 card found to watch (amdgpu cards with a KFD node of gfx_target_version 110000)." | tee "$LOG"; exit 97
fi
T0=$(date '+%Y-%m-%d %H:%M:%S'); U0=$(awk '{print $1}' /proc/uptime)
# the kernel log: journalctl -k (readable when it prints this boot's kernel lines), else dmesg, else none
if [ -n "$(journalctl -k -b -n 1 --no-pager -q 2>/dev/null)" ]; then KLOG=journal
elif dmesg > /dev/null 2>&1; then KLOG=dmesg
else KLOG=none; fi
klog() {   # the kernel log since the run started
    case $KLOG in
        journal) journalctl -k --no-pager --since "$T0" 2>/dev/null ;;
        dmesg) dmesg 2>/dev/null | awk -v u0="$U0" '{ if (match($0, /^\[ *[0-9]+\.[0-9]+\]/)) { t = substr($0, 2, RLENGTH - 2) + 0; if (t >= u0) print } }' ;;
    esac
}
faults() { klog | grep -E "$FAULTS" | grep -vE "$DISPLAY_NOISE"; }   # the fault lines, the display core's register waits left out
GH=${CAMINUS_GPU_HIGH:-$(command -v gpu-high 2>/dev/null || true)}
[ -z "$GH" ] && [ -x "$HOME/.local/bin/gpu-high" ] && GH=$HOME/.local/bin/gpu-high
: > "$LOG"
note() { echo "$1" | tee -a "$LOG"; }
if [ "$KLOG" = none ]; then
    note "GUARD WARNING: the kernel log is not readable by $(id -un) (journalctl -k printed nothing, dmesg is restricted): the amdgpu fault watch is OFF for this run. Add the user to the systemd-journal (or adm) group, or set kernel.dmesg_restrict=0, to enable it."
fi
if [ -z "$GH" ]; then
    note "GUARD NOTE: no gpu-high (CAMINUS_GPU_HIGH, PATH, ~/.local/bin): the command runs at the current clocks (decode is ~13 % slower at DPM auto)."
    RUN=()
else
    RUN=("$GH")
fi
export CAMINUS_GUARD=$$     # (caminus serve: already guarded, no second guard)
setsid timeout "$SECS" "${RUN[@]}" "$@" >> "$LOG" 2>&1 &
P=$!; t0=$(date +%s.%N); mina=999999; noted=0
PEAK=(); for ((i = 0; i < ${#CARDS[@]}; i++)); do PEAK+=(0); done
while kill -0 $P 2>/dev/null; do
  a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  line="$(date +%s)"; over=; kill_=0
  for ((i = 0; i < ${#CARDS[@]}; i++)); do
    u=$(( $(cat "${DEVS[i]}/mem_info_vram_used") >> 20 ))
    line+=" $u"; over+=" ${LABELS[i]}=$u/${LINES[i]}"
    (( u > PEAK[i] )) && PEAK[i]=$u
    (( u > LINES[i] )) && kill_=1
  done
  echo "$line $a" >> "$LOG.mon"
  (( a < mina )) && mina=$a
  (( a < 1500 )) && kill_=1
  if (( kill_ )); then echo "GUARD KILL$over avail=$a MiB (lines: used/line MiB; host MemAvailable floor 1500)" | tee -a "$LOG"; kill -TERM -- -$P; sleep 3; kill -KILL -- -$P 2>/dev/null; fi
  F=$(faults | head -3)
  if (( ! noted )); then
    D=$(klog | grep -E "$DISPLAY_NOISE" | head -1)
    [ -n "$D" ] && { note "GUARD NOTE: a display-engine register wait timed out (a monitor sleeping, waking or changing mode); not a GPU fault, the run continues: $D"; noted=1; }
  fi
  if [ -n "$F" ]; then echo "$F" >> "$STOP"; echo "GPU FAULT: $F -- killed the run; STOP all GPU work, do not re-run, report to the lead." | tee -a "$LOG"; kill -TERM -- -$P; sleep 3; kill -KILL -- -$P 2>/dev/null; fi
  sleep 1
done
wait $P; rc=$?
[ -e "$STOP" ] && [ -n "$(faults)" ] && rc=99
peaks=; for ((i = 0; i < ${#CARDS[@]}; i++)); do peaks+=" peak ${LABELS[i]} ${PEAK[i]} MiB"; done
echo "rc=$rc wall=$(echo "$(date +%s.%N) - $t0" | bc) s$peaks min avail ${mina} MiB"
exit $rc
