# GPU safety

By the end of this page you'll know the rules for running GPU code while working on Caminus, and why each exists. They
come from real incidents: a desktop session lost to repeated GPU resets.

## Every GPU process runs under the guard

```bash
tools/guard.sh LOG SECS CMD...      # in a checkout (the same script as `caminus guard`)
```

The guard watches every gfx1100 card. It kills the process when a card passes 97% of its VRAM or the host runs low on
memory, and on any amdgpu page fault, reset or timeout in the kernel log. `caminus serve`, `caminus replay` and
`caminus doctor --gpu` start themselves under it.

- **Exit 99: a GPU fault.** Stop all GPU work. Don't rerun anything, not even "just to check". Read `journalctl -k`.
- **Exit 98: a fault is on record.** Every guarded run refuses until a person has inspected the fault and removed
  `~/.local/state/k3/GPU_FAULT_STOP` by hand.

Why: an out-of-bounds kernel was once rerun about six times after its first fault. The driver reset both GPUs nine times
and the desktop session was lost.

## New kernel code runs first on a card that doesn't drive a display

On a two-card machine, run a new or changed kernel on the non-display card alone first
(`HIP_VISIBLE_DEVICES=<that card>`), with a small input, before any run on the display card. A fault there doesn't take
the desktop with it.

## Every wait is bounded

The engine's kernels wait on each other through counters in memory. Every such wait has a time limit and reports an error
when it runs out; the host then retries the pass, which gives the same bits. Never write a wait without a limit, and keep
every launch well under 200 ms: a 30-second spin once froze the desktop.

## Bounds are checked twice

Every position the host sends is checked against the allocation before launch, and the device clamps and flags anything
out of range instead of reading or writing past it. To show that an unchecked path is unsafe, emulate it on the CPU (as
`tests/prefill/test_prefill_safety.py` does). Never launch it with bad inputs.

## Memory is counted before it's allocated

Every allocation is projected and checked against the guard's line first (`caminus.device.capacity`), with 512 MiB kept
free on a display card. If you add a buffer, count it there in the same change.
