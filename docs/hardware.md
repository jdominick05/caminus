# Hardware

By the end of this page you'll know whether your cards are supported, what runs on them, and how to get full speed.

## Supported cards

| Card | Status |
|---|---|
| RX 7900 XTX (24 GB) | Supported and tested. |
| RX 7900 XT (20 GB) | Supported and tested. |
| Other cards with the same chip (gfx1100): RX 7900 GRE, Radeon Pro W7900, W7800 | Untested. Caminus uses them and asks you for a hardware report. |
| Every other GPU, including other RX 7000 cards and NVIDIA cards | Not supported yet. Caminus lists them and leaves them alone. |

All testing so far happened on one machine: an RX 7900 XTX and an RX 7900 XT on Linux with 29 GB of RAM. If you run
Caminus on anything else, please [send a hardware report](https://github.com/jdominick05/caminus/issues/new?template=hardware_report.yml),
even when everything works. That's how cards move from untested to supported.

## What runs on how many cards

| Cards | What you can run |
|---|---|
| One | The one-card models. `qwen3.6` fits any supported card; `qwen3.6-hauhau` and `arex-2` need a 24 GB card. |
| Two | `qwen3-coder-next`; `coder` on an XTX + XT; or two one-card models side by side. |
| Three or more | Two cards for a two-card model; one-card models on the rest. |

Caminus finds your cards itself and gives the bigger card the bigger share of a two-card model. `qwen3-coder-next` runs on
any pair. `coder` runs on an XTX + XT in this release: on other pairs its memory can be projected, but its prompt reading
hasn't been tested there yet, so Caminus refuses it rather than guess. On two XT cards the projection also comes out short
of memory; a hardware report from such a pair helps.

## Slots and links

A two-card model sends data between the cards on every token, so both cards need:

- a PCIe 4.0 x4 link or faster (an x16 slot running at x4 is fine),
- direct card-to-card access (peer-to-peer). Most desktop boards have it.

`caminus doctor` checks both. A pair on a slower link still runs one-card models.

## Your monitors

If a card drives your monitors, Caminus keeps 512 MB of its memory free for the desktop. Start models after your desktop
session is up. A browser or game that grabs a lot of memory later can push the card over its limit (see
[troubleshooting](troubleshooting.md)).

## The guard

Every GPU process Caminus starts runs under a guard. It stops the process when a card passes 97% of its memory, and it
stops all GPU work after a GPU fault, so a crash can't repeat and take your desktop down. The guard reads the kernel log
for faults, so add yourself to the `systemd-journal` group (or `adm`) and log in again. `caminus doctor` tells you if it
can't read the log.

## Getting full speed

The cards decode about 13% faster with their clocks held high. The guard does this when it finds a `gpu-high` helper:
any executable that sets each card's `power_dpm_force_performance_level` to `high`, runs the command it's given, and sets
it back afterwards. Writing those files needs root, so the helper needs its own sudo or udev rule. Caminus doesn't ship
one; whether to add one is up to you. Without it, everything works at normal clocks.

## Memory and disk

- System RAM: the one-card models need little. `coder` was tested with 29 GB.
- Disk: the model files (12-59 GB each), plus up to 30 GB for the prompt cache and 10 GB for the request log, both under
  `~/.cache/caminus`. `caminus purge` empties them.
