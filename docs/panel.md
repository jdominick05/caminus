# The panel

By the end of this page you'll know every part of the Caminus panel and its keys. The panel is a terminal UI. It's the
easiest way to download models, start and stop them, and run Qwen Code on them.

```bash
caminus panel            # the basic layout
caminus panel -advanced  # every server option, the card choice, the brain / hands team
```

![The Caminus panel](panel.svg)

## The parts

| Part | What it shows |
|---|---|
| **MODELS** | Every model Caminus knows, how many cards it needs, its size, and whether it's on disk. |
| **SERVERS** | The models you started: their address, context size and current speed. |
| **QWEN CODE** | Whether Qwen Code is installed and set up for your server. Caminus keeps its own Qwen Code settings, so yours in `~/.qwen` stay untouched. |
| **OPTIONS** | Advanced layout only: every option of the server, for the next start. |
| **HARDWARE** | Shows up only when something is wrong with your cards, for example a card in a slot that's too slow. |

`tab` moves between the parts. The keys at the bottom always belong to the part you're in.

## Keys

| Where | Key | What it does |
|---|---|---|
| MODELS | `enter` | Start the model. |
| | `p` | Download it. |
| | `c` | Advanced: choose the card (auto, or a card number). |
| | `b` | Advanced: start the brain / hands team (below). |
| SERVERS | `enter` | Open Qwen Code on this server. |
| | `x` | Stop it. |
| | `t` | Restart it. |
| | `l` | Show its log. |
| | `m` | Open the live monitor. |
| QWEN CODE | `enter` | Launch Qwen Code. |
| | `u` | Set it up for the running server. |
| | `i` | Install it (needs Node.js and npm). |
| OPTIONS | `enter` | Edit or toggle an option. |
| | `d` | Reset it to its default. |
| Anywhere | `a` | Switch between the basic and advanced layouts. |
| | `R` | Refresh. |
| | `q` | Quit. Running models keep running. |

The panel remembers your layout and option values in `~/.config/caminus/panel.json`.

## The live monitor

`m` in **SERVERS**, or `caminus top` in a terminal, opens a live view of a running server: its speed, how much of each
prompt came from the cache, each card's memory and power, and any errors.

### Agent-state lighting

The monitor can light your PC through OpenRGB. Start OpenRGB's **SDK Server**, then add this section to
`~/.config/caminus/top.json` (keep any other settings):

```json
{
  "openrgb": {
    "enabled": true,
    "host": "127.0.0.1:6742",
    "brightness": 0.65,
    "switch_modes": false,
    "fps": 20,
    "speed": 1,
    "fade_ms": 300
  }
}
```

`caminus top` and the panel's monitor inherit these preferences while choosing the selected server's address and agent
sessions. `LLMTOP_CONFIG` selects another preferences file. `caminus top -config FILE` uses a complete monitor config
directly. Your preferences file is left untouched.

By default, the monitor uses devices already in a per-LED mode, such as Direct. Set `switch_modes` to `true` if you want
it to temporarily select a supported Direct, Custom or Static mode. On a normal exit it restores the colors and full
mode settings that OpenRGB reported when it first connected. A lost connection or changed device layout can require
reapplying your OpenRGB profile.

`devices` limits the lights to matching device names, for example `["MYSTIC", "Vengeance"]`; leaving it empty includes
every usable device. The monitor never changes your zone sizes or saves a profile. Pause other lighting effects while
using it. The RGB status appears in **AGENTS**. Preview frames (`-once`) and the headless web view leave lighting alone.

The effects follow what the agent is doing:

| State | Lights |
|---|---|
| Server offline | Dim red. |
| Loading | Orange orbiting lights. |
| Idle or quiet | Steady dim cyan. |
| Between steps | Steady blue. |
| Waiting for you | A slow turquoise breath. |
| Reading a prompt | An amber sweep. |
| Writing tokens | A green comet. |
| Running a tool | Warm orange with a moving highlight. |
| Compacting context | Two orange-red lights that meet and separate. |

Colors fade between states instead of jumping. `fade_ms` sets the fade length (0–2000 ms; 0 disables it), `speed` sets
the animation rate (0.1–4), and `fps` sets the update rate (1–60). The defaults are 300 ms, 1×, and 20 FPS. Try 10 FPS for
sluggish RAM or GPU lighting. Steady frames are sent only when they change.

Eight-LED motherboard zones named **JARGB 2 / JARGB 3** (including **JARGB_V2_2 / JARGB_V2_3**) get circular effects in
opposite directions. RAM and GPU zones stay linear. The monitor uses the LED counts you configured in OpenRGB.
To choose a layout for another zone, add `zones` inside `openrgb`:

```json
"zones": {
  "Pump ring": "ring",
  "Rear fan": "reverse-ring",
  "JARGB 2": "strip"
}
```

The names match OpenRGB's zone names, ignoring case. `strip` disables circular effects for that zone. To change a state
color, add `colors`, for example `"colors": {"writing": "#00ff80", "waiting": "#00b4dc"}`. The state names in the first
column above are `offline`, `loading`, `idle` / `quiet`, `between`, `waiting`, `reading`, `writing`, `tool`, and `compacting`.

## The brain / hands team

With two cards you can run two models together: AREX-2 plans on the 24 GB card (the brain) and Qwen3.6 does the work on
the other card (the hands). Press `b` in the advanced layout, or run `caminus start --team brain-hands`. Qwen Code then
uses the brain as its main model and hands tasks to Qwen3.6.
