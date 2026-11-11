# Design choices

The project's lasting choices, with their reasons. Add one here when a pull request settles something future
contributors would otherwise ask about. Older code comments cite decision numbers (D-NNN) from the private history; this
page replaces them going forward.

| Decision | Why |
|---|---|
| Support RX 7900 XTX and XT first; other gfx1100 cards are allowed but untested. | Every kernel is tuned and verified on those two. Hardware reports widen the list. |
| No card is hardcoded. Cards are found by PCI address and placed by size, then link speed, then address. | Users have any mix and order of cards. A deterministic order keeps prompt-cache keys valid. |
| Engine changes must be bitwise equal to the previous build, or held to llama.cpp's own noise. | It's the only way to change fast code often without silently losing quality. |
| Serve mixture-of-experts and low-active-parameter models; the dense exceptions are AREX-2 and the Qwen3.8 27B line (Swift 1.5). | Decode speed on these cards is bound by bytes read per token. |
| Drafters are DFlash and MTP only. | Measured to pay on agent workloads; copy and n-gram drafters weren't kept. |
| One persistent kernel launch per pass. | Launch overhead dominates small-batch decoding on these cards. |
| Development happens in the public repository: topic branches into `testing`, then `main` for releases. | Contributors see and review the real history. |
| Outside contributions need a CLA. | It keeps the option to offer the project under another licence next to the AGPL. |
| Every GPU process runs under the guard; new kernel code runs first on a non-display card. | Repeated GPU faults once reset both cards nine times and lost the desktop session. |
