// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

// Web view: the same frame as the TUI, rendered to SVG and served over HTTP, for watching from a browser on
// another machine. Off unless "web": {"enabled": true} or -web. Read-only: GET only, nothing is written.
package main

import (
	"compress/gzip"
	"crypto/subtle"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/charmbracelet/x/ansi"
)

type webCfg struct {
	Enabled bool   `json:"enabled"`
	Listen  string `json:"listen"` // host:port; ":8765" = every interface
	Token   string `json:"token"`  // required as ?token= or Bearer when set
}

// ---------- web ----------

// serveWeb blocks; it returns the listener's error. A bare port means every interface.
func (st *state) serveWeb(addr, token string) error {
	srv := &http.Server{Addr: listenAddr(addr), Handler: st.webHandler(token), ReadHeaderTimeout: 5 * time.Second,
		WriteTimeout: 15 * time.Second, IdleTimeout: 60 * time.Second}
	return srv.ListenAndServe()
}

func listenAddr(a string) string {
	if !strings.Contains(a, ":") {
		return ":" + a
	}
	return a
}

func (st *state) webHandler(token string) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		fmt.Fprint(w, webPage)
	})
	mux.HandleFunc("/frame", func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query()
		num := func(k string, def, lo, hi int) int {
			n, err := strconv.Atoi(q.Get(k))
			if err != nil {
				n = def
			}
			return min(max(n, lo), hi)
		}
		// the page keeps its own selection (a session path), so a browser never moves the TUI's
		sel := q.Get("sel")
		m := model{st: st, w: num("w", 120, minW, 200), h: num("h", 40, 10, 150), sel: &sel,
			frame: int(time.Now().UnixMilli() / 150)}
		if d := num("d", 0, -1, 1); d != 0 {
			m.pick(d)
		}
		svg := frameSVG(m.render())
		h := w.Header()
		h.Set("Content-Type", "image/svg+xml")
		h.Set("X-Llmtop-Sel", url.QueryEscape(sel))
		if !strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
			fmt.Fprint(w, svg)
			return
		}
		h.Set("Content-Encoding", "gzip")
		z := gzip.NewWriter(w)
		fmt.Fprint(z, svg)
		z.Close()
	})
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h := w.Header()
		h.Set("Cache-Control", "no-store")
		h.Set("X-Content-Type-Options", "nosniff")
		h.Set("Referrer-Policy", "no-referrer") // the token may be in the URL
		h.Set("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			h.Set("Allow", "GET, HEAD")
			http.Error(w, "read-only", http.StatusMethodNotAllowed)
			return
		}
		if token != "" {
			got := r.URL.Query().Get("token")
			if a := r.Header.Get("Authorization"); strings.HasPrefix(a, "Bearer ") {
				got = a[7:]
			}
			if subtle.ConstantTimeCompare([]byte(got), []byte(token)) != 1 {
				http.Error(w, "token required: open /?token=<web.token>", http.StatusUnauthorized)
				return
			}
		}
		mux.ServeHTTP(w, r)
	})
}

// frameSVG draws a rendered frame on a cols x rows grid; the page scales it to the window's width.
func frameSVG(s string) string {
	rows := parseANSI(strings.TrimRight(s, "\n"))
	cols := 0
	for _, r := range rows {
		cols = max(cols, len(r))
	}
	var b strings.Builder
	fmt.Fprintf(&b, `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %.0f %.0f" data-cols="%d" font-family="%s" font-size="14" xml:space="preserve">`,
		float64(cols)*cw, float64(len(rows))*chH, cols, svgFont)
	for i, r := range rows {
		if ln := svgLine(r); ln != "" {
			fmt.Fprintf(&b, `<g transform="translate(0 %.1f)">%s</g>`, float64(i)*chH, ln)
		}
	}
	b.WriteString("</svg>")
	return b.String()
}

// webPage fetches a frame sized to the window every second. Tab / shift+tab (or a tap) pick the session, like
// in the TUI. When the machine can't be reached the last frame stays, dimmed, with the time it was taken.
const webPage = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>llm-top</title>
<style>
html,body{margin:0;background:#1d2129;color:#6b7489;font:13px system-ui,sans-serif}
#f{padding:8px}#f svg{display:block;width:100%;height:auto}
#f.stale{opacity:.45}
#s{position:fixed;right:8px;bottom:6px;display:none;background:#3b4252;color:#ebcb8b;padding:2px 8px;border-radius:4px}
</style></head><body><div id="f"></div><div id="s"></div>
<script>
const tok = new URLSearchParams(location.search).get('token') || '', f = document.getElementById('f'),
  s = document.getElementById('s');
let sel = '', busy = false, last = 0;
function size() { // a cell is 8.4 x 17 at 14 px. Below 110 columns the panels stack (a phone): the page
  // scrolls then, so the frame gets at least 70 rows instead of squeezing every panel onto one screen.
  const fit = (innerWidth - 16) / 8.4, w = Math.max(40, Math.min(200, Math.floor(fit)));
  const h = Math.floor((innerHeight - 16) / 17 / (fit / w)); // the SVG is scaled by fit / w
  return [w, Math.max(w < 110 ? 70 : 20, h)];
}
async function load(d) {
  if (busy && !d) return;
  busy = true;
  const [w, h] = size();
  try {
    const r = await fetch('frame?w=' + w + '&h=' + h + '&d=' + (d || 0) + '&sel=' + sel +
      (tok ? '&token=' + encodeURIComponent(tok) : ''), {cache: 'no-store', signal: AbortSignal.timeout(4000)});
    if (!r.ok) throw new Error(r.status + ' ' + (await r.text()));
    sel = r.headers.get('X-Llmtop-Sel') || '';
    f.innerHTML = await r.text();
    f.className = '', s.style.display = 'none', last = Date.now();
  } catch (e) {
    f.className = 'stale', s.style.display = 'block';
    s.textContent = (last ? 'offline since ' + new Date(last).toLocaleTimeString() : 'no connection') + ': ' + e.message;
  } finally { busy = false; }
}
addEventListener('keydown', e => { if (e.key === 'Tab') { e.preventDefault(); load(e.shiftKey ? -1 : 1); } });
f.addEventListener('click', () => load(1));
addEventListener('resize', () => load(0));
load(0);
setInterval(() => { // a dropped VPN can hang a request without an error: dim as soon as frames stop
  if (last && Date.now() - last > 3000 && !f.className) {
    f.className = 'stale', s.style.display = 'block';
    s.textContent = 'no update since ' + new Date(last).toLocaleTimeString();
  }
  load(0);
}, 1000);
</script></body></html>
`

const svgFont = `'MesloLGS NF','Cascadia Mono','JetBrains Mono','SF Mono',Menlo,Consolas,'DejaVu Sans Mono',monospace`

// ---------- ANSI -> SVG ----------

// Shared with the README screenshots (screenshots_test.go).

// A cell is one terminal column. A wide glyph (emoji, CJK) takes two, the second a cont cell, so columns line up
// with render()'s layout; s is the whole grapheme cluster, combining marks included.
type cell struct {
	r        rune
	s        string
	cont     bool
	fg, bg   string
	bold     bool
	hasStyle bool
}

func parseANSI(s string) [][]cell {
	var out [][]cell
	for _, l := range strings.Split(s, "\n") {
		var row []cell
		fg, bg, bold := "", "", false
		for i := 0; i < len(l); {
			if l[i] == 0x1b && i+1 < len(l) && l[i+1] == '[' {
				j := i + 2
				for j < len(l) && (l[j] < 0x40 || l[j] > 0x7e) {
					j++
				}
				if j < len(l) && l[j] == 'm' {
					ps := strings.Split(l[i+2:j], ";")
					for k := 0; k < len(ps); k++ {
						switch ps[k] {
						case "", "0":
							fg, bg, bold = "", "", false
						case "1":
							bold = true
						case "22":
							bold = false
						case "39":
							fg = ""
						case "49":
							bg = ""
						case "38", "48":
							if k+4 < len(ps) && ps[k+1] == "2" {
								c := rgb(ps[k+2], ps[k+3], ps[k+4])
								if ps[k] == "38" {
									fg = c
								} else {
									bg = c
								}
								k += 4
							}
						}
					}
				}
				i = j + 1
				continue
			}
			g, wd := ansi.FirstGraphemeCluster(l[i:], ansi.GraphemeWidth)
			if g == "" {
				g = l[i : i+1]
			}
			i += len(g)
			if wd == 0 { // nothing render() counted a column for
				if n := len(row); n > 0 {
					row[n-1].s += g
				}
				continue
			}
			r, _ := utf8.DecodeRuneInString(g)
			row = append(row, cell{r: r, s: g, fg: fg, bg: bg, bold: bold})
			for ; wd > 1; wd-- {
				row = append(row, cell{cont: true, fg: fg, bg: bg, bold: bold})
			}
		}
		out = append(out, row)
	}
	return out
}

func rgb(r, g, b string) string {
	x, _ := strconv.Atoi(r)
	y, _ := strconv.Atoi(g)
	z, _ := strconv.Atoi(b)
	return fmt.Sprintf("#%02x%02x%02x", x, y, z)
}

const (
	cw, chH  = 8.4, 17.0 // cell size at 14 px
	padX     = 16.0
	padTop   = 40.0
	bgColor  = "#1d2129"
	defColor = "#d8dee9"
)

var eighths = map[rune]int{'▁': 1, '▂': 2, '▃': 3, '▄': 4, '▅': 5, '▆': 6, '▇': 7, '█': 8}

// svgLine draws one terminal line at y = 0. Blocks and box-drawing lines become rectangles and text is placed
// word by word, so the layout holds in any monospace font.
func svgLine(row []cell) string {
	var b strings.Builder
	for i := 0; i < len(row); {
		c := row[i]
		j := i + 1
		for j < len(row) && row[j].fg == c.fg && row[j].bg == c.bg && row[j].bold == c.bold {
			j++
		}
		if c.bg != "" {
			fmt.Fprintf(&b, `<rect x="%.1f" y="0" width="%.1f" height="%.1f" fill="%s"/>`, float64(i)*cw, float64(j-i)*cw+.3, chH, c.bg)
		}
		col := c.fg
		if col == "" {
			col = defColor
		}
		weight := ""
		if c.bold {
			weight = ` font-weight="bold"`
		}
		for k := i; k < j; {
			r := row[k].r
			switch {
			case eighths[r] > 0 || r == '░': // merge equal blocks into one rect
				m := k + 1
				for m < j && row[m].r == r {
					m++
				}
				x, w := float64(k)*cw, float64(m-k)*cw+.3
				if r == '░' {
					fmt.Fprintf(&b, `<rect x="%.1f" y="3" width="%.1f" height="%.1f" fill="%s" opacity=".35"/>`, x, w-.3, chH-6, col)
				} else {
					h := chH * float64(eighths[r]) / 8
					extra := 0.0
					if r == '█' {
						extra = .3 // no hairline between rows
					}
					fmt.Fprintf(&b, `<rect x="%.1f" y="%.2f" width="%.1f" height="%.2f" fill="%s"/>`, x, chH-h, w, h+extra, col)
				}
				k = m
			case boxes[r] != "":
				fmt.Fprintf(&b, `<g fill="%s">%s</g>`, col, boxRects(r, float64(k)*cw))
				k++
			case r == ' ':
				k++
			default: // a word: up to the next space, block or box char
				m := k + 1
				for m < j && row[m].r != ' ' && eighths[row[m].r] == 0 && row[m].r != '░' && boxes[row[m].r] == "" {
					m++
				}
				fmt.Fprintf(&b, `<text x="%.1f" y="13" fill="%s"%s textLength="%.1f" lengthAdjust="spacingAndGlyphs">%s</text>`,
					float64(k)*cw, col, weight, float64(m-k)*cw, xmlEsc(textOf(row[k:m])))
				k = m
			}
		}
		i = j
	}
	return b.String()
}

// boxes: which sides a box-drawing char reaches (l r u d); upper case = double line.
var boxes = map[rune]string{
	'─': "lr", '│': "ud", '╭': "rd", '╮': "ld", '╰': "ru", '╯': "lu", '┌': "rd", '┐': "ld", '└': "ru", '┘': "lu",
	'═': "LR", '║': "UD", '╔': "RD", '╗': "LD", '╚': "RU", '╝': "LU",
}

// boxRects draws a box char as rects. Double lines sit 1.6 px either side of the center; at a corner the outer
// line starts where the other outer line runs, the inner one where the other inner one runs.
func boxRects(r rune, x float64) string {
	spec := boxes[r]
	double := spec[0] < 'a'
	sp := strings.ToLower(spec)
	has := func(c byte) bool { return strings.IndexByte(sp, c) >= 0 }
	cx, cy := x+cw/2, chH/2
	const t = 1.3
	var b strings.Builder
	rect := func(x0, y0, x1, y1 float64) {
		fmt.Fprintf(&b, `<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f"/>`, x0, y0, x1-x0, y1-y0)
	}
	sh, sv := 0.0, 0.0 // which way the arms go at a corner
	if has('r') != has('l') {
		sh = map[bool]float64{true: 1, false: -1}[has('r')]
	}
	if has('d') != has('u') {
		sv = map[bool]float64{true: 1, false: -1}[has('d')]
	}
	offs := []float64{0}
	if double {
		offs = []float64{-1.6, 1.6}
	}
	for _, o := range offs {
		if has('l') || has('r') {
			x0, x1 := x, x+cw+.3
			if sh != 0 && sv != 0 {
				if st := cx + sh*sv*o; sh > 0 {
					x0 = st - t/2
				} else {
					x1 = st + t/2
				}
			}
			rect(x0, cy+o-t/2, x1, cy+o+t/2)
		}
		if has('u') || has('d') {
			y0, y1 := 0.0, chH+.3
			if sh != 0 && sv != 0 {
				if st := cy + sv*sh*o; sv > 0 {
					y0 = st - t/2
				} else {
					y1 = st + t/2
				}
			}
			rect(cx+o-t/2, y0, cx+o+t/2, y1)
		}
	}
	return b.String()
}

func textOf(cs []cell) string {
	var b strings.Builder
	for _, c := range cs {
		if !c.cont {
			b.WriteString(c.s)
		}
	}
	return b.String()
}

func xmlEsc(s string) string {
	return strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;", `"`, "&quot;").Replace(s)
}
