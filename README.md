# 🔤 Boggle Finder

A camera-based Boggle solver that runs entirely in your browser. Scan a board with
your phone, correct any misreads, and get every valid word — with the exact path
drawn over your photo when you tap a word.

**Live app:** https://therealsamkothatsreal.github.io/bogglefinder/

## How it works

1. **Scan** — take a photo (or upload one) of a 4×4 or 5×5 board.
2. **Align** — drag four corner handles so the grid overlay sits on the letters.
3. **Read** — [Tesseract.js](https://tesseract.projectnaptha.com/) OCRs each cell at all
   four 90° rotations and keeps the most-confident letter, so sideways dice are read
   correctly (a lone **Q** becomes **Qu**). Low-confidence cells are highlighted for you to fix.
4. **Solve** — a trie-backed depth-first search finds every dictionary word reachable
   by the Boggle adjacency rules (8-way neighbours, no cell reused).
5. **Explore** — words are listed longest-first with Boggle scores; tap one to see the
   numbered path traced on your board image.

No manual scan? Enter the letters by hand and it solves a rendered board instead.

## Features

- 📷 Works from the camera or a saved image (EXIF orientation handled)
- 🔠 Per-cell OCR with a perspective-correct board rectifier and rotation-invariant reading
- ✏️ Editable grid with confidence highlighting and automatic **Qu**
- 🧠 ~172k-word [ENABLE](https://github.com/dolph/dictionary) dictionary, exhaustive DFS
- 🗺️ Path overlay drawn on the original photo (start = green, end = red)
- 📱 Installable PWA, fully offline after first load

## Boggle rules used

- Words are ≥ 3 letters, formed from horizontally, vertically or diagonally adjacent dice.
- No die may be used twice in a single word.
- The **Qu** die counts as two letters.
- Scoring: 3–4 = 1, 5 = 2, 6 = 3, 7 = 5, 8+ = 11.

## Tech

Plain static site — no build step. ES modules, Canvas 2D for the warp/overlay,
Tesseract.js (CDN) for OCR, a service worker for offline use. Hosted on GitHub Pages.

## Local development

```bash
python3 -m http.server 8000   # then open http://localhost:8000
```

(A local server is required — ES modules and the service worker won't run from `file://`.)
