// Dice OCR by template matching — no external engine, fully offline.
//
// Tumbled Boggle dice defeat a general OCR engine: the letters sit at arbitrary
// rotations, the two-letter cubes (Qu, He…) have a tiny second glyph, and each
// die is a bright face ringed by dark tray shadow. So instead of Tesseract we:
//   1. isolate each die's letter (bright face → convex-hull silhouette → erode
//      inward to drop the bevel shadow → keep the dark ink inside),
//   2. match it against every candidate letter rendered in a bold sans font,
//      trying 24 rotations, and keep the best normalised-cross-correlation.
import { rectifyBoard, cropCell } from './warp.js';

const CELL = 160;          // rectified px per cell (bigger than display for OCR)
const NORM = 56;           // template/glyph normalisation canvas
const FILL = 44;           // glyph fills this many px inside NORM, centred
const BLUR = 1.5;          // px, softens stroke/rotation misalignment before NCC
const ANGLES = [];
for (let a = 0; a < 360; a += 15) ANGLES.push(a);

const SINGLES = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
// Multi-letter cubes the solver understands; matched as their printed 2-glyph form.
const BLOCKS = [['QU', 'Qu'], ['HE', 'He'], ['IN', 'In'], ['AN', 'An'], ['TH', 'Th'], ['ER', 'Er']];

// ---------- small canvas helpers ----------
function newCanvas(w, h) {
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  return cv;
}

// A binary ink mask: { w, h, data: Uint8Array } where 1 = ink.
function maskFromAlpha(imgData, threshold = 127) {
  const { data, width: w, height: h } = imgData;
  const out = new Uint8Array(w * h);
  for (let i = 0, j = 0; i < data.length; i += 4, j++) out[j] = data[i] > threshold ? 1 : 0;
  return { w, h, data: out };
}

function bbox(mask) {
  const { w, h, data } = mask;
  let minx = w, miny = h, maxx = -1, maxy = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (data[y * w + x]) {
        if (x < minx) minx = x; if (x > maxx) maxx = x;
        if (y < miny) miny = y; if (y > maxy) maxy = y;
      }
    }
  }
  if (maxx < minx) return null;
  return { minx, miny, maxx, maxy, w: maxx - minx + 1, h: maxy - miny + 1 };
}

// Draw a mask (white ink on black) into a canvas at native size.
function maskToCanvas(mask) {
  const cv = newCanvas(mask.w, mask.h);
  const ctx = cv.getContext('2d');
  const im = ctx.createImageData(mask.w, mask.h);
  for (let j = 0; j < mask.data.length; j++) {
    const v = mask.data[j] ? 255 : 0, k = j * 4;
    im.data[k] = im.data[k + 1] = im.data[k + 2] = v; im.data[k + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
  return cv;
}

// Normalise a mask to a zero-mean unit-norm blurred vector for NCC matching.
function normVec(mask) {
  const b = bbox(mask);
  if (!b || b.w < 2 || b.h < 2) return null;
  const src = maskToCanvas(mask);
  const scale = FILL / Math.max(b.w, b.h);
  const dw = Math.max(1, Math.round(b.w * scale));
  const dh = Math.max(1, Math.round(b.h * scale));
  const cv = newCanvas(NORM, NORM);
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, NORM, NORM);
  ctx.filter = `blur(${BLUR}px)`;
  ctx.drawImage(src, b.minx, b.miny, b.w, b.h,
    (NORM - dw) / 2, (NORM - dh) / 2, dw, dh);
  ctx.filter = 'none';
  const px = ctx.getImageData(0, 0, NORM, NORM).data;
  const N = NORM * NORM;
  const v = new Float32Array(N);
  let mean = 0;
  for (let i = 0, j = 0; j < N; i += 4, j++) { v[j] = px[i]; mean += px[i]; }
  mean /= N;
  let norm = 0;
  for (let j = 0; j < N; j++) { v[j] -= mean; norm += v[j] * v[j]; }
  norm = Math.sqrt(norm);
  if (norm < 1e-6) return null;
  for (let j = 0; j < N; j++) v[j] /= norm;
  return v;
}

// Rotate a mask by deg (clockwise), return a new tight mask.
function rotateMask(mask, deg) {
  if (deg % 360 === 0) return mask;
  const src = maskToCanvas(mask);
  const diag = Math.ceil(Math.hypot(mask.w, mask.h));
  const cv = newCanvas(diag, diag);
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, diag, diag);
  ctx.translate(diag / 2, diag / 2);
  ctx.rotate((deg * Math.PI) / 180);
  ctx.drawImage(src, -mask.w / 2, -mask.h / 2);
  return maskFromAlpha(ctx.getImageData(0, 0, diag, diag), 100);
}

// ---------- templates ----------
let templates = null;
function renderGlyph(text) {
  const S = 200;
  const cv = newCanvas(S, S);
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, S, S);
  ctx.fillStyle = '#fff';
  ctx.font = 'bold 130px "Liberation Sans", Arial, "Helvetica Neue", sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, S / 2, S / 2);
  return maskFromAlpha(ctx.getImageData(0, 0, S, S), 127);
}
function buildTemplates() {
  templates = [];
  for (const ch of SINGLES) templates.push({ name: ch, vec: normVec(renderGlyph(ch)) });
  for (const [name, disp] of BLOCKS) templates.push({ name, vec: normVec(renderGlyph(disp)) });
  templates = templates.filter((t) => t.vec);
}

// ---------- isolation ----------
function otsu(hist, total) {
  let sum = 0;
  for (let t = 0; t < 256; t++) sum += t * hist[t];
  let sumB = 0, wB = 0, max = -1, thr = 127;
  for (let t = 0; t < 256; t++) {
    wB += hist[t]; if (!wB) continue;
    const wF = total - wB; if (!wF) break;
    sumB += t * hist[t];
    const mB = sumB / wB, mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > max) { max = between; thr = t; }
  }
  return thr;
}

// 4-connected component labelling of a boolean mask.
function label4(mask, w, h) {
  const lab = new Int32Array(w * h);
  const sizes = [0];
  const stack = [];
  let n = 0;
  for (let s = 0; s < w * h; s++) {
    if (!mask[s] || lab[s]) continue;
    n++; lab[s] = n; let sz = 0;
    stack.length = 0; stack.push(s);
    while (stack.length) {
      const p = stack.pop(); sz++;
      const x = p % w, y = (p / w) | 0;
      if (x > 0 && mask[p - 1] && !lab[p - 1]) { lab[p - 1] = n; stack.push(p - 1); }
      if (x < w - 1 && mask[p + 1] && !lab[p + 1]) { lab[p + 1] = n; stack.push(p + 1); }
      if (y > 0 && mask[p - w] && !lab[p - w]) { lab[p - w] = n; stack.push(p - w); }
      if (y < h - 1 && mask[p + w] && !lab[p + w]) { lab[p + w] = n; stack.push(p + w); }
    }
    sizes.push(sz);
  }
  return { lab, n, sizes };
}

// Andrew's monotone chain convex hull. pts: [{x,y}] → hull [{x,y}].
function convexHull(pts) {
  if (pts.length < 3) return pts;
  pts = pts.slice().sort((a, b) => a.x - b.x || a.y - b.y);
  const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const lo = [];
  for (const p of pts) {
    while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], p) <= 0) lo.pop();
    lo.push(p);
  }
  const up = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], p) <= 0) up.pop();
    up.push(p);
  }
  lo.pop(); up.pop();
  return lo.concat(up);
}

// Rasterise a polygon into a boolean mask.
function fillPolygon(hull, w, h) {
  const cv = newCanvas(w, h);
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = '#fff';
  ctx.beginPath();
  hull.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
  ctx.closePath(); ctx.fill();
  const px = ctx.getImageData(0, 0, w, h).data;
  const out = new Uint8Array(w * h);
  for (let i = 0, j = 0; j < w * h; i += 4, j++) out[j] = px[i] > 127 ? 1 : 0;
  return out;
}

// Morphological erosion by a (2r+1) square (separable). Out-of-bounds samples
// clamp to the edge (replicate), matching PIL's MinFilter, so a silhouette that
// touches the cell border isn't eaten away there.
function erode(mask, w, h, r) {
  if (r < 1) return mask.slice();
  const tmp = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let ok = 1;
      for (let dx = -r; dx <= r; dx++) {
        let nx = x + dx; if (nx < 0) nx = 0; else if (nx >= w) nx = w - 1;
        if (!mask[y * w + nx]) { ok = 0; break; }
      }
      tmp[y * w + x] = ok;
    }
  }
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let ok = 1;
      for (let dy = -r; dy <= r; dy++) {
        let ny = y + dy; if (ny < 0) ny = 0; else if (ny >= h) ny = h - 1;
        if (!tmp[ny * w + x]) { ok = 0; break; }
      }
      out[y * w + x] = ok;
    }
  }
  return out;
}

// Extract the letter ink from one rectified cell. Returns a mask or null.
function isolate(cellData) {
  const W = CELL, N = W * W, px = cellData.data;
  const val = new Uint8Array(N), gray = new Float32Array(N);
  const sat = new Float32Array(N), vhist = new Array(256).fill(0);
  for (let i = 0, j = 0; j < N; i += 4, j++) {
    const r = px[i], g = px[i + 1], b = px[i + 2];
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    val[j] = mx; sat[j] = mx > 0 ? (mx - mn) / mx : 0;
    gray[j] = r * 0.299 + g * 0.587 + b * 0.114;
    vhist[mx]++;
  }
  const vThr = Math.min(otsu(vhist, N), 140); // 0.55*255 ≈ 140

  // Bright, low-saturation die face; keep the component covering the centre.
  const face = new Uint8Array(N);
  for (let j = 0; j < N; j++) face[j] = (val[j] > vThr && sat[j] < 0.35) ? 1 : 0;
  const { lab, n, sizes } = label4(face, W, W);
  if (!n) return null;
  const lo = W >> 2, hi = W - lo;
  const central = new Array(n + 1).fill(0);
  for (let y = lo; y < hi; y++) for (let x = lo; x < hi; x++) { const l = lab[y * W + x]; if (l) central[l]++; }
  let best = 0;
  for (let i = 1; i <= n; i++) if (central[i] > central[best]) best = i;
  if (!best || sizes[best] < 0.06 * N) return null;

  const facePts = [];
  let gsum = 0, gcount = 0, fminx = W, fminy = W, fmaxx = -1, fmaxy = -1;
  for (let p = 0; p < N; p++) {
    if (lab[p] !== best) continue;
    const x = p % W, y = (p / W) | 0;
    facePts.push({ x, y });
    gsum += gray[p]; gcount++;
    if (x < fminx) fminx = x; if (x > fmaxx) fmaxx = x;
    if (y < fminy) fminy = y; if (y > fmaxy) fmaxy = y;
  }
  const facemean = gsum / gcount;
  const dieMax = Math.max(fmaxx - fminx + 1, fmaxy - fminy + 1);

  const hull = convexHull(facePts);
  if (hull.length < 3) return null;
  // Erode the die silhouette inward to shed the bevel shadow ring (the dark arc
  // where the die meets the tray), which otherwise sticks to the letter.
  const rad = Math.max(2, Math.round(0.13 * dieMax));
  const silFull = fillPolygon(hull, W, W);
  const interior = erode(silFull, W, W, rad);
  let interiorCount = 0;
  for (let j = 0; j < N; j++) interiorCount += interior[j];
  if (interiorCount < 50) return null;
  // The outer few px of the interior are where leftover bevel lives; ink blobs
  // that reach into this ring get dropped.
  const inner = erode(interior, W, W, 3);

  // Letter ink = clearly-dark pixels inside the eroded silhouette.
  const ink = new Uint8Array(N);
  let inkCount = 0;
  for (let j = 0; j < N; j++) {
    if (interior[j] && gray[j] < facemean * 0.72) { ink[j] = 1; inkCount++; }
  }
  if (inkCount < 12) return null;

  const li = label4(ink, W, W);
  // Leftover bevel lies ALONG the interior's outer ring, so most of such a blob
  // sits in the ring; a real letter stroke only crosses it. Drop blobs that are
  // mostly ring (bevel), keep the rest — this spares strokes near the die edge.
  const ringPx = new Float64Array(li.n + 1);
  for (let p = 0; p < N; p++) {
    if (ink[p] && interior[p] && !inner[p]) ringPx[li.lab[p]]++;
  }
  const minBlob = Math.max(15, 0.015 * inkCount);
  const bevel = (i) => ringPx[i] / li.sizes[i] > 0.5;
  let keep = new Set();
  for (let i = 1; i <= li.n; i++) if (li.sizes[i] >= minBlob && !bevel(i)) keep.add(i);
  if (!keep.size) for (let i = 1; i <= li.n; i++) if (li.sizes[i] >= minBlob) keep.add(i);
  if (!keep.size) return null;

  const out = new Uint8Array(N);
  let count = 0, lcx = 0, lcy = 0;
  for (let p = 0; p < N; p++) if (keep.has(li.lab[p])) { out[p] = 1; count++; lcx += p % W; lcy += (p / W) | 0; }
  if (count < 12) return null;
  lcx /= count; lcy /= count;
  const m = { w: W, h: W, data: out };
  const b = bbox(m);
  if (!b) return null;
  // Return a tight crop, plus the orientation-dot direction (if any) for the
  // rotationally-confusable cubes.
  const tight = new Uint8Array(b.w * b.h);
  for (let y = 0; y < b.h; y++)
    for (let x = 0; x < b.w; x++)
      tight[y * b.w + x] = out[(b.miny + y) * W + (b.minx + x)];
  const dotAngle = findDot(silFull, gray, facemean * 0.72, out, lcx, lcy, dieMax, W, N);
  return { w: b.w, h: b.h, data: tight, dotAngle };
}

// Some Boggle cubes print a small dot under the letter to mark "down" (so M vs W
// and Z vs N can be told apart). Find a small, compact ink blob that sits inside
// the die but apart from the letter; return the clockwise-from-up angle pointing
// at it (i.e. the letter's "down"), or null if there's no unambiguous single dot.
function findDot(sil, gray, thr, letter, lcx, lcy, dieMax, W, N) {
  const dark = new Uint8Array(N);
  let dieArea = 0;
  for (let j = 0; j < N; j++) { if (sil[j]) { dieArea++; if (gray[j] < thr) dark[j] = 1; } }
  const { lab, n, sizes } = label4(dark, W, W);
  if (!n) return null;
  const overlap = new Float64Array(n + 1), cx = new Float64Array(n + 1), cy = new Float64Array(n + 1);
  const bx0 = new Int32Array(n + 1).fill(W), by0 = new Int32Array(n + 1).fill(W);
  const bx1 = new Int32Array(n + 1).fill(-1), by1 = new Int32Array(n + 1).fill(-1);
  for (let p = 0; p < N; p++) {
    const l = lab[p]; if (!l) continue;
    const x = p % W, y = (p / W) | 0;
    cx[l] += x; cy[l] += y; if (letter[p]) overlap[l]++;
    if (x < bx0[l]) bx0[l] = x; if (x > bx1[l]) bx1[l] = x;
    if (y < by0[l]) by0[l] = y; if (y > by1[l]) by1[l] = y;
  }
  const minA = 0.003 * dieArea, maxA = 0.05 * dieArea;
  const cands = [];
  for (let i = 1; i <= n; i++) {
    const s = sizes[i];
    if (s < minA || s > maxA) continue;
    if (overlap[i] > 0.15 * s) continue;          // part of the letter itself
    const bw = bx1[i] - bx0[i] + 1, bh = by1[i] - by0[i] + 1;
    if (Math.max(bw, bh) / Math.min(bw, bh) > 1.8) continue; // not roundish
    if (s / (bw * bh) < 0.5) continue;            // not solid/compact
    const dcx = cx[i] / s, dcy = cy[i] / s;
    if (Math.hypot(dcx - lcx, dcy - lcy) < 0.15 * dieMax) continue; // basically on the letter
    cands.push({ dcx, dcy });
  }
  if (cands.length !== 1) return null;            // only trust a single clear dot
  let ang = Math.atan2(cands[0].dcx - lcx, -(cands[0].dcy - lcy)) * 180 / Math.PI;
  return ang < 0 ? ang + 360 : ang;
}

// ---------- matching ----------
// These read as rotations of each other, so free rotation can't tell them apart;
// the orientation dot decides. Cubes that carry a dot: M, W, Z (not N).
const AMBIGUOUS = new Set(['M', 'W', 'Z', 'N']);
const DOTTED = new Set(['M', 'W', 'Z']);

// Best NCC per template across the given glyph rotations.
function matchAngles(mask, angles) {
  const scores = new Map();
  for (const deg of angles) {
    const rv = normVec(rotateMask(mask, deg));
    if (!rv) continue;
    for (const t of templates) {
      let dot = 0;
      const a = rv, b = t.vec;
      for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
      if (dot > (scores.get(t.name) ?? -2)) scores.set(t.name, dot);
    }
  }
  return [...scores.entries()].sort((a, b) => b[1] - a[1]);
}

function classify(mask) {
  let ranked = matchAngles(mask, ANGLES);
  if (!ranked.length) return { char: '', confidence: 0 };

  // Disambiguate M/W/Z/N with the orientation dot when the top guess is one.
  if (AMBIGUOUS.has(ranked[0][0])) {
    if (mask.dotAngle != null) {
      // Re-read at the orientation that puts the dot at the bottom.
      const canon = ((180 - mask.dotAngle) % 360 + 360) % 360;
      const angles = [];
      for (let d = -20; d <= 20; d += 5) angles.push(((canon + d) % 360 + 360) % 360);
      const r2 = matchAngles(mask, angles);
      if (r2.length) ranked = r2;
    } else {
      // No dot ⇒ it can't be a dotted cube; drop those and re-pick (usually N).
      const r2 = ranked.filter(([n]) => !DOTTED.has(n));
      if (r2.length) ranked = r2;
    }
  }

  let bestName = ranked[0][0];
  const best = ranked[0][1];
  const runnerUp = ranked[1] ? ranked[1][1] : 0;
  if (bestName === 'Q') bestName = 'QU'; // a lone Q is always Qu
  return { char: bestName, confidence: confidence(best, best - runnerUp) };
}

// Confidence for the edit grid's "please check this" flagging. Rotated single
// glyphs are confusable (O/C, R/H…), so a cell only reads as confident when the
// match is both strong in absolute terms AND wins by a clear margin. This is a
// review hint, not ground truth — OCR of tumbled dice needs a human pass.
function confidence(best, margin) {
  const abs = Math.max(0, Math.min(1, (best - 0.35) / 0.45)); // 0.35→0 … 0.80→1
  const mar = Math.max(0, Math.min(1, margin / 0.12));        // 0 → 0 … 0.12→1
  return Math.round(abs * (0.35 + 0.65 * mar) * 100);
}

// Runs matching on every cell. onProgress(done,total) as it goes.
// Returns array (row-major) of { char, confidence, rotation, image }.
export async function scanBoard(img, quad, rows, cols, onProgress) {
  if (!templates) buildTemplates();
  const rect = rectifyBoard(img, quad, rows, cols, CELL);
  const rctx = rect.getContext('2d');
  const results = [];
  const total = rows * cols;
  let done = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cellData = rctx.getImageData(c * CELL, r * CELL, CELL, CELL);
      const iso = isolate(cellData);
      const { char, confidence } = iso ? classify(iso) : { char: '', confidence: 0 };
      results.push({
        char,
        confidence,
        rotation: 0,
        image: cropCell(rect, rows, cols, r, c, CELL).toDataURL('image/png'),
      });
      onProgress && onProgress(++done, total);
      // Yield so the progress text can paint between cells.
      await new Promise((res) => setTimeout(res, 0));
    }
  }
  return results;
}
