// Geometry: map a 4-corner quad over the board to individual cells,
// rectify the board for OCR, and locate cell centres for path overlay.
// quad = [TL, TR, BR, BL], each { x, y }, in the source image's pixel space.

function lerp(a, b, t) {
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
}

// Bilinear interpolation across the quad. u,v in [0,1]; (0,0)=TL, (1,1)=BR.
export function bilinear(quad, u, v) {
  const [TL, TR, BR, BL] = quad;
  const top = lerp(TL, TR, u);
  const bottom = lerp(BL, BR, u);
  return lerp(top, bottom, v);
}

export function cellCenter(quad, rows, cols, r, c) {
  return bilinear(quad, (c + 0.5) / cols, (r + 0.5) / rows);
}

// Affine-warp a source triangle onto a destination triangle, clipped.
function drawTriangle(ctx, img, s, d) {
  const [s0, s1, s2] = s;
  const [d0, d1, d2] = d;
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(d0.x, d0.y);
  ctx.lineTo(d1.x, d1.y);
  ctx.lineTo(d2.x, d2.y);
  ctx.closePath();
  ctx.clip();

  const denom = s0.x * (s2.y - s1.y) - s1.x * s2.y + s2.x * s1.y + (s1.x - s2.x) * s0.y;
  if (denom === 0) { ctx.restore(); return; }
  const a = (d0.x * (s2.y - s1.y) - d1.x * s2.y + d2.x * s1.y + (d1.x - d2.x) * s0.y) / denom;
  const b = (d0.y * (s2.y - s1.y) - d1.y * s2.y + d2.y * s1.y + (d1.y - d2.y) * s0.y) / denom;
  const c = (s0.x * (d2.x - d1.x) - s1.x * d2.x + s2.x * d1.x + (s1.x - s2.x) * d0.x) / denom;
  const dd = (s0.x * (d2.y - d1.y) - s1.x * d2.y + s2.x * d1.y + (s1.x - s2.x) * d0.y) / denom;
  const e = (s0.x * (s2.y * d1.x - s1.y * d2.x) + s0.y * (s1.x * d2.x - s2.x * d1.x) + (s2.x * s1.y - s1.x * s2.y) * d0.x) / denom;
  const f = (s0.x * (s2.y * d1.y - s1.y * d2.y) + s0.y * (s1.x * d2.y - s2.x * d1.y) + (s2.x * s1.y - s1.x * s2.y) * d0.y) / denom;

  ctx.setTransform(a, b, c, dd, e, f);
  ctx.drawImage(img, 0, 0);
  ctx.restore();
}

// Best-guess the board quad by finding the tray: the largest contrasting blob
// around the image centre, then its four extreme corner points (works for a
// board photographed at an angle). Returns [TL,TR,BR,BL] in full-res px, or null.
export function autoDetectQuad(img) {
  const scale = 240 / Math.max(img.width, img.height);
  const w = Math.max(1, Math.round(img.width * scale));
  const h = Math.max(1, Math.round(img.height * scale));
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  cv.getContext('2d').drawImage(img, 0, 0, w, h);
  const px = cv.getContext('2d').getImageData(0, 0, w, h).data;
  const N = w * h;
  // Estimate the table background from the outer border ring; the board is
  // everything that differs from it (tray + dice together = one blob).
  const ring = Math.max(2, (Math.min(w, h) * 0.06) | 0);
  let bR = 0, bG = 0, bB = 0, bc = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x < ring || x >= w - ring || y < ring || y >= h - ring) {
        const k = (y * w + x) * 4; bR += px[k]; bG += px[k + 1]; bB += px[k + 2]; bc++;
      }
    }
  }
  bR /= bc; bG /= bc; bB /= bc;
  const dist = new Uint8Array(N);
  for (let j = 0; j < N; j++) {
    const k = j * 4;
    dist[j] = Math.min(255, Math.round(Math.hypot(px[k] - bR, px[k + 1] - bG, px[k + 2] - bB)));
  }
  // Threshold at the background noise level so the whole board (tray + dice)
  // stays one blob, rather than Otsu splitting dice from tray.
  let dSum = 0, dSum2 = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x < ring || x >= w - ring || y < ring || y >= h - ring) {
        const d = dist[y * w + x]; dSum += d; dSum2 += d * d;
      }
    }
  }
  const bm = dSum / bc, bStd = Math.sqrt(Math.max(0, dSum2 / bc - bm * bm));
  const thr = Math.min(120, Math.max(22, bm + 3 * bStd));
  const mask = new Uint8Array(N);
  for (let j = 0; j < N; j++) mask[j] = dist[j] > thr ? 1 : 0;

  // Largest connected component (4-connectivity, iterative flood fill).
  const seen = new Uint8Array(N);
  let comp = [], compSize = 0;
  const stack = [];
  for (let s = 0; s < N; s++) {
    if (!mask[s] || seen[s]) continue;
    const cur = []; stack.length = 0; stack.push(s); seen[s] = 1;
    while (stack.length) {
      const p = stack.pop(); cur.push(p);
      const x = p % w, y = (p / w) | 0;
      if (x > 0 && mask[p - 1] && !seen[p - 1]) { seen[p - 1] = 1; stack.push(p - 1); }
      if (x < w - 1 && mask[p + 1] && !seen[p + 1]) { seen[p + 1] = 1; stack.push(p + 1); }
      if (y > 0 && mask[p - w] && !seen[p - w]) { seen[p - w] = 1; stack.push(p - w); }
      if (y < h - 1 && mask[p + w] && !seen[p + w]) { seen[p + w] = 1; stack.push(p + w); }
    }
    if (cur.length > compSize) { compSize = cur.length; comp = cur; }
  }
  if (compSize < N * 0.10) return null; // no convincing board region

  // The blob above is the whole tray (rim included). The grid we want sits on
  // the DICE, which are the brighter faces inside the tray — narrow to those so
  // the quad lands on the die grid, not the outer rim.
  const hist2 = new Array(256).fill(0);
  for (const p of comp) hist2[(px[p * 4] * 0.299 + px[p * 4 + 1] * 0.587 + px[p * 4 + 2] * 0.114) | 0]++;
  let sum2 = 0;
  for (let t = 0; t < 256; t++) sum2 += t * hist2[t];
  let sB = 0, wB2 = 0, bv = -1, dThr = 127;
  for (let t = 0; t < 256; t++) {
    wB2 += hist2[t]; if (!wB2) continue;
    const wF = compSize - wB2; if (!wF) break;
    sB += t * hist2[t];
    const mB = sB / wB2, mF = (sum2 - sB) / wF;
    const bt = wB2 * wF * (mB - mF) * (mB - mF);
    if (bt > bv) { bv = bt; dThr = t; }
  }
  const dice = [];
  for (const p of comp) {
    const g = (px[p * 4] * 0.299 + px[p * 4 + 1] * 0.587 + px[p * 4 + 2] * 0.114) | 0;
    if (g > dThr) dice.push(p);
  }
  // Use the dice extent only if it's a believable share of the board.
  const pts = dice.length > compSize * 0.15 && dice.length < compSize * 0.95 ? dice : comp;

  // Four extreme points → quad corners (robust to rotation/perspective).
  let tl, tr, br, bl, tlv = 1e9, brv = -1e9, trv = -1e9, blv = 1e9;
  for (const p of pts) {
    const x = p % w, y = (p / w) | 0, s = x + y, d = x - y;
    if (s < tlv) { tlv = s; tl = { x, y }; }
    if (s > brv) { brv = s; br = { x, y }; }
    if (d > trv) { trv = d; tr = { x, y }; }
    if (d < blv) { blv = d; bl = { x, y }; }
  }
  const up = (p) => ({ x: p.x / scale, y: p.y / scale });
  return [up(tl), up(tr), up(br), up(bl)];
}

// Warp the board region into an upright rows*cols grid image (for OCR).
export function rectifyBoard(img, quad, rows, cols, cellPx = 128) {
  const W = cols * cellPx, H = rows * cellPx;
  const cv = document.createElement('canvas');
  cv.width = W; cv.height = H;
  const ctx = cv.getContext('2d');
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const p00 = bilinear(quad, c / cols, r / rows);
      const p10 = bilinear(quad, (c + 1) / cols, r / rows);
      const p11 = bilinear(quad, (c + 1) / cols, (r + 1) / rows);
      const p01 = bilinear(quad, c / cols, (r + 1) / rows);
      const d00 = { x: c * cellPx, y: r * cellPx };
      const d10 = { x: (c + 1) * cellPx, y: r * cellPx };
      const d11 = { x: (c + 1) * cellPx, y: (r + 1) * cellPx };
      const d01 = { x: c * cellPx, y: (r + 1) * cellPx };
      drawTriangle(ctx, img, [p00, p10, p01], [d00, d10, d01]);
      drawTriangle(ctx, img, [p10, p11, p01], [d10, d11, d01]);
    }
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  return cv;
}

// Crop one cell as a clean colour image (for showing the die in the edit grid,
// so the player can see the letter's actual orientation).
export function cropCell(rect, rows, cols, r, c, cellPx, out = 104) {
  const inset = Math.round(cellPx * 0.06);
  const sx = c * cellPx + inset, sy = r * cellPx + inset, s = cellPx - inset * 2;
  const cv = document.createElement('canvas');
  cv.width = out; cv.height = out;
  const ctx = cv.getContext('2d');
  ctx.drawImage(rect, sx, sy, s, s, 0, 0, out, out);
  return cv;
}

// Rectify the board and return a data-URL colour crop for every cell.
export function cellColorImages(img, quad, rows, cols, cellPx = 128) {
  const rect = rectifyBoard(img, quad, rows, cols, cellPx);
  const arr = [];
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      arr.push(cropCell(rect, rows, cols, r, c, cellPx).toDataURL('image/png'));
  return arr;
}

// Extract one cell as a clean, centred black-on-white glyph ready for OCR.
// Strategy that works on real photos: binarise (Otsu), decide text polarity by
// the majority-is-background rule, find the glyph's bounding box, then re-centre
// and scale it onto a white canvas — exactly what Tesseract wants to see.
export function extractCell(rectCanvas, rows, cols, r, c, cellPx, out = 180) {
  const inset = Math.round(cellPx * 0.11); // trim tray gaps / die edges
  const sx = c * cellPx + inset;
  const sy = r * cellPx + inset;
  const W = cellPx - inset * 2;

  const work = document.createElement('canvas');
  work.width = W; work.height = W;
  const wctx = work.getContext('2d');
  wctx.drawImage(rectCanvas, sx, sy, W, W, 0, 0, W, W);

  const px = wctx.getImageData(0, 0, W, W).data;
  const N = W * W;
  const gray = new Uint8Array(N);
  const hist = new Array(256).fill(0);
  for (let i = 0, j = 0; i < px.length; i += 4, j++) {
    const g = (px[i] * 0.299 + px[i + 1] * 0.587 + px[i + 2] * 0.114) | 0;
    gray[j] = g; hist[g]++;
  }

  // Otsu threshold.
  let sum = 0;
  for (let t = 0; t < 256; t++) sum += t * hist[t];
  let sumB = 0, wB = 0, maxVar = -1, thr = 127;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = N - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB, mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > maxVar) { maxVar = between; thr = t; }
  }

  // Ink = the minority class (letters cover less area than the die face).
  let darkCount = 0;
  for (let j = 0; j < N; j++) if (gray[j] < thr) darkCount++;
  const textIsDark = darkCount <= N - darkCount;

  // Bounding box of ink, ignoring a thin outer ring (tray/edge bleed).
  const ring = Math.round(W * 0.04);
  let minX = W, minY = W, maxX = -1, maxY = -1, inkCount = 0;
  const ink = new Uint8Array(N);
  for (let y = ring; y < W - ring; y++) {
    for (let x = ring; x < W - ring; x++) {
      const on = (gray[y * W + x] < thr) === textIsDark;
      if (on) {
        ink[y * W + x] = 1; inkCount++;
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
      }
    }
  }

  const outc = document.createElement('canvas');
  outc.width = out; outc.height = out;
  const octx = outc.getContext('2d');
  octx.fillStyle = '#fff';
  octx.fillRect(0, 0, out, out);

  const frac = inkCount / N;
  if (maxX < minX || frac < 0.004 || frac > 0.85) return outc; // nothing usable

  // Copy the isolated glyph to its own canvas, then draw it centred at ~70%.
  const gw = maxX - minX + 1, gh = maxY - minY + 1;
  const glyph = document.createElement('canvas');
  glyph.width = gw; glyph.height = gh;
  const gim = glyph.getContext('2d').createImageData(gw, gh);
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      const v = ink[(minY + y) * W + (minX + x)] ? 0 : 255;
      const k = (y * gw + x) * 4;
      gim.data[k] = gim.data[k + 1] = gim.data[k + 2] = v; gim.data[k + 3] = 255;
    }
  }
  glyph.getContext('2d').putImageData(gim, 0, 0);

  const scale = (out * 0.7) / Math.max(gw, gh);
  const dw = gw * scale, dh = gh * scale;
  octx.imageSmoothingEnabled = true;
  octx.drawImage(glyph, 0, 0, gw, gh, (out - dw) / 2, (out - dh) / 2, dw, dh);
  return outc;
}
