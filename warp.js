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

// Best-guess the board quad by finding the DICE directly. Boggle dice are bright
// and near-white; the tray gridlines between them are dark, so tray+dice never
// form one blob — the old "largest contrasting region" approach found a single
// die and gave up. Instead: threshold for bright, low-saturation pixels (the die
// faces), keep every sizeable blob, drop stray outliers (table glare, a stray
// die outside the tray), and take the four extreme corners of their union.
// Returns [TL,TR,BR,BL] in full-res px, or null if no convincing grid is found.
export function autoDetectQuad(img) {
  const scale = 320 / Math.max(img.width, img.height);
  const w = Math.max(1, Math.round(img.width * scale));
  const h = Math.max(1, Math.round(img.height * scale));
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  const ctx = cv.getContext('2d');
  ctx.drawImage(img, 0, 0, w, h);
  const px = ctx.getImageData(0, 0, w, h).data;
  const N = w * h;

  // Brightness (max channel) and saturation for every pixel.
  const val = new Uint8Array(N);
  const sat = new Float32Array(N);
  const hist = new Array(256).fill(0);
  for (let j = 0; j < N; j++) {
    const k = j * 4, r = px[k], g = px[k + 1], b = px[k + 2];
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    val[j] = mx; sat[j] = mx > 0 ? (mx - mn) / mx : 0;
    hist[mx]++;
  }
  // Otsu on brightness to separate the bright dice from the darker background.
  let sumAll = 0;
  for (let t = 0; t < 256; t++) sumAll += t * hist[t];
  let sumB = 0, wB = 0, maxVar = -1, vThr = 127;
  for (let t = 0; t < 256; t++) {
    wB += hist[t]; if (!wB) continue;
    const wF = N - wB; if (!wF) break;
    sumB += t * hist[t];
    const mB = sumB / wB, mF = (sumAll - sumB) / wF;
    const bt = wB * wF * (mB - mF) * (mB - mF);
    if (bt > maxVar) { maxVar = bt; vThr = t; }
  }
  // Dice = bright AND low-saturation (white-ish), which excludes coloured wood
  // and the coloured tray even when they are locally bright.
  const mask = new Uint8Array(N);
  for (let j = 0; j < N; j++) mask[j] = (val[j] > vThr && sat[j] < 0.30) ? 1 : 0;

  // Connected components (8-connectivity so a single die stays whole).
  const lab = new Int32Array(N);
  const sizes = [0]; // 1-indexed; sizes[0] unused
  const stack = [];
  let nComp = 0;
  for (let s = 0; s < N; s++) {
    if (!mask[s] || lab[s]) continue;
    nComp++; lab[s] = nComp; let sz = 0;
    stack.length = 0; stack.push(s);
    while (stack.length) {
      const p = stack.pop(); sz++;
      const x = p % w, y = (p / w) | 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue;
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          const q = ny * w + nx;
          if (mask[q] && !lab[q]) { lab[q] = nComp; stack.push(q); }
        }
      }
    }
    sizes.push(sz);
  }
  if (nComp === 0) return null;

  const median = (arr) => {
    const a = arr.slice().sort((x, y) => x - y), m = a.length >> 1;
    return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
  };

  // Candidate blobs above a speckle floor.
  const cand = [];
  for (let i = 1; i <= nComp; i++) if (sizes[i] >= Math.max(40, 0.001 * N)) cand.push(i);
  if (cand.length < 4) return null;
  // Dice are near-uniform in size, so the median candidate IS a die. Keep only
  // die-sized blobs — table glare/reflections are much larger, letter-glints and
  // texture much smaller — so an oversized bright patch can't stretch the quad.
  const medSize = median(cand.map((i) => sizes[i]));
  const keep = new Uint8Array(nComp + 1);
  for (const i of cand) if (sizes[i] >= 0.35 * medSize && sizes[i] <= 2.5 * medSize) keep[i] = 1;
  // Centroids of kept components.
  const cents = []; // { id, cx, cy }
  const sumX = new Float64Array(nComp + 1), sumY = new Float64Array(nComp + 1);
  for (let p = 0; p < N; p++) {
    const i = lab[p]; if (!i || !keep[i]) continue;
    sumX[i] += p % w; sumY[i] += (p / w) | 0;
  }
  for (let i = 1; i <= nComp; i++) {
    if (!keep[i]) continue;
    cents.push({ id: i, cx: sumX[i] / sizes[i], cy: sumY[i] / sizes[i] });
  }
  if (cents.length < 4) return null; // not enough dice to trust a grid

  // Reject spatial outliers: any kept blob far from the median die centroid
  // (a stray die on the table, a die-sized reflection) shouldn't stretch the quad.
  const mcx = median(cents.map((c) => c.cx)), mcy = median(cents.map((c) => c.cy));
  const dists = cents.map((c) => Math.hypot(c.cx - mcx, c.cy - mcy));
  const medD = median(dists);
  const mad = median(dists.map((d) => Math.abs(d - medD)));
  const p75 = dists.slice().sort((a, b) => a - b)[Math.min(dists.length - 1, Math.floor(dists.length * 0.75))];
  const keepR = Math.max(medD + 2.5 * mad * 1.4826, p75 * 1.5);
  const keptIds = new Set(cents.filter((c, i) => dists[i] <= keepR).map((c) => c.id));
  if (keptIds.size < 4) return null;

  // Four extreme points over the union of kept dice → quad corners.
  let tl, tr, br, bl, tlv = 1e9, brv = -1e9, trv = -1e9, blv = 1e9, count = 0;
  for (let p = 0; p < N; p++) {
    if (!keptIds.has(lab[p])) continue;
    count++;
    const x = p % w, y = (p / w) | 0, s = x + y, d = x - y;
    if (s < tlv) { tlv = s; tl = { x, y }; }
    if (s > brv) { brv = s; br = { x, y }; }
    if (d > trv) { trv = d; tr = { x, y }; }
    if (d < blv) { blv = d; bl = { x, y }; }
  }
  if (count < N * 0.05) return null; // too little dice area to be a board
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
