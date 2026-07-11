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

// Extract one cell as a preprocessed high-contrast canvas ready for OCR.
export function extractCell(rectCanvas, rows, cols, r, c, cellPx, out = 140) {
  const inset = Math.round(cellPx * 0.16); // trim grid lines / die edges
  const sx = c * cellPx + inset;
  const sy = r * cellPx + inset;
  const sSize = cellPx - inset * 2;

  const cv = document.createElement('canvas');
  cv.width = out; cv.height = out;
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, out, out);
  const pad = Math.round(out * 0.12);
  ctx.drawImage(rectCanvas, sx, sy, sSize, sSize, pad, pad, out - pad * 2, out - pad * 2);

  // Grayscale + Otsu threshold, orient to dark-text-on-white.
  const im = ctx.getImageData(0, 0, out, out);
  const px = im.data;
  const hist = new Array(256).fill(0);
  const gray = new Uint8Array(out * out);
  for (let i = 0, j = 0; i < px.length; i += 4, j++) {
    const g = (px[i] * 0.299 + px[i + 1] * 0.587 + px[i + 2] * 0.114) | 0;
    gray[j] = g; hist[g]++;
  }
  const total = out * out;
  let sum = 0;
  for (let t = 0; t < 256; t++) sum += t * hist[t];
  let sumB = 0, wB = 0, maxVar = -1, thr = 127;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB, mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > maxVar) { maxVar = between; thr = t; }
  }
  let darkCount = 0;
  for (let j = 0; j < gray.length; j++) if (gray[j] < thr) darkCount++;
  const invert = darkCount > total * 0.5; // if mostly dark, letters are light
  for (let i = 0, j = 0; i < px.length; i += 4, j++) {
    let dark = gray[j] < thr;
    if (invert) dark = !dark;
    const v = dark ? 0 : 255;
    px[i] = px[i + 1] = px[i + 2] = v;
  }
  ctx.putImageData(im, 0, 0);
  return cv;
}
