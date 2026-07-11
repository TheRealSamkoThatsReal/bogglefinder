// OCR wrapper around Tesseract.js (lazy-loaded from CDN).
import { rectifyBoard, extractCell } from './warp.js';

const TESS_URL = 'https://cdn.jsdelivr.net/npm/tesseract.js@5.1.1/dist/tesseract.min.js';
let workerPromise = null;

function loadTesseract() {
  if (window.Tesseract) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = TESS_URL;
    s.onload = resolve;
    s.onerror = () => reject(new Error('Failed to load Tesseract.js (are you online?)'));
    document.head.appendChild(s);
  });
}

async function getWorker() {
  if (!workerPromise) {
    workerPromise = (async () => {
      await loadTesseract();
      const worker = await window.Tesseract.createWorker('eng');
      await worker.setParameters({
        tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        tessedit_pageseg_mode: '10', // treat image as a single character
      });
      return worker;
    })();
  }
  return workerPromise;
}

const CELL_PX = 128;

// Runs OCR on every cell. onProgress(done, total) is called as it goes.
// Returns array (row-major) of { char, confidence }.
export async function scanBoard(img, quad, rows, cols, onProgress) {
  const worker = await getWorker();
  const rect = rectifyBoard(img, quad, rows, cols, CELL_PX);
  const results = [];
  const total = rows * cols;
  let done = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cell = extractCell(rect, rows, cols, r, c, CELL_PX);
      // Dice land at random orientations, so read each cell at all four
      // 90° rotations and keep the most confident letter.
      let best = { char: '', confidence: -1, rotation: 0 };
      for (const deg of [0, 90, 180, 270]) {
        const { data } = await worker.recognize(rotateCanvas(cell, deg));
        let ch = (data.text || '').replace(/[^A-Za-z]/g, '').toUpperCase();
        ch = ch ? ch[0] : '';
        const conf = ch ? data.confidence : -1;
        if (conf > best.confidence) best = { char: ch, confidence: conf, rotation: deg };
      }
      let char = best.char;
      if (char === 'Q') char = 'Qu';
      results.push({
        char,
        confidence: char ? Math.round(Math.max(0, best.confidence)) : 0,
        rotation: best.rotation,
      });
      onProgress && onProgress(++done, total);
    }
  }
  return results;
}

// Rotate a square canvas by a multiple of 90°, on a white background.
function rotateCanvas(src, deg) {
  if (deg % 360 === 0) return src;
  const cv = document.createElement('canvas');
  cv.width = src.width; cv.height = src.height;
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.translate(cv.width / 2, cv.height / 2);
  ctx.rotate((deg * Math.PI) / 180);
  ctx.drawImage(src, -src.width / 2, -src.height / 2);
  return cv;
}

export { CELL_PX };
