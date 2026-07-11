// OCR wrapper around Tesseract.js (lazy-loaded from CDN).
import { rectifyBoard, extractCell, cropCell } from './warp.js';

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
        // Mixed case: block cubes print their second letter lowercase (Qu, Th,
        // In, He, Er) and Tesseract only reads them correctly with lowercase allowed.
        tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
        tessedit_pageseg_mode: '8', // treat image as a single word (1-2 letters)
      });
      return worker;
    })();
  }
  return workerPromise;
}

const CELL_PX = 128;
// Multi-letter Boggle cubes the OCR should recognise as one cell.
const BLOCKS = new Set(['QU', 'IN', 'AN', 'TH', 'HE', 'ER']);

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
      // Dice land at random orientations, so read the cell at all four 90°
      // rotations and collect every reading.
      const readings = [];
      for (const deg of [0, 90, 180, 270]) {
        const { data } = await worker.recognize(rotateCanvas(cell, deg));
        const t = (data.text || '').replace(/[^A-Za-z]/g, '').toUpperCase();
        if (t) readings.push({ text: t, confidence: data.confidence, rotation: deg });
      }
      // A rotated block frequently reads as a confident single letter, so prefer
      // any rotation that produced a valid two-letter block; otherwise take the
      // most-confident single letter.
      const blockReads = readings.filter((r) => BLOCKS.has(r.text.slice(0, 2)));
      blockReads.sort((a, b) => b.confidence - a.confidence);
      readings.sort((a, b) => b.confidence - a.confidence);
      const best = blockReads[0]
        ? { text: blockReads[0].text.slice(0, 2), confidence: blockReads[0].confidence, rotation: blockReads[0].rotation }
        : (readings[0]
          ? { text: readings[0].text[0], confidence: readings[0].confidence, rotation: readings[0].rotation }
          : { text: '', confidence: -1, rotation: 0 });

      let char = best.text;
      if (char === 'Q') char = 'QU'; // a lone Q is always Qu
      if (char) char = char[0] + char.slice(1).toLowerCase(); // "Th", "Qu", "A"
      results.push({
        char,
        confidence: char ? Math.round(Math.max(0, best.confidence)) : 0,
        rotation: best.rotation,
        image: cropCell(rect, rows, cols, r, c, CELL_PX).toDataURL('image/png'),
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
