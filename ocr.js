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
      const { data } = await worker.recognize(cell);
      let char = (data.text || '').replace(/[^A-Za-z]/g, '').toUpperCase();
      char = char ? char[0] : '';
      if (char === 'Q') char = 'Qu';
      results.push({ char, confidence: char ? Math.round(data.confidence) : 0 });
      onProgress && onProgress(++done, total);
    }
  }
  return results;
}

export { CELL_PX };
