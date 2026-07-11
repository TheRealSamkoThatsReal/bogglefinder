import { buildTrie, solve, scoreWord } from './solver.js';
import { bilinear, cellCenter, cellColorImages, autoDetectQuad } from './warp.js';
import { scanBoard } from './ocr.js';

// ---------- State ----------
const state = {
  rows: 4, cols: 4,
  img: null,            // ImageBitmap of the scanned photo (EXIF-corrected)
  quad: null,           // [TL,TR,BR,BL] in img pixel coords
  cells: [],            // grid letters, row-major: strings like 'a', 'qu', 'in'
  confidences: [],      // per-cell OCR confidence (0-100), parallel to cells
  cellImages: [],       // per-cell cropped die image (data URL) or null
  boardBg: null,        // canvas/bitmap drawn behind the path overlay
  boardQuad: null,      // quad in boardBg pixel coords used for overlay
  results: [],          // [{ word, path, len, score }]
  trie: null,
  selectedWord: null,
};

const $ = (s) => document.querySelector(s);
const screens = {
  capture: $('#screen-capture'),
  align: $('#screen-align'),
  edit: $('#screen-edit'),
  results: $('#screen-results'),
};
function show(name) {
  for (const k in screens) screens[k].classList.toggle('active', k === name);
  window.scrollTo(0, 0);
}

// ---------- Dictionary ----------
const dictReady = fetch('./words.txt')
  .then((r) => r.text())
  .then((txt) => {
    const words = txt.split('\n').filter(Boolean);
    state.trie = buildTrie(words);
    $('#dict-count').textContent = words.length.toLocaleString();
  })
  .catch(() => { $('#dict-count').textContent = '⚠ failed to load'; });

// ---------- Capture ----------
$('#size-select').addEventListener('change', (e) => {
  const [r, c] = e.target.value.split('x').map(Number);
  state.rows = r; state.cols = c;
});

async function loadImageFile(file) {
  const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
  state.img = bitmap;
  state.quad = defaultQuad(bitmap);
  const auto = tryAutoDetect(); // snap corners to the tray if we can find it
  show('align');
  setupAlign();
  const prog = $('#ocr-progress');
  if (auto) {
    prog.hidden = false;
    prog.textContent = 'Corners auto-detected — drag any dot to fine-tune.';
    setTimeout(() => { prog.hidden = true; }, 2800);
  }
}

function defaultQuad(bitmap) {
  const w = bitmap.width, h = bitmap.height, mx = w * 0.1, my = h * 0.1;
  return [{ x: mx, y: my }, { x: w - mx, y: my }, { x: w - mx, y: h - my }, { x: mx, y: h - my }];
}

function tryAutoDetect() {
  try {
    const q = autoDetectQuad(state.img);
    if (q) { state.quad = q; return true; }
  } catch (e) { /* fall back to the inset default */ }
  return false;
}

$('#file-camera').addEventListener('change', (e) => { if (e.target.files[0]) loadImageFile(e.target.files[0]); });
$('#file-upload').addEventListener('change', (e) => { if (e.target.files[0]) loadImageFile(e.target.files[0]); });
$('#btn-manual').addEventListener('click', () => {
  state.img = null; state.quad = null;
  state.cells = new Array(state.rows * state.cols).fill('');
  state.confidences = new Array(state.rows * state.cols).fill(100);
  state.cellImages = new Array(state.rows * state.cols).fill(null);
  buildEditGrid();
  show('edit');
});

// ---------- Align (drag 4 corners) ----------
const alignCanvas = $('#align-canvas');
const actx = alignCanvas.getContext('2d');
let alignScale = 1;
let dragHandle = -1;
const HANDLE_R = 16;

function fitCanvas(canvas, imgW, imgH, maxW) {
  const scale = Math.min(maxW / imgW, 1);
  canvas.width = Math.round(imgW * scale);
  canvas.height = Math.round(imgH * scale);
  return scale;
}

function setupAlign() {
  const maxW = Math.min(alignCanvas.parentElement.clientWidth || 720, 720);
  alignScale = fitCanvas(alignCanvas, state.img.width, state.img.height, maxW);
  drawAlign();
}

function drawAlign() {
  if (!state.img) return;
  actx.clearRect(0, 0, alignCanvas.width, alignCanvas.height);
  actx.drawImage(state.img, 0, 0, alignCanvas.width, alignCanvas.height);
  const q = state.quad.map((p) => ({ x: p.x * alignScale, y: p.y * alignScale }));

  // Grid preview.
  actx.strokeStyle = 'rgba(94,234,212,0.9)';
  actx.lineWidth = 2;
  actx.beginPath();
  actx.moveTo(q[0].x, q[0].y);
  for (let i = 1; i < 4; i++) actx.lineTo(q[i].x, q[i].y);
  actx.closePath();
  actx.stroke();

  actx.strokeStyle = 'rgba(94,234,212,0.35)';
  actx.lineWidth = 1;
  for (let i = 1; i < state.cols; i++) {
    const a = bilinear(q, i / state.cols, 0), b = bilinear(q, i / state.cols, 1);
    actx.beginPath(); actx.moveTo(a.x, a.y); actx.lineTo(b.x, b.y); actx.stroke();
  }
  for (let i = 1; i < state.rows; i++) {
    const a = bilinear(q, 0, i / state.rows), b = bilinear(q, 1, i / state.rows);
    actx.beginPath(); actx.moveTo(a.x, a.y); actx.lineTo(b.x, b.y); actx.stroke();
  }

  const labels = ['TL', 'TR', 'BR', 'BL'];
  for (let i = 0; i < 4; i++) {
    actx.beginPath();
    actx.arc(q[i].x, q[i].y, HANDLE_R, 0, Math.PI * 2);
    actx.fillStyle = i === dragHandle ? '#f0abfc' : '#5eead4';
    actx.fill();
    actx.fillStyle = '#07211d';
    actx.font = 'bold 11px system-ui';
    actx.textAlign = 'center'; actx.textBaseline = 'middle';
    actx.fillText(labels[i], q[i].x, q[i].y);
  }
}

function canvasPoint(evt) {
  const rect = alignCanvas.getBoundingClientRect();
  const t = evt.touches ? evt.touches[0] : evt;
  return {
    x: (t.clientX - rect.left) * (alignCanvas.width / rect.width),
    y: (t.clientY - rect.top) * (alignCanvas.height / rect.height),
  };
}
function startDrag(evt) {
  const p = canvasPoint(evt);
  for (let i = 0; i < 4; i++) {
    const h = { x: state.quad[i].x * alignScale, y: state.quad[i].y * alignScale };
    if (Math.hypot(h.x - p.x, h.y - p.y) < HANDLE_R * 1.8) { dragHandle = i; evt.preventDefault(); return; }
  }
}
function moveDrag(evt) {
  if (dragHandle < 0) return;
  evt.preventDefault();
  const p = canvasPoint(evt);
  state.quad[dragHandle] = {
    x: Math.max(0, Math.min(state.img.width, p.x / alignScale)),
    y: Math.max(0, Math.min(state.img.height, p.y / alignScale)),
  };
  drawAlign();
}
function endDrag() { if (dragHandle < 0) return; dragHandle = -1; drawAlign(); }
alignCanvas.addEventListener('mousedown', startDrag);
window.addEventListener('mousemove', moveDrag);
window.addEventListener('mouseup', endDrag);
alignCanvas.addEventListener('touchstart', startDrag, { passive: false });
alignCanvas.addEventListener('touchmove', moveDrag, { passive: false });
alignCanvas.addEventListener('touchend', endDrag);
window.addEventListener('resize', () => { if (state.img && screens.align.classList.contains('active')) setupAlign(); });

$('#btn-back-capture').addEventListener('click', () => show('capture'));
$('#btn-autofit').addEventListener('click', () => {
  const prog = $('#ocr-progress');
  prog.hidden = false;
  if (tryAutoDetect()) {
    drawAlign();
    prog.textContent = 'Corners auto-detected — drag to fine-tune.';
  } else {
    prog.textContent = "Couldn't find the tray automatically — drag the corners.";
  }
  setTimeout(() => { prog.hidden = true; }, 2800);
});
$('#btn-scan').addEventListener('click', runOcr);
$('#btn-skip-ocr').addEventListener('click', () => {
  const n = state.rows * state.cols;
  state.cells = new Array(n).fill('');
  state.confidences = new Array(n).fill(100);
  // No OCR, but still show each die crop so letters can be read off the photo.
  state.cellImages = (state.img && state.quad)
    ? cellColorImages(state.img, state.quad, state.rows, state.cols)
    : new Array(n).fill(null);
  buildEditGrid();
  show('edit');
});

async function runOcr() {
  const btn = $('#btn-scan');
  btn.disabled = true;
  const prog = $('#ocr-progress');
  prog.hidden = false;
  prog.textContent = 'Loading OCR engine…';
  try {
    const res = await scanBoard(state.img, state.quad, state.rows, state.cols,
      (done, total) => { prog.textContent = `Reading letters… ${done}/${total}`; });
    state.cells = res.map((r) => (r.char ? r.char.toLowerCase() : ''));
    state.confidences = res.map((r) => r.confidence);
    state.cellImages = res.map((r) => r.image || null);
    buildEditGrid();
    show('edit');
  } catch (err) {
    prog.textContent = '⚠ ' + err.message + ' — you can enter letters manually instead.';
    setTimeout(() => {
      const n = state.rows * state.cols;
      state.cells = new Array(n).fill('');
      state.confidences = new Array(n).fill(100);
      state.cellImages = (state.img && state.quad)
        ? cellColorImages(state.img, state.quad, state.rows, state.cols)
        : new Array(n).fill(null);
      buildEditGrid();
      show('edit');
    }, 1500);
  } finally {
    btn.disabled = false;
    prog.hidden = true;
  }
}

// ---------- Edit grid ----------
// Recognised multi-letter Boggle cubes: Qu (classic) plus the Super Big Boggle
// blocks. The solver treats each as one cell worth two letters.
const BLOCKS = new Set(['QU', 'IN', 'AN', 'TH', 'HE', 'ER']);

function buildEditGrid() {
  const grid = $('#edit-grid');
  grid.style.setProperty('--cols', state.cols);
  const hasImages = state.cellImages && state.cellImages.some(Boolean);
  grid.classList.toggle('with-images', hasImages);
  grid.innerHTML = '';
  for (let i = 0; i < state.rows * state.cols; i++) {
    const cell = document.createElement('label');
    cell.className = 'edit-cell';
    if (state.cellImages && state.cellImages[i]) {
      const im = document.createElement('img');
      im.src = state.cellImages[i];
      im.alt = '';
      cell.appendChild(im);
    }
    const inp = document.createElement('input');
    inp.className = 'cell-input';
    inp.maxLength = 2;
    inp.value = state.cells[i] ? state.cells[i].toUpperCase() : '';
    inp.dataset.i = i;
    inp.autocapitalize = 'characters';
    inp.inputMode = 'text';
    const conf = state.confidences[i];
    if (state.cells[i] && conf < 65) cell.classList.add('low-conf');
    inp.addEventListener('input', onCellInput);
    inp.addEventListener('focus', () => inp.select());
    cell.appendChild(inp);
    grid.appendChild(cell);
  }
  const first = grid.querySelector('input');
  if (first) first.focus();
}

function onCellInput(e) {
  const inp = e.target;
  let v = inp.value.replace(/[^A-Za-z]/g, '').toUpperCase();
  if (v.length > 2) v = v.slice(0, 2);
  if (v.length === 2 && !BLOCKS.has(v)) v = v.slice(1); // keep the newest key
  if (v === 'Q') v = 'QU'; // lone Q is always Qu
  inp.value = v;
  inp.closest('.edit-cell')?.classList.remove('low-conf');
  const i = +inp.dataset.i;
  state.cells[i] = v.toLowerCase();
  // Auto-advance, but wait on a single letter that might still become a block.
  const mayExtend = v.length === 1 && [...BLOCKS].some((b) => b[0] === v);
  if (v && !mayExtend) {
    const next = document.querySelector(`#edit-grid input[data-i="${i + 1}"]`);
    if (next) next.focus();
  }
}

$('#btn-back-align').addEventListener('click', () => show(state.img ? 'align' : 'capture'));
$('#btn-solve').addEventListener('click', doSolve);

// ---------- Solve ----------
async function doSolve() {
  await dictReady;
  if (!state.trie) { alert('Dictionary is not loaded yet — please wait a moment.'); return; }
  const filled = state.cells.filter((c) => c).length;
  if (filled < state.rows * state.cols) {
    if (!confirm(`${state.rows * state.cols - filled} cell(s) are empty and will be skipped. Continue?`)) return;
  }
  prepareBoardBackground();
  const grid = { rows: state.rows, cols: state.cols, cells: state.cells.map((l) => ({ letters: l })) };
  const map = solve(grid, state.trie, { minLen: 3 });
  state.results = [...map.entries()]
    .map(([word, path]) => ({ word, path, len: word.length, score: scoreWord(word.length) }))
    .sort((a, b) => b.len - a.len || a.word.localeCompare(b.word));
  show('results');
  renderResults();
}

// Build the background image + quad used for the path overlay.
function prepareBoardBackground() {
  if (state.img && state.quad) {
    state.boardBg = state.img;
    state.boardQuad = state.quad;
  } else {
    // Synthesise a board image from the letters (manual-entry mode).
    const cp = 100;
    const cv = document.createElement('canvas');
    cv.width = state.cols * cp; cv.height = state.rows * cp;
    const ctx = cv.getContext('2d');
    ctx.fillStyle = '#0b1220'; ctx.fillRect(0, 0, cv.width, cv.height);
    for (let r = 0; r < state.rows; r++) {
      for (let c = 0; c < state.cols; c++) {
        const x = c * cp, y = r * cp, i = r * state.cols + c;
        ctx.fillStyle = '#16233b';
        ctx.fillRect(x + 6, y + 6, cp - 12, cp - 12);
        ctx.fillStyle = '#e8eef7';
        ctx.font = 'bold 46px system-ui';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        const t = state.cells[i] ? state.cells[i].toUpperCase() : '';
        ctx.fillText(t, x + cp / 2, y + cp / 2 + 2);
      }
    }
    state.boardBg = cv;
    state.boardQuad = [
      { x: 0, y: 0 }, { x: cv.width, y: 0 },
      { x: cv.width, y: cv.height }, { x: 0, y: cv.height },
    ];
  }
}

// ---------- Results + overlay ----------
const overlay = $('#overlay-canvas');
const octx = overlay.getContext('2d');
let overlayScale = 1;

function fitOverlay() {
  const bg = state.boardBg;
  const maxW = Math.min(overlay.parentElement.clientWidth || 640, 640);
  overlayScale = Math.min(maxW / bg.width, 1);
  overlay.width = Math.round(bg.width * overlayScale);
  overlay.height = Math.round(bg.height * overlayScale);
  drawOverlay();
}

function drawOverlay() {
  const bg = state.boardBg;
  octx.clearRect(0, 0, overlay.width, overlay.height);
  octx.drawImage(bg, 0, 0, overlay.width, overlay.height);
  const item = state.results.find((r) => r.word === state.selectedWord);
  if (!item) return;

  const pts = item.path.map((idx) => {
    const r = Math.floor(idx / state.cols), c = idx % state.cols;
    const p = cellCenter(state.boardQuad, state.rows, state.cols, r, c);
    return { x: p.x * overlayScale, y: p.y * overlayScale };
  });

  // Path line.
  octx.lineJoin = 'round'; octx.lineCap = 'round';
  octx.strokeStyle = 'rgba(240,171,252,0.95)';
  octx.lineWidth = Math.max(3, overlay.width * 0.012);
  octx.shadowColor = 'rgba(0,0,0,0.6)'; octx.shadowBlur = 6;
  octx.beginPath();
  pts.forEach((p, i) => (i ? octx.lineTo(p.x, p.y) : octx.moveTo(p.x, p.y)));
  octx.stroke();
  octx.shadowBlur = 0;

  // Node markers, first = green, last = red.
  const rad = Math.max(9, overlay.width * 0.02);
  pts.forEach((p, i) => {
    octx.beginPath();
    octx.arc(p.x, p.y, rad, 0, Math.PI * 2);
    octx.fillStyle = i === 0 ? '#34d399' : i === pts.length - 1 ? '#fb7185' : '#f0abfc';
    octx.fill();
    octx.strokeStyle = 'rgba(0,0,0,0.55)'; octx.lineWidth = 2; octx.stroke();
    octx.fillStyle = '#07211d';
    octx.font = `bold ${Math.round(rad)}px system-ui`;
    octx.textAlign = 'center'; octx.textBaseline = 'middle';
    octx.fillText(String(i + 1), p.x, p.y);
  });
}

let minLenFilter = 3;
let searchTerm = '';

function renderResults() {
  const total = state.results.length;
  const totalScore = state.results.reduce((s, r) => s + r.score, 0);
  $('#stat-count').textContent = total.toLocaleString();
  $('#stat-score').textContent = totalScore.toLocaleString();
  $('#stat-longest').textContent = total ? state.results[0].word.toUpperCase() : '—';
  fitOverlay();
  renderList();
}

function renderList() {
  const list = $('#word-list');
  const filtered = state.results.filter((r) =>
    r.len >= minLenFilter && (!searchTerm || r.word.includes(searchTerm)));
  $('#list-shown').textContent = filtered.length.toLocaleString();

  const frag = document.createDocumentFragment();
  let lastLen = null;
  for (const r of filtered) {
    if (r.len !== lastLen) {
      const h = document.createElement('div');
      h.className = 'len-header';
      h.textContent = `${r.len} letters`;
      frag.appendChild(h);
      lastLen = r.len;
    }
    const el = document.createElement('button');
    el.className = 'word-row' + (r.word === state.selectedWord ? ' selected' : '');
    el.dataset.word = r.word;
    el.innerHTML = `<span class="w">${r.word.toUpperCase()}</span><span class="pts">${r.score} pt${r.score !== 1 ? 's' : ''}</span>`;
    frag.appendChild(el);
  }
  list.innerHTML = '';
  list.appendChild(frag);
}

$('#word-list').addEventListener('click', (e) => {
  const row = e.target.closest('.word-row');
  if (!row) return;
  state.selectedWord = row.dataset.word;
  document.querySelectorAll('.word-row.selected').forEach((n) => n.classList.remove('selected'));
  row.classList.add('selected');
  drawOverlay();
  $('#current-word').textContent = state.selectedWord.toUpperCase();
  if (window.matchMedia('(max-width: 860px)').matches) {
    $('#overlay-canvas').scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
});

$('#min-len').addEventListener('change', (e) => { minLenFilter = +e.target.value; renderList(); });
$('#search').addEventListener('input', (e) => { searchTerm = e.target.value.trim().toLowerCase(); renderList(); });
$('#btn-restart').addEventListener('click', () => { state.selectedWord = null; show('capture'); });
$('#btn-edit-again').addEventListener('click', () => { buildEditGrid(); show('edit'); });
window.addEventListener('resize', () => { if (state.boardBg && screens.results.classList.contains('active')) fitOverlay(); });

// ---------- Service worker ----------
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => navigator.serviceWorker.register('./sw.js').catch(() => {}));
}
