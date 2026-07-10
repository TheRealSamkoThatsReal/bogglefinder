// Boggle solver: trie dictionary + DFS with "Qu" cell support.

export function buildTrie(words) {
  const root = Object.create(null);
  for (const w of words) {
    let node = root;
    for (let i = 0; i < w.length; i++) {
      const ch = w[i];
      node = node[ch] || (node[ch] = Object.create(null));
    }
    node.$ = true; // marks end of a valid word
  }
  return root;
}

export function scoreWord(len) {
  if (len < 3) return 0;
  if (len <= 4) return 1;
  if (len === 5) return 2;
  if (len === 6) return 3;
  if (len === 7) return 5;
  return 11; // 8+
}

// grid: { rows, cols, cells: [{ letters }] } in row-major order.
// `letters` is a lowercase string, usually one char, or "qu" for a Q die.
// Returns Map<word, pathIndices[]> — one representative path per word.
export function solve(grid, root, opts = {}) {
  const { rows, cols, cells } = grid;
  const minLen = opts.minLen ?? 3;
  const n = rows * cols;

  // Precompute the up-to-8 neighbours of every cell.
  const neighbors = new Array(n);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const idx = r * cols + c;
      const list = [];
      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          if (dr === 0 && dc === 0) continue;
          const nr = r + dr, nc = c + dc;
          if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) list.push(nr * cols + nc);
        }
      }
      neighbors[idx] = list;
    }
  }

  const found = new Map();
  const used = new Uint8Array(n);
  const path = [];

  function advance(node, s) {
    let nd = node;
    for (let i = 0; i < s.length; i++) {
      nd = nd[s[i]];
      if (!nd) return null;
    }
    return nd;
  }

  function dfs(idx, node, wordLen, wordStr) {
    const s = cells[idx].letters;
    if (!s) return;
    const nd = advance(node, s);
    if (!nd) return; // no dictionary word continues with this prefix — prune

    used[idx] = 1;
    path.push(idx);
    const newLen = wordLen + s.length;
    const newStr = wordStr + s;

    if (nd.$ && newLen >= minLen && !found.has(newStr)) {
      found.set(newStr, path.slice());
    }
    const nbs = neighbors[idx];
    for (let k = 0; k < nbs.length; k++) {
      const nb = nbs[k];
      if (!used[nb]) dfs(nb, nd, newLen, newStr);
    }
    used[idx] = 0;
    path.pop();
  }

  for (let i = 0; i < n; i++) dfs(i, root, 0, '');
  return found;
}
