// Python's random.Random and the Snake game, reproduced exactly in the browser,
// so a human can play the very board the models played.

// ---- MT19937, seeded the way CPython seeds from an int ----
function PyRandom(seed) {
  const N = 624, mt = new Uint32Array(N);
  let mti = N + 1;
  const mul = (a, b) => Math.imul(a, b) >>> 0;
  function initGenrand(s) {
    mt[0] = s >>> 0;
    for (mti = 1; mti < N; mti++) {
      const p = mt[mti - 1] ^ (mt[mti - 1] >>> 30);
      mt[mti] = (mul(1812433253, p) + mti) >>> 0;
    }
  }
  function initByArray(key) {
    initGenrand(19650218);
    let i = 1, j = 0;
    for (let k = Math.max(N, key.length); k; k--) {
      const p = mt[i - 1] ^ (mt[i - 1] >>> 30);
      mt[i] = ((mt[i] ^ mul(p, 1664525)) + key[j] + j) >>> 0;
      i++; j++;
      if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
      if (j >= key.length) j = 0;
    }
    for (let k = N - 1; k; k--) {
      const p = mt[i - 1] ^ (mt[i - 1] >>> 30);
      mt[i] = ((mt[i] ^ mul(p, 1566083941)) - i) >>> 0;
      i++;
      if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
    }
    mt[0] = 0x80000000;
  }
  function genrand() {
    let y;
    if (mti >= N) {
      let kk = 0;
      for (; kk < N - 397; kk++) {
        y = (mt[kk] & 0x80000000) | (mt[kk + 1] & 0x7fffffff);
        mt[kk] = mt[kk + 397] ^ (y >>> 1) ^ ((y & 1) ? 0x9908b0df : 0);
      }
      for (; kk < N - 1; kk++) {
        y = (mt[kk] & 0x80000000) | (mt[kk + 1] & 0x7fffffff);
        mt[kk] = mt[kk + (397 - N)] ^ (y >>> 1) ^ ((y & 1) ? 0x9908b0df : 0);
      }
      y = (mt[N - 1] & 0x80000000) | (mt[0] & 0x7fffffff);
      mt[N - 1] = mt[396] ^ (y >>> 1) ^ ((y & 1) ? 0x9908b0df : 0);
      mti = 0;
    }
    y = mt[mti++];
    y ^= y >>> 11;
    y = (y ^ ((y << 7) & 0x9d2c5680)) >>> 0;
    y = (y ^ ((y << 15) & 0xefc60000)) >>> 0;
    y ^= y >>> 18;
    return y >>> 0;
  }
  // Seed: abs(int) split into 32-bit words; zero becomes [0].
  let n = Math.abs(seed), key = [];
  while (n > 0) { key.push(n % 4294967296); n = Math.floor(n / 4294967296); }
  if (!key.length) key = [0];
  initByArray(key);

  const getrandbits = k => (k === 0 ? 0 : genrand() >>> (32 - k));
  const randbelow = n => {               // Python 3.14 _randbelow_with_getrandbits
    const k = Math.floor(Math.log2(n)) + 1;
    let r = getrandbits(k);
    while (r >= n) r = getrandbits(k);
    return r;
  };
  return { choice: arr => arr[randbelow(arr.length)], randbelow };
}

// ---- the game, mirroring sysone/snake/game.py ----
const DIRS = { UP: [0, -1], DOWN: [0, 1], LEFT: [-1, 0], RIGHT: [1, 0] };
const OPP = { UP: "DOWN", DOWN: "UP", LEFT: "RIGHT", RIGHT: "LEFT" };

function SnakeGame(seed, W = 10, H = 10) {
  const rng = PyRandom(seed);
  const g = { W, H, direction: "RIGHT", score: 0, steps: 0, alive: true, food: [0, 0] };
  const cx = Math.floor(W / 2), cy = Math.floor(H / 2);
  g.body = [[cx, cy], [cx - 1, cy], [cx - 2, cy]];
  const has = (cells, c) => cells.some(b => b[0] === c[0] && b[1] === c[1]);
  g.placeFood = () => {
    const free = [];
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
      if (!has(g.body, [x, y])) free.push([x, y]);
    g.food = free.length ? rng.choice(free) : [-1, -1];
  };
  g.step = move => {
    if (!(move in DIRS) || move === OPP[g.direction]) move = g.direction;
    const [dx, dy] = DIRS[move], [hx, hy] = g.body[0];
    const nxt = [hx + dx, hy + dy];
    const eats = nxt[0] === g.food[0] && nxt[1] === g.food[1];
    const fatal = nxt[0] < 0 || nxt[0] >= W || nxt[1] < 0 || nxt[1] >= H ||
                  has(g.body.slice(0, -1), nxt);
    g.direction = move; g.steps++;
    if (fatal) { g.alive = false; return; }
    g.body.unshift(nxt);
    if (eats) { g.score++; g.placeFood(); } else g.body.pop();
  };
  g.placeFood();
  return g;
}

if (typeof module !== "undefined") module.exports = { PyRandom, SnakeGame, DIRS, OPP };
