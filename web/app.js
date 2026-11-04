// app.js: user interface for the demonstrations. All numerics are in lialign.wasm
// (compiled from src/lialign.c) and in the WebGPU kernel src/lialign.wgsl.

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const GROUP_NAMES = { 1: 'SO', 2: 'SL', 3: 'SO(3,', 4: 'Sp', 5: 'SE' };

if (window.renderMathInElement) renderMathInElement(document.body, { delimiters: [{ left: '$$', right: '$$', display: true }, { left: '$', right: '$', display: false }], throwOnError: false });

// ---------------------------------------------------------------- WebAssembly

const wasm = await (async () => {
  try {
    const res = await fetch('lialign.wasm');
    const { instance } = await WebAssembly.instantiate(await res.arrayBuffer(), {});
    $('wasm-status').textContent = 'WebAssembly: ready';
    $('wasm-status').classList.add('ok');
    return instance.exports;
  } catch (e) {
    $('wasm-status').textContent = 'WebAssembly: failed to load (serve this folder over HTTP)';
    $('wasm-status').classList.add('bad');
    throw e;
  }
})();
const f64 = (ptr, len) => new Float64Array(wasm.memory.buffer, ptr, len);
const f32 = (ptr, len) => new Float32Array(wasm.memory.buffer, ptr, len);
const MATS = () => [wasm.ptr_Gt(), wasm.ptr_G0(), wasm.ptr_Gla(), wasm.ptr_Ggn()];
const readMat = (k, n) => Array.from(f64(MATS()[k], n * n));

// ---------------------------------------------------------------- helpers

function fmt(x, d = 2) {
  if (!isFinite(x)) return '—';
  if (x === 0) return '0';
  const a = Math.abs(x);
  if (a < 1e-15) return '< 1e-15';
  if (a >= 1e-3 && a < 1e4) return x.toPrecision(d + 1);
  return x.toExponential(d);
}
function setupCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}
function bindRange(id, show) {
  const el = $(id), out = $(id + '-out');
  const update = () => { if (out) out.textContent = show(+el.value); };
  el.addEventListener('input', update); update();
  return () => +el.value;
}
function table(el, head, rows) {
  el.innerHTML = '<thead><tr>' + head.map((h, i) => `<th class="${i ? 'num' : ''}">${h}</th>`).join('') + '</tr></thead><tbody>' +
    rows.map((r) => '<tr>' + r.map((c, i) => `<td class="${i ? 'num' : ''}">${c}</td>`).join('') + '</tr>').join('') + '</tbody>';
}
const TESTS_GPU = { run: null, median: null };
const swatch = (v) => `<i class="swatch" style="--c:var(${v})"></i>`;
const pow10 = (v) => Math.pow(10, v);
function axes(ctx, w, h, pad, xr, yr, xlabel, ylabel, opts = {}) {
  ctx.strokeStyle = css('--rule'); ctx.fillStyle = css('--ink-soft'); ctx.lineWidth = 1;
  ctx.font = '11px "IBM Plex Sans", sans-serif';
  const X = (x) => pad.l + (x - xr[0]) / (xr[1] - xr[0]) * (w - pad.l - pad.r);
  const Y = (y) => h - pad.b - (y - yr[0]) / (yr[1] - yr[0]) * (h - pad.t - pad.b);
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b); ctx.stroke();
  const ticks = (r, k) => { const s = Math.pow(10, Math.floor(Math.log10((r[1] - r[0]) / k))); const st = [1, 2, 5, 10].map((m) => m * s).find((m) => (r[1] - r[0]) / m <= k); const out = []; for (let t = Math.ceil(r[0] / st) * st; t <= r[1] + 1e-12; t += st) out.push(+t.toPrecision(10)); return out; };
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (const t of opts.xticks || ticks(xr, 6)) { ctx.fillText(opts.xfmt ? opts.xfmt(t) : t, X(t), h - pad.b + 5); ctx.beginPath(); ctx.moveTo(X(t), h - pad.b); ctx.lineTo(X(t), h - pad.b + 3); ctx.stroke(); }
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (const t of opts.yticks || ticks(yr, 5)) { ctx.fillText(opts.yfmt ? opts.yfmt(t) : t, pad.l - 6, Y(t)); }
  ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
  ctx.fillText(xlabel, (pad.l + w - pad.r) / 2, h - 6);
  ctx.save(); ctx.translate(13, (pad.t + h - pad.b) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(ylabel, 0, 0); ctx.restore();
  return { X, Y };
}

// ---------------------------------------------------------------- shared helpers for the examples

// small seeded random generator for choices made by the interface (directions, pointings)
function mulberry(a) { return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const randomUnit = (rnd) => { const z = 2 * rnd() - 1, t = 2 * Math.PI * rnd(), r = Math.sqrt(1 - z * z); return [r * Math.cos(t), r * Math.sin(t), z]; };
function logLines(cv, xs, series, xr, xlabel, ylabel, opt = {}) {
  const { ctx, w, h } = setupCanvas(cv);
  if (opt.floor !== undefined) series = series.map((s) => ({ ...s, ys: s.ys.map((y) => Math.max(y, opt.floor)) }));
  const all = series.flatMap((s) => s.ys).filter((v) => isFinite(v));
  const yr = [Math.floor(Math.min(...all)) - (opt.floor !== undefined ? 1 : 0), Math.ceil(Math.max(...all))];
  if (yr[0] === yr[1]) yr[1]++;
  const ints = (r) => { const o = []; const st = Math.max(1, Math.ceil((r[1] - r[0]) / 6)); for (let t = Math.ceil(r[0]); t <= r[1]; t += st) o.push(t); return o; };
  const a = axes(ctx, w, h, { l: 50, r: 16, t: 12, b: 38 }, xr.range, yr, xlabel, ylabel, { xticks: xr.ticks, xfmt: xr.fmt, yticks: ints(yr), yfmt: (v) => '1e' + v });
  if (opt.floor !== undefined) {
    ctx.fillStyle = css('--rule'); ctx.globalAlpha = 0.6;
    ctx.fillRect(a.X(xr.range[0]), a.Y(opt.floor), a.X(xr.range[1]) - a.X(xr.range[0]), a.Y(yr[0]) - a.Y(opt.floor));
    ctx.globalAlpha = 1; ctx.fillStyle = css('--ink-soft'); ctx.font = '11px "IBM Plex Sans", sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText(opt.floorLabel || '', a.X(xr.range[0]) + 8, (a.Y(opt.floor) + a.Y(yr[0])) / 2);
  }
  for (const s of series) {
    ctx.strokeStyle = css(s.c); ctx.fillStyle = css(s.c); ctx.lineWidth = 2; ctx.setLineDash(s.dash || []);
    ctx.beginPath(); let first = true;
    s.ys.forEach((y, i) => { if (!isFinite(y)) { first = true; return; } const X = a.X(xs[i]), Y = a.Y(Math.max(yr[0], y)); first ? ctx.moveTo(X, Y) : ctx.lineTo(X, Y); first = false; });
    ctx.stroke(); ctx.setLineDash([]);
    if (s.dots) s.ys.forEach((y, i) => { if (isFinite(y)) { ctx.beginPath(); ctx.arc(a.X(xs[i]), a.Y(y), 2.6, 0, 2 * Math.PI); ctx.fill(); } });
  }
}

// ---------------------------------------------------------------- Lorentz transformations from particle data

{
  const getEta = bindRange('lz-eta', (v) => Math.tanh(v).toFixed(3));
  const getM = bindRange('lz-m', (v) => v);
  const getSigma = bindRange('lz-sigma', (v) => (1000 * pow10(v)).toPrecision(2) + ' MeV');
  let seed = 4, last = null;
  const MP = 0.93827;

  function run() {
    const rnd = mulberry(seed * 7919);
    const m = getM();
    wasm.seed(seed);
    wasm.make_lorentz(getEta(), Math.acos(2 * rnd() - 1), 2 * Math.PI * rnd(), 2 * Math.PI * rnd());
    wasm.gen_particles(m, pow10(getSigma()), 1.0);
    if (!wasm.align(4, m, 3, 30)) return;
    wasm.diagnostics(4, m, 3);
    const info = Array.from(f64(wasm.ptr_info(), 11));
    const mats = [1, 2, 3].map((k) => readMat(k, 4));
    const d = randomUnit(rnd);
    const massErr = (G, p) => {
      const x = [Math.hypot(MP, p), p * d[0], p * d[1], p * d[2]];
      const y = [0, 1, 2, 3].map((i) => G[4 * i] * x[0] + G[4 * i + 1] * x[1] + G[4 * i + 2] * x[2] + G[4 * i + 3] * x[3]);
      const m2 = y[0] ** 2 - y[1] ** 2 - y[2] ** 2 - y[3] ** 2;
      return Math.abs(Math.sign(m2) * Math.sqrt(Math.abs(m2)) - MP);
    };
    const ps = []; for (let e = -1; e <= 3.001; e += 0.05) ps.push(e);
    last = { ps, ys: mats.map((G) => ps.map((e) => Math.log10(Math.max(1e-16, massErr(G, pow10(e)))))) };
    const names = ['Least squares', 'Lie algebra', `Gauss-Newton (${info[0]} steps)`];
    const mev = (v) => (v < 1e-9 ? '< 1e-6' : (1000 * v).toPrecision(2)) + ' MeV';
    table($('lz-table'), ['Estimate', 'Relative error', 'Defect ‖ΛᵀηΛ − η‖', 'Proton mass error at 1 GeV', 'at 10 GeV', 'at 100 GeV'], [0, 1, 2].map((k) => [
      swatch(['--c-ls', '--c-la', '--c-gn'][k]) + names[k], fmt(info[1 + 3 * k]), fmt(info[3 + 3 * k]), mev(massErr(mats[k], 1)), mev(massErr(mats[k], 10)), mev(massErr(mats[k], 100)),
    ]));
    draw();
  }
  function draw() {
    if (!last) return;
    logLines($('lz-plot'), last.ps, [{ c: '--c-ls', ys: last.ys[0] }, { c: '--c-la', ys: last.ys[1] }, { c: '--c-gn', ys: last.ys[2], dash: [5, 4] }],
      { range: [-1, 3], ticks: [-1, 0, 1, 2, 3], fmt: (v) => ['0.1', '1', '10', '100', '1000'][v + 1] }, 'proton momentum |p| (GeV)', 'mass error (GeV)', { floor: -12, floorLabel: 'below 1e-12 GeV: rounding error only' });
  }
  for (const id of ['lz-eta', 'lz-m', 'lz-sigma']) $(id).addEventListener('input', run);
  $('lz-draw').addEventListener('click', () => { seed++; run(); });
  window.addEventListener('resize', draw);
  run();
}

// ---------------------------------------------------------------- molecule superposition

{
  const getSigma = bindRange('rg-sigma', (v) => v.toFixed(2) + ' Å');
  const getMag = bindRange('rg-mag', (v) => '×' + v);
  const cv = $('rg-canvas');
  let seed = 3, yaw = 0.6, pitch = 0.35, data = null;

  function molecule(kind) {
    let P = [], bonds = [];
    if (kind === 'c60') {
      const f = (1 + Math.sqrt(5)) / 2, base = [];
      for (const s1 of [-1, 1]) for (const s2 of [-1, 1]) base.push([0, s1, 3 * f * s2]);
      for (const s1 of [-1, 1]) for (const s2 of [-1, 1]) for (const s3 of [-1, 1]) { base.push([s1, (2 + f) * s2, 2 * f * s3]); base.push([f * s1, 2 * s2, (2 * f + 1) * s3]); }
      for (const [a, b, c] of base) P.push([a, b, c], [b, c, a], [c, a, b]);
      P = P.map((p) => p.map((v) => v * 0.7));
    } else if (kind === 'helix') {
      for (let i = 0; i < 24; i++) { const t = (i * 100 * Math.PI) / 180; P.push([2.3 * Math.cos(t), 2.3 * Math.sin(t), 1.5 * i]); }
      for (let i = 1; i < 24; i++) bonds.push([i - 1, i]);
    } else if (kind === 'adamantane') {
      const a = 0.89, b = 1.78;
      P = [[a, a, a], [a, -a, -a], [-a, a, -a], [-a, -a, a], [b, 0, 0], [-b, 0, 0], [0, b, 0], [0, -b, 0], [0, 0, b], [0, 0, -b]];
    } else {
      const h = 1.09 / Math.sqrt(3);
      P = [[0, 0, 0], [h, h, h], [h, -h, -h], [-h, h, -h], [-h, -h, h]];
    }
    const c = [0, 1, 2].map((i) => P.reduce((s, p) => s + p[i], 0) / P.length);
    P = P.map((p) => p.map((v, i) => v - c[i]));
    if (!bonds.length) for (let i = 0; i < P.length; i++) for (let j = i + 1; j < P.length; j++) if (Math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1], P[i][2] - P[j][2]) < 1.8) bonds.push([i, j]);
    return { P, bonds };
  }

  function run() {
    const { P, bonds } = molecule($('rg-shape').value), m = P.length;
    const X = f64(wasm.ptr_X(), 4 * m);
    for (let k = 0; k < m; k++) { X[k] = P[k][0]; X[m + k] = P[k][1]; X[2 * m + k] = P[k][2]; X[3 * m + k] = 1; }
    wasm.seed(seed);
    wasm.gen_from_X(4, m, 5, getSigma(), 0.9);
    if (!wasm.align(4, m, 5, 30)) return;
    wasm.diagnostics(4, m, 5);
    const info = Array.from(f64(wasm.ptr_info(), 11));
    const Y = Array.from(f64(wasm.ptr_Y(), 4 * m));
    const pts = [0, 2, 3].map((k) => { wasm.transform(k, 4, m); return Array.from(f64(wasm.ptr_track(), 4 * m)); });
    const rms = (A) => { let s = 0; for (let k = 0; k < 3 * m; k++) s += (A[k] - pts[0][k]) ** 2; return Math.sqrt(s / m); };
    data = { m, Y, pts, bonds };
    table($('rg-table'), ['', 'Lie algebra', 'Gauss-Newton'], [
      ['RMS position error', fmt(rms(pts[1])) + ' Å', fmt(rms(pts[2])) + ' Å'],
      ['Relative error of the motion', fmt(info[4]), fmt(info[7])],
      ['Residual ‖gX − Y‖', fmt(info[5], 4) + ' Å', fmt(info[8], 4) + ' Å'],
    ]);
    draw();
  }

  function draw() {
    if (!data) return;
    const { ctx, w, h } = setupCanvas(cv);
    const { m, Y, pts, bonds } = data;
    let cx = 0, cy = 0, cz = 0, rad = 0;
    for (let k = 0; k < m; k++) { cx += pts[0][k]; cy += pts[0][m + k]; cz += pts[0][2 * m + k]; }
    cx /= m; cy /= m; cz /= m;
    for (let k = 0; k < m; k++) rad = Math.max(rad, Math.hypot(pts[0][k] - cx, pts[0][m + k] - cy, pts[0][2 * m + k] - cz));
    const cyw = Math.cos(yaw), syw = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const s = Math.min(w, h) / (2.6 * rad);
    const proj = (x, y, z) => {
      x -= cx; y -= cy; z -= cz;
      const x1 = cyw * x + syw * y, y1 = -syw * x + cyw * y;
      const y2 = cp * y1 - sp * z, z2 = sp * y1 + cp * z;
      const f = (6 * rad) / (6 * rad + y2);
      return [w / 2 + s * f * x1, h / 2 - s * f * z2];
    };
    const mag = getMag();
    const T = pts[0], amp = (A) => A.map((v, i) => T[i] + mag * (v - T[i]));
    const est = [amp(pts[1]), amp(pts[2])];
    const Ym = Y;
    // depth cue: nearer bonds and atoms are drawn later and more opaque
    const depth = (A, i) => { const x = A[i] - cx, y = A[m + i] - cy, z = A[2 * m + i] - cz; const y1 = -syw * x + cyw * y; return cp * y1 - sp * z; };
    const items = [];
    for (const [e, color, width] of [[0, css('--c-la'), 2.6], [1, css('--c-gn'), 2]]) {
      const A = est[e];
      for (const [i, j] of bonds) items.push({ d: (depth(A, i) + depth(A, j)) / 2, draw: (alpha) => {
        const p = proj(A[i], A[m + i], A[2 * m + i]), q = proj(A[j], A[m + j], A[2 * m + j]);
        ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = 'round';
        ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke(); } });
    }
    for (let k = 0; k < m; k++) items.push({ d: depth(Ym, k), draw: (alpha) => {
      const p = proj(Ym[k], Ym[m + k], Ym[2 * m + k]);
      ctx.globalAlpha = alpha; ctx.fillStyle = css('--ink-soft'); ctx.beginPath(); ctx.arc(p[0], p[1], 3, 0, 2 * Math.PI); ctx.fill(); } });
    const ds = items.map((it) => it.d), dmin = Math.min(...ds), dmax = Math.max(...ds);
    items.sort((p, q) => q.d - p.d).forEach((it) => it.draw(0.35 + 0.65 * (dmax - it.d) / (dmax - dmin + 1e-9)));
    ctx.globalAlpha = 1;
  }

  let drag = null;
  cv.addEventListener('pointerdown', (e) => { drag = [e.clientX, e.clientY]; cv.setPointerCapture(e.pointerId); cv.style.cursor = 'grabbing'; });
  cv.addEventListener('pointermove', (e) => {
    if (!drag) return;
    yaw -= (e.clientX - drag[0]) * 0.01; pitch = Math.max(-1.4, Math.min(1.4, pitch + (e.clientY - drag[1]) * 0.01));
    drag = [e.clientX, e.clientY]; draw();
  });
  cv.addEventListener('pointerup', () => { drag = null; cv.style.cursor = 'grab'; });
  for (const id of ['rg-shape', 'rg-sigma']) $(id).addEventListener('input', run);
  $('rg-mag').addEventListener('input', draw);
  $('rg-draw').addEventListener('click', () => { seed++; run(); });
  window.addEventListener('resize', draw);
  run();
}

// ---------------------------------------------------------------- storage ring

{
  const getQx = bindRange('ar-qx', (v) => v.toFixed(3));
  const getQy = bindRange('ar-qy', (v) => v.toFixed(3));
  const getSigma = bindRange('ar-sigma', (v) => (100 * pow10(v)).toPrecision(2) + '%');
  const getTurns = bindRange('ar-turns', (v) => v);
  let seed = 2, result = null;
  const PHASE_TURNS = 20000, AMP_TURNS = 20000, WIN = 200;

  function tunes(which) {
    wasm.eigenvalues(which, 4);
    const e = f64(wasm.ptr_eig(), 8);
    const q = [], dev = [];
    for (let i = 0; i < 4; i++) {
      dev.push(Math.abs(Math.hypot(e[2 * i], e[2 * i + 1]) - 1));
      if (e[2 * i + 1] > 0) q.push(wasm.angle(e[2 * i + 1], e[2 * i]) / (2 * Math.PI));
    }
    return { q: q.sort(), dev: Math.max(...dev) };
  }

  function run() {
    const turns = getTurns();
    wasm.seed(seed);
    wasm.gen_accel(getQx(), getQy(), 0.05, turns, pow10(getSigma()));
    wasm.align(4, turns, 4, 30);
    wasm.diagnostics(4, turns, 4);
    const info = Array.from(f64(wasm.ptr_info(), 11));
    const tracks = [0, 1, 2].map((k) => {
      wasm.track(k, 4, AMP_TURNS, 1e-3, 0.7e-3, 0, 0);
      const T = f64(wasm.ptr_track(), 4 * AMP_TURNS);
      const x = T.slice(0, AMP_TURNS), px = T.slice(2 * AMP_TURNS, 3 * AMP_TURNS);
      const amp = [];
      for (let s = 0; s < AMP_TURNS; s += WIN) { let a = 0; for (let k = s; k < s + WIN; k++) a = Math.max(a, Math.abs(x[k])); amp.push(a); }
      return { x, px, amp: amp.map((a) => a / amp[0]) };
    });
    // undo the Twiss beta function so the true horizontal motion is close to a circle
    const bx = 12.0, ax = -1.1;
    for (const t of tracks) {
      t.u = t.x.map((v) => v / Math.sqrt(bx) * 1e3);
      t.pu = t.x.map((v, k) => (ax * v / Math.sqrt(bx) + Math.sqrt(bx) * t.px[k]) * 1e3);
    }
    const T = [0, 1, 2].map(tunes);
    result = { tracks };
    table($('ar-table'), ['Map', 'Relative error', 'Tunes Qx, Qy', 'Max ||λ| − 1|', 'Symplectic defect', 'Turns to e-fold'], [0, 1, 2].map((k) => [
      swatch(['--ink', '--c-ls', '--c-la'][k]) + ['True map', 'Least squares', 'Lie algebra'][k],
      k ? fmt(info[1 + 3 * (k - 1)]) : '0',
      T[k].q.slice().reverse().map((v) => v.toFixed(5)).join(', '),
      fmt(T[k].dev),
      k ? fmt(info[3 + 3 * (k - 1)]) : '0',
      T[k].dev > 1e-12 ? Math.round(1 / T[k].dev).toLocaleString() : '∞',
    ]));
    draw();
  }

  function draw() {
    if (!result) return;
    const colors = [css('--ink'), css('--c-ls'), css('--c-la')];
    {
      // three small panels with a shared scale: true map, least squares, Lie algebra
      const { ctx, w, h } = setupCanvas($('ar-phase'));
      let r = 0;
      for (let k = 0; k < 500; k++) r = Math.max(r, Math.abs(result.tracks[0].u[k]), Math.abs(result.tracks[0].pu[k]));
      r *= 1.15;
      const names = ['True map', 'Least squares', 'Lie algebra'], gap = 8, pw = (w - 2 * gap) / 3, top = 24;
      const side = Math.min(pw, h - top - 8);
      for (let k = 0; k < 3; k++) {
        const x0 = k * (pw + gap) + (pw - side) / 2, y0 = top;
        ctx.strokeStyle = css('--rule'); ctx.lineWidth = 1; ctx.strokeRect(x0, y0, side, side);
        ctx.beginPath(); ctx.moveTo(x0 + side / 2, y0); ctx.lineTo(x0 + side / 2, y0 + side); ctx.moveTo(x0, y0 + side / 2); ctx.lineTo(x0 + side, y0 + side / 2); ctx.stroke();
        ctx.fillStyle = colors[k]; ctx.font = '600 12px "IBM Plex Sans", sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
        ctx.fillText(names[k], x0 + side / 2, y0 - 6);
        const t = result.tracks[k];
        ctx.globalAlpha = 0.35;
        for (let i = 0; i < PHASE_TURNS; i++) {
          const X = x0 + side / 2 + (t.u[i] / r) * side / 2, Y = y0 + side / 2 - (t.pu[i] / r) * side / 2;
          if (X > x0 && X < x0 + side && Y > y0 && Y < y0 + side) ctx.fillRect(X - 0.6, Y - 0.6, 1.2, 1.2);
        }
        ctx.globalAlpha = 1;
      }
    }
    {
      const { ctx, w, h } = setupCanvas($('ar-amp'));
      let lo = 1, hi = 1;
      for (const t of result.tracks) for (const v of t.amp) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      lo = Math.max(lo, 1e-6); hi = Math.min(hi, 1e6);
      const yl = [Math.log10(lo) - 0.05, Math.log10(hi) + 0.05];
      const a = axes(ctx, w, h, { l: 50, r: 26, t: 12, b: 38 }, [0, AMP_TURNS], yl, 'turn', 'amplitude', { yfmt: (v) => fmt(pow10(v), 1), xfmt: (v) => v.toLocaleString() });
      for (const k of [0, 1, 2]) {
        ctx.strokeStyle = colors[k]; ctx.lineWidth = k === 0 ? 1.5 : 2.2; ctx.setLineDash(k === 0 ? [5, 4] : []);
        ctx.beginPath();
        result.tracks[k].amp.forEach((v, i) => { const X = a.X(i * WIN), Y = a.Y(Math.log10(Math.max(lo, Math.min(hi, v)))); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }
  }
  for (const id of ['ar-qx', 'ar-qy', 'ar-sigma', 'ar-turns']) $(id).addEventListener('input', run);
  $('ar-draw').addEventListener('click', () => { seed++; run(); });
  window.addEventListener('resize', draw);
  run();
}

// ---------------------------------------------------------------- quantum gates

{
  const getM = bindRange('qc-m', (v) => v);
  const getSigma = bindRange('qc-sigma', (v) => fmt(pow10(v), 1));
  let seed = 11;
  const names = ['True gate U', 'Least squares', 'Lie algebra', 'Gauss-Newton'];
  const colors = ['--ink', '--c-ls', '--c-la', '--c-gn'];
  function cell(re, im) {
    const mag = Math.hypot(re, im), hue = ((Math.atan2(im, re) * 180) / Math.PI + 360) % 360;
    const deg = Math.round(hue > 180 ? hue - 360 : hue);
    const txt = mag < 0.05 ? mag.toFixed(2) : `${mag.toFixed(2)}<small>${deg}°</small>`;
    return `<div title="|u| = ${mag.toFixed(3)}, arg = ${(hue > 180 ? hue - 360 : hue).toFixed(0)}°" style="background:hsl(${hue} 65% 50% / ${Math.min(1, mag) * 0.6})">${txt}</div>`;
  }
  function run() {
    const m = getM();
    wasm.seed(seed);
    wasm.gen_quantum(+$('qc-gate').value, m, pow10(getSigma()));
    wasm.align(8, 2 * m, 6, 30);
    wasm.diagnostics(8, 2 * m, 6);
    const info = Array.from(f64(wasm.ptr_info(), 11));
    $('qc-matrices').innerHTML = [0, 1, 2, 3].map((k) => {
      const G = readMat(k, 8), cells = [];
      for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) cells.push(cell(G[i * 8 + j], G[(i + 4) * 8 + j]));
      return `<div class="mat"><h4><i style="--c:var(${colors[k]})"></i>${names[k]}</h4><div class="grid complex" style="grid-template-columns:repeat(4,1fr)">${cells.join('')}</div></div>`;
    }).join('');
    const F = [1, 2, 3].map((k) => wasm.gate_fidelity(k, 8));
    table($('qc-table'), ['Estimate', 'Gate infidelity 1 − |tr U†V|/4', 'Relative error', 'Unitarity defect ‖V†V − I‖'], [
      [swatch('--c-ls') + 'Least squares', fmt(1 - F[0]), fmt(info[1]), fmt(info[3])],
      [swatch('--c-la') + 'Lie algebra', fmt(1 - F[1]), fmt(info[4]), fmt(info[6])],
      [swatch('--c-gn') + `Gauss-Newton (${info[0]} steps)`, fmt(1 - F[2]), fmt(info[7]), fmt(info[9])],
    ]);
  }
  for (const id of ['qc-gate', 'qc-m', 'qc-sigma']) $(id).addEventListener('input', run);
  $('qc-draw').addEventListener('click', () => { seed++; run(); });
  run();
}

// ---------------------------------------------------------------- Monte Carlo on the GPU and in WebAssembly

{
  const getTrials = bindRange('mc-trials', (v) => (2 ** v).toLocaleString());
  const getM = bindRange('mc-m', (v) => v);
  const getSigma = bindRange('mc-sigma', (v) => fmt(pow10(v), 1));
  const results = { gpu: null, cpu: null };
  const BINS = [-5, 0.5, 90];

  // WebGPU setup
  let gpu = null;
  try {
    if (!navigator.gpu) throw new Error('navigator.gpu is not available');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('no adapter');
    const device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 2 ** 28) } });
    const code = await (await fetch('src/lialign.wgsl')).text();
    const module = device.createShaderModule({ code });
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    gpu = { device, pipeline };
    const info = adapter.info || {};
    $('gpu-status').textContent = 'WebGPU: ' + ([info.vendor, info.architecture].filter(Boolean).join(' ') || 'ready');
    $('gpu-status').classList.add('ok');
  } catch (e) {
    $('gpu-status').textContent = 'WebGPU: not available in this browser';
    $('gpu-status').classList.add('bad');
    $('mc-gpu').disabled = true; $('mc-sweep-btn').disabled = true;
    $('mc-cap-gpu').textContent = 'WebGPU is not available here. Try a recent Chrome, Edge or Safari.';
    console.warn(e);
  }

  const params = () => {
    const [g, n] = $('mc-group').value.split(',').map(Number);
    return { g, n, trials: 2 ** getTrials(), m: Math.max(getM(), n + 1), sigma: pow10(getSigma()), scale: 0.8 };
  };

  async function runGPU(over = {}) {
    const p = { ...params(), ...over };
    const { device, pipeline } = gpu;
    const u = new ArrayBuffer(32), uu = new Uint32Array(u), uf = new Float32Array(u);
    uu[0] = p.n; uu[1] = p.m; uu[2] = p.g; uu[3] = p.trials; uf[4] = p.sigma; uf[5] = p.scale; uu[6] = p.seed ?? ((Math.random() * 2 ** 32) >>> 0);
    const ubuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(ubuf, 0, u);
    const size = p.trials * 3 * 4;
    const out = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ubuf } }, { binding: 1, resource: { buffer: out } }] });
    const t0 = performance.now();
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(p.trials / 64));
    pass.end();
    enc.copyBufferToBuffer(out, 0, read, 0, size);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const ms = performance.now() - t0;
    const data = new Float32Array(read.getMappedRange().slice(0));
    read.unmap(); [ubuf, out, read].forEach((b) => b.destroy());
    return { p, ms, la: data.filter((_, i) => i % 3 === 1), gn: data.filter((_, i) => i % 3 === 2) };
  }

  async function runCPU() {
    const p = params();
    const la = new Float32Array(p.trials), gn = new Float32Array(p.trials);
    const chunk = 2048;
    wasm.seed((Math.random() * 1e9) | 0);
    let ms = 0;
    for (let s = 0; s < p.trials; s += chunk) {
      const k = Math.min(chunk, p.trials - s);
      const t0 = performance.now();
      wasm.monte_carlo(p.n, p.m, p.g, p.sigma, p.scale, k, 4);
      ms += performance.now() - t0;
      const E = f32(wasm.ptr_errs(), 3 * k);
      for (let i = 0; i < k; i++) { la[s + i] = E[3 * i + 1]; gn[s + i] = E[3 * i + 2]; }
      $('mc-cap-cpu').textContent = `WebAssembly, f64: ${Math.round(100 * (s + k) / p.trials)}%`;
      await new Promise((r) => setTimeout(r, 0));
    }
    return { p, ms, la, gn };
  }

  const median = (a) => { const b = Float32Array.from(a.filter((v) => v >= 0 && isFinite(v))).sort(); return b[b.length >> 1]; };

  function hist(cv, r) {
    const { ctx, w, h } = setupCanvas(cv);
    const [lo, hi, nb] = BINS, bw = (hi - lo) / nb;
    if (!r) { ctx.fillStyle = css('--ink-soft'); ctx.font = '13px "IBM Plex Sans", sans-serif'; ctx.textAlign = 'center'; ctx.fillText('Not run yet', w / 2, h / 2); return; }
    const count = (arr) => { const c = new Float64Array(nb); for (const v of arr) { if (!(v > 0)) continue; const b = Math.floor((Math.log10(v) - lo) / bw); if (b >= 0 && b < nb) c[b]++; } return c; };
    const cs = [count(r.la), count(r.gn)];
    const ymax = Math.max(...cs[0], ...cs[1]) / r.la.length / bw * 1.08;
    const a = axes(ctx, w, h, { l: 50, r: 14, t: 12, b: 38 }, [lo, hi], [0, ymax], 'relative error', 'density', { xticks: [-5, -4, -3, -2, -1, 0], xfmt: (v) => (v === 0 ? '1' : '1e' + v), yfmt: (v) => v.toFixed(1) });
    [['--c-la', cs[0]], ['--c-gn', cs[1]]].forEach(([c, cnt]) => {
      ctx.beginPath(); ctx.moveTo(a.X(lo), a.Y(0));
      for (let b = 0; b < nb; b++) { const y = a.Y(cnt[b] / r.la.length / bw); ctx.lineTo(a.X(lo + b * bw), y); ctx.lineTo(a.X(lo + (b + 1) * bw), y); }
      ctx.lineTo(a.X(hi), a.Y(0)); ctx.closePath();
      ctx.fillStyle = css(c); ctx.globalAlpha = 0.35; ctx.fill(); ctx.globalAlpha = 1; ctx.strokeStyle = css(c); ctx.lineWidth = 1.5; ctx.stroke();
    });
  }

  function report() {
    hist($('mc-hist-gpu'), results.gpu); hist($('mc-hist-cpu'), results.cpu);
    const rows = [];
    for (const [key, label] of [['gpu', 'WebGPU (f32)'], ['cpu', 'WebAssembly (f64)']]) {
      const r = results[key]; if (!r) continue;
      const name = `${GROUP_NAMES[r.p.g]}${r.p.g === 3 ? '1)' : '(' + (r.p.g === 5 ? r.p.n - 1 : r.p.n) + ')'}`;
      rows.push([label, name, r.p.trials.toLocaleString(), fmt(median(r.la)), fmt(median(r.gn)), (median(r.la) / median(r.gn)).toFixed(3), r.ms.toFixed(0) + ' ms', Math.round(r.p.trials / r.ms * 1000).toLocaleString()]);
    }
    if (results.gpu && results.cpu) $('mc-cap-gpu').textContent = `WebGPU, f32: ${(results.cpu.ms / results.cpu.p.trials / (results.gpu.ms / results.gpu.p.trials)).toFixed(0)}× the WebAssembly throughput`;
    table($('mc-table'), ['Backend', 'Group', 'Trials', 'Median error, Lie algebra', 'Median error, Gauss-Newton', 'Ratio', 'Time', 'Trials per second'], rows);
  }

  // noise sweep: median errors against sigma, one GPU dispatch per noise level
  let sweep = null;
  async function runSweep() {
    const sig = [], la = [], gn = [];
    for (let e = -4; e <= -0.5 + 1e-9; e += 0.25) {
      const r = await runGPU({ sigma: pow10(e), trials: 2 ** 15 });
      sig.push(e); la.push(Math.log10(median(r.la))); gn.push(Math.log10(median(r.gn)));
    }
    sweep = { sig, la, gn, p: params() };
  }
  function drawSweep() {
    const { ctx, w, h } = setupCanvas($('mc-sweep'));
    if (!sweep) { ctx.fillStyle = css('--ink-soft'); ctx.font = '13px "IBM Plex Sans", sans-serif'; ctx.textAlign = 'center'; ctx.fillText(gpu ? 'Press “Sweep noise on GPU”' : 'Needs WebGPU', w / 2, h / 2); return; }
    const all = [...sweep.la, ...sweep.gn].filter(isFinite);
    const yr = [Math.floor(Math.min(...all)), Math.ceil(Math.max(...all))];
    const tk = (r) => { const o = []; for (let t = Math.ceil(r[0]); t <= r[1]; t++) o.push(t); return o; };
    const a = axes(ctx, w, h, { l: 50, r: 16, t: 12, b: 38 }, [-4, -0.5], yr, 'noise σ', 'median relative error', { xticks: tk([-4, -0.5]), yticks: tk(yr), xfmt: (v) => '1e' + v, yfmt: (v) => '1e' + v });
    // reference slope 1: error proportional to sigma
    ctx.strokeStyle = css('--rule'); ctx.setLineDash([4, 4]); ctx.beginPath();
    const y0 = sweep.gn[0] + 4 - 0.7;
    ctx.moveTo(a.X(-4), a.Y(y0 - 4)); ctx.lineTo(a.X(-0.5), a.Y(y0 - 0.5)); ctx.stroke(); ctx.setLineDash([]);
    for (const [c, ys] of [['--c-la', sweep.la], ['--c-gn', sweep.gn]]) {
      ctx.strokeStyle = css(c); ctx.fillStyle = css(c); ctx.lineWidth = 2; ctx.beginPath();
      ys.forEach((y, i) => (i ? ctx.lineTo(a.X(sweep.sig[i]), a.Y(y)) : ctx.moveTo(a.X(sweep.sig[i]), a.Y(y))));
      ctx.stroke();
      ys.forEach((y, i) => { ctx.beginPath(); ctx.arc(a.X(sweep.sig[i]), a.Y(y), 2.8, 0, 2 * Math.PI); ctx.fill(); });
    }
    // direct labels near the left end of each line
    ctx.font = '12px "IBM Plex Sans", sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillStyle = css('--c-la'); ctx.fillText('Lie algebra', a.X(sweep.sig[1]) - 10, a.Y(sweep.la[1]) - 12);
    ctx.fillStyle = css('--c-gn'); ctx.textBaseline = 'top'; ctx.fillText('Gauss-Newton', a.X(sweep.sig[1]) + 6, a.Y(sweep.gn[1]) + 8);
    ctx.fillStyle = css('--ink-soft'); ctx.fillText('slope 1', a.X(-1.5) + 6, a.Y(y0 - 1.5) + 6);
  }
  window.addEventListener('resize', drawSweep);

  const busy = (b) => { $('mc-gpu').disabled = b || !gpu; $('mc-sweep-btn').disabled = b || !gpu; $('mc-cpu').disabled = b; };
  $('mc-sweep-btn').addEventListener('click', async () => { busy(true); const t0 = performance.now(); try { await runSweep(); $('mc-cap-sweep').textContent = `Median error against noise, ${15} noise levels × 32,768 trials on the GPU in ${(performance.now() - t0).toFixed(0)} ms. The dashed line has slope 1.`; } finally { busy(false); drawSweep(); } });
  drawSweep();
  $('mc-gpu').addEventListener('click', async () => { busy(true); try { results.gpu = await runGPU(); $('mc-cap-gpu').textContent = 'WebGPU, f32'; } finally { busy(false); report(); } });
  $('mc-cpu').addEventListener('click', async () => { busy(true); try { results.cpu = await runCPU(); $('mc-cap-cpu').textContent = 'WebAssembly, f64'; } finally { busy(false); report(); } });
  window.addEventListener('resize', report);
  report();
  // fill the WebAssembly histogram the first time the section scrolls into view
  new IntersectionObserver((entries, obs) => { if (entries.some((e) => e.isIntersecting)) { obs.disconnect(); if (!results.cpu) $('mc-cpu').click(); } }).observe($('mc-hist-cpu'));
  // self-check: the same seed must give the same numbers. Some GPU drivers (seen in headless Chrome on
  // Apple GPUs) return inconsistent results for long-running shaders; warn instead of showing them silently.
  if (gpu) {
    const probe = { g: 4, n: 4, m: 12, sigma: 0.02, scale: 0.8, trials: 4096, seed: 12345 };
    const [r1, r2] = [await runGPU(probe), await runGPU(probe)];
    const bad = r1.gn.filter((v, i) => !(v === r2.gn[i] || (isNaN(v) && isNaN(r2.gn[i])))).length;
    if (bad) {
      $('gpu-status').textContent += ' (inconsistent results; GPU numbers may be wrong)';
      $('gpu-status').classList.replace('ok', 'bad');
      $('mc-cap-gpu').textContent = `Warning: this GPU gave different answers for identical input in ${bad} of 4096 trials.`;
    }
  }
  if (gpu) { TESTS_GPU.run = runGPU; TESTS_GPU.median = median; }
  if (gpu) { busy(true); try { results.gpu = await runGPU(); } finally { busy(false); report(); } }
}

// ---------------------------------------------------------------- tests

{
  const GROUPS = [[1, 3, 'SO(3)', 1], [2, 3, 'SL(3)', 0.5], [3, 4, 'SO(3,1)', 0.5], [4, 4, 'Sp(4)', 0.5], [5, 4, 'SE(3)', 1], [1, 6, 'SO(6)', 0.6], [4, 6, 'Sp(6)', 0.4], [6, 8, 'U(4)', 1]];
  const info = (k) => Array.from(f64(wasm.ptr_info(), k));
  const worst = (f) => Math.max(...GROUPS.map(([g, n, , sc]) => f(g, n, sc)));
  const medianOf = (a) => { const b = a.slice().sort((x, y) => x - y); return b[b.length >> 1]; };
  const errsMC = (n, m, g, sigma, scale, trials, gn) => { wasm.monte_carlo(n, m, g, sigma, scale, trials, gn); const E = f32(wasm.ptr_errs(), 3 * trials); return [0, 1, 2].map((c) => Array.from({ length: trials }, (_, i) => E[3 * i + c])); };
  const allGroups = 'all eight groups';

  const TESTS = [
    { name: 'The projection is idempotent', ref: 'Theorem 2.1', what: `max ‖P(P(L)) − P(L)‖ over ${allGroups}`, tol: 1e-12,
      run: () => worst((g, n) => { wasm.test_projection(n, g); return info(1)[0]; }) },
    { name: 'The projection is orthogonal', ref: 'Theorem 2.1', what: 'max |⟨L − P(L), P(L)⟩|, the part removed is perpendicular to the algebra', tol: 1e-12,
      run: () => worst((g, n) => { wasm.test_projection(n, g); return info(2)[1]; }) },
    { name: 'The exponential maps the algebra into the group', ref: 'Appendix A', what: 'max distance of exp(P(L)) from the group', tol: 1e-10,
      run: () => worst((g, n) => { wasm.test_projection(n, g); return info(3)[2]; }) },
    { name: 'The logarithm inverts the exponential', ref: 'Appendix A', what: 'max ‖exp(log g) − g‖ / ‖g‖ for random g', tol: 1e-10,
      run: () => worst((g, n, sc) => { wasm.test_explog(n, g, sc); return info(1)[0]; }) },
    { name: 'The logarithm of a group element lies in the algebra', ref: 'Supplement, Lemma S1.4', what: 'max ‖P(log g) − log g‖', tol: 1e-10,
      run: () => worst((g, n, sc) => { wasm.test_explog(n, g, sc); return info(2)[1]; }) },
    { name: 'The method is exact for noiseless data', ref: 'Section 2.3', what: 'max relative error of g with σ = 0, 30 vectors', tol: 1e-10,
      run: () => worst((g, n, sc) => { wasm.gen_data(n, 30, g, 0, sc); wasm.align(n, 30, g, 0); wasm.diagnostics(n, 30, g); return info(5)[4]; }) },
    { name: 'The estimate always lies in the group', ref: 'Section 2', what: 'max defect of g with σ = 0.1', tol: 1e-10,
      run: () => worst((g, n, sc) => { wasm.gen_data(n, 30, g, 0.1, sc); wasm.align(n, 30, g, 0); wasm.diagnostics(n, 30, g); return info(7)[6]; }) },
    { name: 'Gauss-Newton never increases the residual', ref: 'Section 3.3', what: 'max of ‖g_GN X − Y‖ − ‖gX − Y‖ (should be ≤ 0)', tol: 1e-12,
      run: () => worst((g, n, sc) => { wasm.gen_data(n, 30, g, 0.05, sc); wasm.align(n, 30, g, 30); wasm.diagnostics(n, 30, g); const I = info(9); return I[8] - I[5]; }) },
    { name: 'The error grows in proportion to the noise', ref: 'Theorem 3.1', what: 'median error at σ = 1e-3 divided by median at σ = 1e-4, SO(3); should be 10', lo: 9, hi: 11,
      run: () => { wasm.seed(17); const a = medianOf(errsMC(3, 20, 1, 1e-4, 1, 400, 0)[1]); wasm.seed(17); const b = medianOf(errsMC(3, 20, 1, 1e-3, 1, 400, 0)[1]); return b / a; } },
    { name: 'Isotropic data: the method matches the best rotation', ref: 'Corollary 3.3', what: 'median |err(g) / err(g_GN) − 1| for SO(3) with XXᵀ = mI, σ = 1e-4', tol: 1e-3,
      run: () => { wasm.set_isotropic(1); try { const [, la, gn] = errsMC(3, 20, 1, 1e-4, 1, 300, 20); return medianOf(la.map((v, i) => Math.abs(v / gn[i] - 1))); } finally { wasm.set_isotropic(0); } } },
    { name: 'Isotropic data in U(4) as well', ref: 'Corollary 3.3', what: 'the same quantity for the unitary group', tol: 1e-3,
      run: () => { wasm.set_isotropic(1); try { const [, la, gn] = errsMC(8, 20, 6, 1e-4, 1, 200, 20); return medianOf(la.map((v, i) => Math.abs(v / gn[i] - 1))); } finally { wasm.set_isotropic(0); } } },
    { name: 'Anisotropic data: the method is no longer optimal', ref: 'Corollary 3.3', what: 'median err(g) / err(g_GN) for SO(3) with condition number 10; should be clearly above 1', lo: 1.3, hi: Infinity,
      run: () => { wasm.set_aniso(10); try { const [, la, gn] = errsMC(3, 20, 1, 1e-4, 1, 300, 20); return medianOf(la.map((v, i) => v / gn[i])); } finally { wasm.set_aniso(1); } } },
    { name: 'A noncompact group: no first order optimality even for isotropic data', ref: 'Section 3.2', what: 'median |err(g) / err(g_GN) − 1| for Sp(4) with XXᵀ = mI; should not be small', lo: 0.05, hi: Infinity,
      run: () => { wasm.set_isotropic(1); try { const [, la, gn] = errsMC(4, 20, 4, 1e-4, 0.5, 300, 20); return medianOf(la.map((v, i) => Math.abs(v / gn[i] - 1))); } finally { wasm.set_isotropic(0); } } },
    { name: 'Lorentz: noiseless particle data gives the exact transformation', ref: 'Section 2.3', what: 'relative error, rapidity 1.2, 20 particles', tol: 1e-10,
      run: () => { wasm.make_lorentz(1.2, 0.7, 0.3, 0.4); wasm.gen_particles(20, 0, 1); wasm.align(4, 20, 3, 0); wasm.diagnostics(4, 20, 3); return info(5)[4]; } },
    { name: 'Storage ring: noiseless turns give the exact tunes', ref: 'Section 7.1', what: 'max |Q − Q_true| from the eigenvalues of g', tol: 1e-9,
      run: () => { wasm.seed(2); wasm.gen_accel(0.31, 0.27, 0.05, 64, 0); wasm.align(4, 64, 4, 0); wasm.eigenvalues(2, 4); const e = f64(wasm.ptr_eig(), 8); const q = []; for (let i = 0; i < 4; i++) if (e[2 * i + 1] > 0) q.push(wasm.angle(e[2 * i + 1], e[2 * i]) / (2 * Math.PI)); q.sort(); return Math.max(Math.abs(q[0] - 0.27), Math.abs(q[1] - 0.31)); } },
    { name: 'Quantum gate: noiseless probes give fidelity one', ref: 'Section 2.3', what: '1 − |tr U†g| / 4 for √iSWAP, 6 probe states', tol: 1e-12,
      run: () => { wasm.gen_quantum(2, 6, 0); wasm.align(8, 12, 6, 0); return 1 - wasm.gate_fidelity(2, 8); } },
    { name: 'Expected failure: no real logarithm', ref: 'Section 2.3', what: 'g = diag(−2, −1/2) in SL(2) has negative eigenvalues, so the method cannot be exact; relative error of g', lo: 0.1, hi: Infinity, expectFail: true,
      run: () => { const G = f64(wasm.ptr_Gt(), 4); G.set([-2, 0, 0, -0.5]); const X = f64(wasm.ptr_X(), 40); for (let i = 0; i < 40; i++) X[i] = Math.sin(1.7 * i + 1); wasm.noisy_outputs(2, 20, 2, 0); wasm.align(2, 20, 2, 0); wasm.diagnostics(2, 20, 2); return info(5)[4]; } },
    { name: 'The GPU kernel agrees with the WebAssembly code', ref: 'this page', what: 'largest relative difference of the median errors of g and Gauss-Newton, 5 groups', tol: 0.05, gpu: true,
      run: async () => {
        let worstDiff = 0;
        for (const [g, n] of [[1, 3], [2, 3], [3, 4], [4, 4], [5, 4]]) {
          const r = await TESTS_GPU.run({ g, n, m: 12, sigma: 0.02, scale: 0.8, trials: 16384 });
          wasm.seed(99); const [, la, gn] = errsMC(n, 12, g, 0.02, 0.8, 4096, 4);
          worstDiff = Math.max(worstDiff, Math.abs(TESTS_GPU.median(r.la) / medianOf(la) - 1), Math.abs(TESTS_GPU.median(r.gn) / medianOf(gn) - 1));
        }
        return worstDiff;
      } },
  ];

  async function runAll() {
    $('ts-run').disabled = true;
    const rows = []; let passed = 0, ran = 0;
    wasm.seed(Date.now() % 100000);
    for (const t of TESTS) {
      let value = NaN, status;
      if (t.gpu && !TESTS_GPU.run) status = 'skipped (no WebGPU)';
      else {
        try {
          value = await t.run();
          const ok = t.tol !== undefined ? Math.abs(value) <= t.tol || value <= 0 : value >= t.lo && value <= t.hi;
          status = ok ? (t.expectFail ? 'pass (fails as expected)' : 'pass') : 'FAIL';
          ran++; if (ok) passed++;
        } catch (e) { status = 'error: ' + e.message; ran++; }
      }
      const bound = t.tol !== undefined ? '≤ ' + fmt(t.tol, 0) : t.hi === Infinity ? '≥ ' + t.lo : `${t.lo} to ${t.hi}`;
      rows.push([`<b>${t.name}</b><br><span class="note">${t.what}</span>`, t.ref, fmt(value), bound, `<span class="${status.startsWith('pass') ? 'ok' : status === 'FAIL' || status.startsWith('error') ? 'bad' : ''}">${status}</span>`]);
      table($('ts-table'), ['Test', 'Paper', 'Value', 'Required', 'Result'], rows);
      await new Promise((r) => setTimeout(r, 0));
    }
    $('ts-summary').textContent = `${passed} of ${ran} tests passed.`;
    $('ts-run').disabled = false;
  }
  $('ts-run').addEventListener('click', runAll);
  table($('ts-table'), ['Test', 'Paper', 'Value', 'Required', 'Result'], TESTS.map((t) => [`<b>${t.name}</b><br><span class="note">${t.what}</span>`, t.ref, '', '', 'not run']));
  if (new URLSearchParams(location.search).has('autotest')) { await runAll(); document.body.dataset.testsDone = '1'; }
}

// ---------------------------------------------------------------- code viewer and math

{
  const tabs = document.querySelectorAll('.tabs button');
  const cache = {};
  async function show(btn) {
    tabs.forEach((b) => b.setAttribute('aria-selected', b === btn));
    const file = btn.dataset.file;
    $('code-name').textContent = file; $('code-dl').href = file;
    const view = $('code-view');
    view.className = 'language-' + btn.dataset.lang;
    view.removeAttribute('data-highlighted');
    view.textContent = cache[file] ??= await (await fetch(file)).text();
    if (window.hljs) hljs.highlightElement(view);
  }
  tabs.forEach((b) => b.addEventListener('click', () => show(b)));
  show(tabs[0]);
  if (window.hljs) document.querySelectorAll('.snippet code').forEach((el) => hljs.highlightElement(el));
}
