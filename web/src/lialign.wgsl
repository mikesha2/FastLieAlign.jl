// lialign.wgsl: Monte Carlo of the Lie algebra method on the GPU.
// One invocation = one independent alignment problem:
//   draw g* = exp(proj(scale * randn)), X = randn (n x m),
//   Y = g* X + sigma * noise,
//   g0 = (Y X^T)(X X^T)^{-1},  g = exp(proj(log g0)).
//   then a few Gauss-Newton steps on the group, which minimize ||gX - Y|| over g in G.
// Writes the relative errors of g0, g and the Gauss-Newton estimate, three numbers per trial.
//
// Matrices are n x n with n <= 4, held in the built-in mat4x4<f32> type
// (element (i, j) is m[j][i]) and padded with zeros outside the leading n x n block.
// Zero padding is preserved by sums and products, so most operations need no knowledge of n.

struct Params {
  n: u32, m: u32, group: u32, trials: u32,
  sigma: f32, scale: f32, seed: u32, pad: u32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> outErr: array<f32>;

alias M = mat4x4<f32>;

const GRP_SO: u32 = 1u;
const GRP_SL: u32 = 2u;
const GRP_LORENTZ: u32 = 3u;
const GRP_SP: u32 = 4u;
const GRP_SE: u32 = 5u;
const TAU: f32 = 6.283185307179586;

// ---------------------------------------------------------------- random numbers (PCG)
fn pcg(s: ptr<function, u32>) -> f32 {
  *s = *s * 747796405u + 2891336453u;
  var w = ((*s >> ((*s >> 28u) + 4u)) ^ *s) * 277803737u;
  w = (w >> 22u) ^ w;
  return (f32(w >> 8u) + 0.5) / 16777216.0;
}
fn hash(x: u32) -> u32 {
  var s = x * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn randn(s: ptr<function, u32>) -> f32 {
  let u = pcg(s);
  let v = pcg(s);
  return sqrt(-2.0 * log(u)) * cos(TAU * v);
}

// ---------------------------------------------------------------- matrices
// mask(n): 1 on rows/columns below n, 0 on the padding
fn mask(n: u32) -> vec4<f32> {
  return select(vec4<f32>(0.0), vec4<f32>(1.0), vec4<u32>(0u, 1u, 2u, 3u) < vec4<u32>(n));
}
fn eye(n: u32) -> M {
  let d = mask(n);
  return M(vec4<f32>(d.x, 0.0, 0.0, 0.0), vec4<f32>(0.0, d.y, 0.0, 0.0),
           vec4<f32>(0.0, 0.0, d.z, 0.0), vec4<f32>(0.0, 0.0, 0.0, d.w));
}
fn scaleM(a: M, f: f32) -> M { return M(a[0] * f, a[1] * f, a[2] * f, a[3] * f); }
fn norm1(a: M) -> f32 {
  let one = vec4<f32>(1.0);
  return max(max(dot(abs(a[0]), one), dot(abs(a[1]), one)), max(dot(abs(a[2]), one), dot(abs(a[3]), one)));
}
fn normF(a: M) -> f32 { return sqrt(dot(a[0], a[0]) + dot(a[1], a[1]) + dot(a[2], a[2]) + dot(a[3], a[3])); }
fn trace(a: M) -> f32 { return a[0].x + a[1].y + a[2].z + a[3].w; }

// Inverse of the leading n x n block. The padding is replaced by an identity, so the 4 x 4 matrix
// is block diagonal, and the full inverse is computed in closed form from 2 x 2 minors (the
// cofactor formula). This avoids dynamic indexing, which gave nondeterministic results on some GPUs.
fn inv(a_in: M, n: u32) -> M {
  let pad = eye(4u) - eye(n);
  let m = a_in + pad;
  let a00 = m[0][0]; let a01 = m[0][1]; let a02 = m[0][2]; let a03 = m[0][3];
  let a10 = m[1][0]; let a11 = m[1][1]; let a12 = m[1][2]; let a13 = m[1][3];
  let a20 = m[2][0]; let a21 = m[2][1]; let a22 = m[2][2]; let a23 = m[2][3];
  let a30 = m[3][0]; let a31 = m[3][1]; let a32 = m[3][2]; let a33 = m[3][3];
  let b00 = a00 * a11 - a01 * a10; let b01 = a00 * a12 - a02 * a10;
  let b02 = a00 * a13 - a03 * a10; let b03 = a01 * a12 - a02 * a11;
  let b04 = a01 * a13 - a03 * a11; let b05 = a02 * a13 - a03 * a12;
  let b06 = a20 * a31 - a21 * a30; let b07 = a20 * a32 - a22 * a30;
  let b08 = a20 * a33 - a23 * a30; let b09 = a21 * a32 - a22 * a31;
  let b10 = a21 * a33 - a23 * a31; let b11 = a22 * a33 - a23 * a32;
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  let f = 1.0 / det;
  let r = M(
    vec4<f32>(a11 * b11 - a12 * b10 + a13 * b09, a02 * b10 - a01 * b11 - a03 * b09,
              a31 * b05 - a32 * b04 + a33 * b03, a22 * b04 - a21 * b05 - a23 * b03),
    vec4<f32>(a12 * b08 - a10 * b11 - a13 * b07, a00 * b11 - a02 * b08 + a03 * b07,
              a32 * b02 - a30 * b05 - a33 * b01, a20 * b05 - a22 * b02 + a23 * b01),
    vec4<f32>(a10 * b10 - a11 * b08 + a13 * b06, a01 * b08 - a00 * b10 - a03 * b06,
              a30 * b04 - a31 * b02 + a33 * b00, a21 * b02 - a20 * b04 - a23 * b00),
    vec4<f32>(a11 * b07 - a10 * b09 - a12 * b06, a00 * b09 - a01 * b07 + a02 * b06,
              a31 * b01 - a30 * b03 - a32 * b00, a20 * b03 - a21 * b01 + a22 * b00));
  return scaleM(r, f) - pad;
}

// exp by scaling and squaring with a Taylor polynomial
fn expm(a: M, n: u32) -> M {
  var s = 0u;
  var nrm = norm1(a);
  while (nrm > 0.25 && s < 40u) { nrm *= 0.5; s++; }
  let b = scaleM(a, exp2(-f32(s)));
  var e = eye(n);
  var t = eye(n);
  for (var k = 1u; k <= 10u; k++) {
    t = scaleM(t * b, 1.0 / f32(k));
    e = e + t;
  }
  for (var i = 0u; i < s; i++) { e = e * e; }
  return e;
}

// square root by the Denman-Beavers iteration
fn sqrtm(a: M, n: u32) -> M {
  var y = a;
  var z = eye(n);
  for (var it = 0u; it < 20u; it++) {
    let yn = scaleM(y + inv(z, n), 0.5);
    let zn = scaleM(z + inv(y, n), 0.5);
    let d = normF(yn - y);
    y = yn;
    z = zn;
    if (d < 1e-6 * normF(y)) { break; }
  }
  return y;
}

// principal log: square roots until near I, then 2 atanh((B - I)(B + I)^{-1})
fn logm(a: M, n: u32) -> M {
  let id = eye(n);
  var b = a;
  var k = 0u;
  loop {
    if (norm1(b - id) < 0.2 || k > 30u) { break; }
    b = sqrtm(b, n);
    k++;
  }
  let z = (b - id) * inv(b + id, n);
  let z2 = z * z;
  var t = z;
  var l = z;
  for (var j = 3u; j < 16u; j += 2u) {
    t = t * z2;
    l = l + scaleM(t, 1.0 / f32(j));
  }
  return scaleM(l, exp2(f32(k + 1u)));
}

// Frobenius-orthogonal projection onto the Lie algebra (same formulas as lialign.c)
fn proj(l: M, n: u32, g: u32) -> M {
  let lt = transpose(l);
  if (g == GRP_SO) { return scaleM(l - lt, 0.5); }
  if (g == GRP_SL) { return l - scaleM(eye(n), trace(l) / f32(n)); }
  if (g == GRP_LORENTZ) {
    // eta = diag(-1, 1, 1, 1), time coordinate first; the projection is the same for -eta
    let eta = M(vec4<f32>(-1.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 1.0, 0.0, 0.0),
                vec4<f32>(0.0, 0.0, 1.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0));
    return scaleM(l - eta * lt * eta, 0.5);
  }
  if (g == GRP_SP) {
    // J = [[0, I], [-I, 0]] for n = 4, or [[0, 1], [-1, 0]] for n = 2
    var j = M(vec4<f32>(0.0, 0.0, -1.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, -1.0),
              vec4<f32>(1.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 1.0, 0.0, 0.0));
    if (n == 2u) { j = M(vec4<f32>(0.0, -1.0, 0.0, 0.0), vec4<f32>(1.0, 0.0, 0.0, 0.0), vec4<f32>(), vec4<f32>()); }
    return scaleM(l + j * lt * j, 0.5);
  }
  if (g == GRP_SE) {
    // skew block for the rotation, keep the translation column, zero last row
    // the GPU code handles SE(3) only, so the homogeneous coordinate is the fourth
    let rm = vec4<f32>(1.0, 1.0, 1.0, 0.0);
    let t = scaleM(l - lt, 0.5);
    return M(t[0] * rm, t[1] * rm, t[2] * rm, l[3] * rm);
  }
  return l;
}

fn fdot(a: M, b: M) -> f32 { return dot(a[0], b[0]) + dot(a[1], b[1]) + dot(a[2], b[2]) + dot(a[3], b[3]); }

// Gauss-Newton on the group, using only C = X X^T and S = Y X^T:
// with G = gX, the normal equations are A c = b with A_ab = <B_a, B_b g C g^T> and
// b_a = <B_a, (S - g C) g^T>, for an orthonormal basis B_a of the Lie algebra. Then g <- exp(sum c_a B_a) g.
fn gauss_newton(g_in: M, cm: M, sm: M, n: u32, grp: u32, iters: u32) -> M {
  var basis: array<M, 16>;
  var d = 0u;
  for (var idx = 0u; idx < 16u; idx++) {
    let i = idx / 4u;
    let j = idx % 4u;
    if (i >= n || j >= n) { continue; }
    var e = M();
    e[j][i] = 1.0;
    var p = proj(e, n, grp);
    for (var b = 0u; b < d; b++) { p = p - scaleM(basis[b], fdot(basis[b], p)); }
    let nrm = normF(p);
    if (nrm > 1e-3) { basis[d] = scaleM(p, 1.0 / nrm); d++; }
  }
  var g = g_in;
  for (var it = 0u; it < iters; it++) {
    let pm = g * cm * transpose(g);
    let qm = (sm - g * cm) * transpose(g);
    var a: array<f32, 256>;
    var rhs: array<f32, 16>;
    for (var r = 0u; r < d; r++) {
      rhs[r] = fdot(basis[r], qm);
      let bp = basis[r] * pm;
      for (var c = 0u; c < d; c++) { a[c * 16u + r] = fdot(basis[c], bp); }
    }
    // Gaussian elimination; A is symmetric positive definite, so no pivoting is needed
    for (var c = 0u; c < d; c++) {
      let piv = a[c * 16u + c];
      for (var r = c + 1u; r < d; r++) {
        let f = a[r * 16u + c] / piv;
        for (var k = c; k < d; k++) { a[r * 16u + k] -= f * a[c * 16u + k]; }
        rhs[r] -= f * rhs[c];
      }
    }
    var delta = M();
    for (var cc = 0u; cc < d; cc++) {
      let c = d - 1u - cc;
      var v = rhs[c];
      for (var k = c + 1u; k < d; k++) { v -= a[c * 16u + k] * rhs[k]; }
      rhs[c] = v / a[c * 16u + c];
      delta = delta + scaleM(basis[c], rhs[c]);
    }
    g = expm(delta, n) * g;
  }
  return g;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= P.trials) { return; }
  let n = P.n;
  let g = P.group;
  var s: u32 = hash(t ^ hash(P.seed));
  let msk = mask(n);

  var a = M();
  for (var c = 0u; c < 4u; c++) {
    a[c] = vec4<f32>(randn(&s), randn(&s), randn(&s), randn(&s)) * msk * (P.scale * msk[c]);
  }
  let gt = expm(proj(a, n, g), n);

  // accumulate C = X X^T and S = Y X^T one column at a time, so X and Y are never stored
  let rows = select(n, n - 1u, g == GRP_SE);
  let rmask = mask(rows);
  let onemask = msk - rmask; // 1 in the homogeneous coordinate of SE(n - 1)
  var cm = M();
  var sm = M();
  for (var k = 0u; k < P.m; k++) {
    let x = vec4<f32>(randn(&s), randn(&s), randn(&s), randn(&s)) * rmask + onemask;
    let y = gt * x + vec4<f32>(randn(&s), randn(&s), randn(&s), randn(&s)) * rmask * P.sigma;
    cm = cm + M(x * x.x, x * x.y, x * x.z, x * x.w);
    sm = sm + M(y * x.x, y * x.y, y * x.z, y * x.w);
  }
  let g0 = sm * inv(cm, n);
  let gla = expm(proj(logm(g0, n), n, g), n);
  let nt = normF(gt);
  let ggn = gauss_newton(gla, cm, sm, n, g, 4u);
  outErr[3u * t] = normF(g0 - gt) / nt;
  outErr[3u * t + 1u] = normF(gla - gt) / nt;
  outErr[3u * t + 2u] = normF(ggn - gt) / nt;
}
