/* lialign.c: the Lie algebra alignment method, compiled to WebAssembly.
 *
 * Given data X (n x m) and Y = g X + noise with g in a matrix Lie group G,
 *   g0 = Y X^+,  L = proj_g(log g0),  g = exp L,
 * optionally followed by Gauss-Newton steps on the group.
 *
 * No libc: everything (exp, log, sin, cos, the RNG) is implemented here.
 * Build:  clang --target=wasm32 -O3 -nostdlib -Wl,--no-entry -Wl,--export-dynamic \
 *               -o ../lialign.wasm lialign.c
 * Matrices are dense, row major, n <= 8.
 */

#define MAXN 8
#define NN (MAXN * MAXN)
#define MAXCOLS 16384 /* n * m <= 8 * 16384 */
#define MAXTRIALS (1 << 20)
#define EXPORT __attribute__((visibility("default")))

enum { GRP_GL = 0, GRP_SO = 1, GRP_SL = 2, GRP_LORENTZ = 3, GRP_SP = 4, GRP_SE = 5, GRP_U = 6 };
/* GRP_U: the unitary group U(k), stored as real 2k x 2k matrices [[A, -B], [B, A]] for A + iB. */

/* Shared buffers that JavaScript reads and writes. */
static double BX[MAXN * MAXCOLS], BY[MAXN * MAXCOLS];
static double GT[NN], G0[NN], GLA[NN], GGN[NN];
static double TRACK[4 * 65536];
static float ERRS[3 * MAXTRIALS];
static double EIG[2 * MAXN];
static double INFO[16];

EXPORT double *ptr_X(void) { return BX; }
EXPORT double *ptr_Y(void) { return BY; }
EXPORT double *ptr_Gt(void) { return GT; }
EXPORT double *ptr_G0(void) { return G0; }
EXPORT double *ptr_Gla(void) { return GLA; }
EXPORT double *ptr_Ggn(void) { return GGN; }
EXPORT double *ptr_track(void) { return TRACK; }
EXPORT float *ptr_errs(void) { return ERRS; }
EXPORT double *ptr_eig(void) { return EIG; }
EXPORT double *ptr_info(void) { return INFO; }

/* ---------------------------------------------------------------- scalar math */

typedef union { double d; unsigned long long u; } dbits;
static const double LN2 = 0.6931471805599453094, PI = 3.14159265358979323846;

static double fabs_(double x) { return x < 0 ? -x : x; }
static double sqrt_(double x) { return __builtin_sqrt(x); }
static double floor_(double x) { return __builtin_floor(x); }

static double exp_(double x) {
  if (x > 709) x = 709;
  if (x < -745) return 0;
  double k = floor_(x / LN2 + 0.5), r = x - k * LN2, s = 1, t = 1;
  for (int i = 1; i < 18; i++) { t *= r / i; s += t; }
  dbits b; b.u = (unsigned long long)((long long)k + 1023) << 52;
  return s * b.d;
}

static double log_(double x) {
  if (x <= 0) return -1e308;
  dbits b; b.d = x;
  int e = (int)((b.u >> 52) & 0x7ff) - 1023;
  b.u = (b.u & 0x000fffffffffffffULL) | 0x3ff0000000000000ULL; /* mantissa in [1,2) */
  double mnt = b.d;
  if (mnt > 1.4142135623730951) { mnt *= 0.5; e++; }
  double z = (mnt - 1) / (mnt + 1), z2 = z * z, s = 0, p = z;
  for (int i = 1; i < 40; i += 2) { s += p / i; p *= z2; }
  return e * LN2 + 2 * s;
}

static double sin_kernel(double r) { double s = r, t = r; for (int i = 1; i < 12; i++) { t *= -r * r / ((2 * i) * (2 * i + 1)); s += t; } return s; }
static double cos_kernel(double r) { double s = 1, t = 1; for (int i = 1; i < 12; i++) { t *= -r * r / ((2 * i - 1) * (2 * i)); s += t; } return s; }
static void sincos_(double x, double *s, double *c) {
  double k = floor_(x / (PI / 2) + 0.5);
  double r = x - k * 1.5707963267341256 - k * 6.077100506506192e-11;
  long long q = (long long)k & 3;
  double sr = sin_kernel(r), cr = cos_kernel(r);
  if (q == 0) { *s = sr; *c = cr; }
  else if (q == 1) { *s = cr; *c = -sr; }
  else if (q == 2) { *s = -sr; *c = -cr; }
  else { *s = -cr; *c = sr; }
}

static double atan_(double x) {
  int neg = x < 0; if (neg) x = -x;
  int inv = x > 1; if (inv) x = 1 / x;
  /* halve the angle twice: atan x = 2 atan(x / (1 + sqrt(1 + x^2))) */
  x = x / (1 + sqrt_(1 + x * x));
  x = x / (1 + sqrt_(1 + x * x));
  double x2 = x * x, s = 0, p = x;
  for (int i = 1; i < 30; i += 2) { s += ((i >> 1) & 1 ? -p : p) / i; p *= x2; }
  s *= 4;
  if (inv) s = PI / 2 - s;
  return neg ? -s : s;
}
static double atan2_(double y, double x) {
  if (x > 0) return atan_(y / x);
  if (x < 0) return y >= 0 ? atan_(y / x) + PI : atan_(y / x) - PI;
  return y > 0 ? PI / 2 : (y < 0 ? -PI / 2 : 0);
}

/* splitmix64 random numbers and Box-Muller normals */
static unsigned long long rng_state = 1;
static double urand(void) {
  unsigned long long z = (rng_state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  z ^= z >> 31;
  return ((z >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}
static int have_spare = 0; static double spare;
static double randn(void) {
  if (have_spare) { have_spare = 0; return spare; }
  double r = sqrt_(-2 * log_(urand())), s, c;
  sincos_(2 * PI * urand(), &s, &c);
  spare = r * s; have_spare = 1;
  return r * c;
}
EXPORT void seed(double s) { rng_state = (unsigned long long)s * 2654435761ULL + 12345; have_spare = 0; }

/* ---------------------------------------------------------------- matrices */

static void mcopy(double *A, const double *B, int n) { for (int i = 0; i < n * n; i++) A[i] = B[i]; }
static void meye(double *A, int n) { for (int i = 0; i < n * n; i++) A[i] = 0; for (int i = 0; i < n; i++) A[i * n + i] = 1; }
static void mmul(double *C, const double *A, const double *B, int n) {
  double T[NN];
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      double s = 0;
      for (int k = 0; k < n; k++) s += A[i * n + k] * B[k * n + j];
      T[i * n + j] = s;
    }
  mcopy(C, T, n);
}
static void mtrans(double *A, const double *B, int n) {
  double T[NN];
  for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) T[i * n + j] = B[j * n + i];
  mcopy(A, T, n);
}
static double norm1(const double *A, int n) {
  double best = 0;
  for (int j = 0; j < n; j++) { double s = 0; for (int i = 0; i < n; i++) s += fabs_(A[i * n + j]); if (s > best) best = s; }
  return best;
}
static double normF(const double *A, int len) { double s = 0; for (int i = 0; i < len; i++) s += A[i] * A[i]; return sqrt_(s); }

/* Gauss-Jordan inverse with partial pivoting; returns 0 if singular. */
static int minv(double *Ainv, const double *A, int n) {
  double M[NN], R[NN];
  mcopy(M, A, n); meye(R, n);
  for (int c = 0; c < n; c++) {
    int p = c;
    for (int r = c + 1; r < n; r++) if (fabs_(M[r * n + c]) > fabs_(M[p * n + c])) p = r;
    if (fabs_(M[p * n + c]) < 1e-300) return 0;
    if (p != c) for (int k = 0; k < n; k++) {
      double t = M[c * n + k]; M[c * n + k] = M[p * n + k]; M[p * n + k] = t;
      t = R[c * n + k]; R[c * n + k] = R[p * n + k]; R[p * n + k] = t;
    }
    double d = 1 / M[c * n + c];
    for (int k = 0; k < n; k++) { M[c * n + k] *= d; R[c * n + k] *= d; }
    for (int r = 0; r < n; r++) if (r != c) {
      double f = M[r * n + c];
      if (f != 0) for (int k = 0; k < n; k++) { M[r * n + k] -= f * M[c * n + k]; R[r * n + k] -= f * R[c * n + k]; }
    }
  }
  mcopy(Ainv, R, n);
  return 1;
}

/* Solve a dense k x k system (k <= 64) in place by Gaussian elimination. */
static void solve(double *A, double *b, int k) {
  for (int c = 0; c < k; c++) {
    int p = c;
    for (int r = c + 1; r < k; r++) if (fabs_(A[r * k + c]) > fabs_(A[p * k + c])) p = r;
    if (p != c) { for (int j = 0; j < k; j++) { double t = A[c * k + j]; A[c * k + j] = A[p * k + j]; A[p * k + j] = t; } double t = b[c]; b[c] = b[p]; b[p] = t; }
    double d = A[c * k + c];
    if (fabs_(d) < 1e-300) { b[c] = 0; continue; }
    for (int r = c + 1; r < k; r++) { double f = A[r * k + c] / d; for (int j = c; j < k; j++) A[r * k + j] -= f * A[c * k + j]; b[r] -= f * b[c]; }
  }
  for (int c = k - 1; c >= 0; c--) {
    double s = b[c];
    for (int j = c + 1; j < k; j++) s -= A[c * k + j] * b[j];
    b[c] = fabs_(A[c * k + c]) < 1e-300 ? 0 : s / A[c * k + c];
  }
}

/* exp by scaling and squaring with a Taylor polynomial */
static void expm(double *E, const double *A, int n) {
  double B[NN], T[NN];
  double nrm = norm1(A, n);
  int s = 0;
  while (nrm > 0.25 && s < 60) { nrm *= 0.5; s++; }
  double scale = 1; for (int i = 0; i < s; i++) scale *= 0.5;
  for (int i = 0; i < n * n; i++) B[i] = A[i] * scale;
  meye(E, n); meye(T, n);
  for (int k = 1; k <= 16; k++) {
    mmul(T, T, B, n);
    for (int i = 0; i < n * n; i++) { T[i] /= k; E[i] += T[i]; }
  }
  for (int i = 0; i < s; i++) mmul(E, E, E, n);
}

/* square root by the Denman-Beavers iteration */
static void sqrtm(double *S, const double *A, int n) {
  double Y[NN], Z[NN], Yi[NN], Zi[NN];
  mcopy(Y, A, n); meye(Z, n);
  for (int it = 0; it < 60; it++) {
    if (!minv(Yi, Y, n) || !minv(Zi, Z, n)) break;
    double change = 0;
    for (int i = 0; i < n * n; i++) {
      double y = 0.5 * (Y[i] + Zi[i]);
      change += (y - Y[i]) * (y - Y[i]);
      Y[i] = y; Z[i] = 0.5 * (Z[i] + Yi[i]);
    }
    if (change < 1e-30 * (1 + normF(Y, n * n))) break;
  }
  mcopy(S, Y, n);
}

/* principal logarithm by inverse scaling and squaring:
 * take square roots until A is near I, then log A = 2 atanh((A - I)(A + I)^{-1}) */
static void logm(double *L, const double *A, int n) {
  double B[NN], P[NN], Q[NN], Z[NN], Z2[NN], T[NN];
  mcopy(B, A, n);
  int k = 0;
  for (;;) {
    mcopy(T, B, n);
    for (int i = 0; i < n; i++) T[i * n + i] -= 1;
    if (norm1(T, n) < 0.2 || k > 50) break;
    sqrtm(B, B, n); k++;
  }
  mcopy(P, B, n); mcopy(Q, B, n);
  for (int i = 0; i < n; i++) { P[i * n + i] -= 1; Q[i * n + i] += 1; }
  minv(Q, Q, n);
  mmul(Z, P, Q, n);
  mmul(Z2, Z, Z, n);
  mcopy(T, Z, n);
  for (int i = 0; i < n * n; i++) L[i] = Z[i];
  for (int j = 3; j < 40; j += 2) {
    mmul(T, T, Z2, n);
    for (int i = 0; i < n * n; i++) L[i] += T[i] / j;
  }
  double f = 2; for (int i = 0; i < k; i++) f *= 2;
  for (int i = 0; i < n * n; i++) L[i] *= f;
}

/* ---------------------------------------------------------------- projections */

/* Orthogonal projection onto the Lie algebra of the group, in the Frobenius inner product.
 *   so(n):      (L - L^T)/2
 *   sl(n):      L - tr(L)/n I
 *   so(n-1,1):  (L - eta L^T eta)/2, eta = diag(-1,1,...,1), time coordinate first
 *   sp(2k):     (L + J L^T J)/2,      J = [[0, I], [-I, 0]]
 *   se(n-1):    skew block, translation column, zero last row */
static void proj(double *P, const double *L, int n, int g) {
  double T[NN];
  if (g == GRP_GL) { mcopy(P, L, n); return; }
  if (g == GRP_U) {
    /* u(k): average the blocks so the matrix is complex linear, then take the skew part */
    int h = n / 2;
    for (int i = 0; i < h; i++) for (int j = 0; j < h; j++) {
      double A = 0.25 * (L[i * n + j] + L[(i + h) * n + j + h] - L[j * n + i] - L[(j + h) * n + i + h]);
      double B = 0.25 * (L[(i + h) * n + j] - L[i * n + j + h] + L[(j + h) * n + i] - L[j * n + i + h]);
      T[i * n + j] = A; T[(i + h) * n + j + h] = A;
      T[(i + h) * n + j] = B; T[i * n + j + h] = -B;
    }
    mcopy(P, T, n);
    return;
  }
  if (g == GRP_SO) {
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) T[i * n + j] = 0.5 * (L[i * n + j] - L[j * n + i]);
  } else if (g == GRP_SL) {
    double tr = 0; for (int i = 0; i < n; i++) tr += L[i * n + i];
    mcopy(T, L, n); for (int i = 0; i < n; i++) T[i * n + i] -= tr / n;
  } else if (g == GRP_LORENTZ) {
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
      double s = ((i == 0) != (j == 0)) ? -1 : 1; /* eta_ii eta_jj */
      T[i * n + j] = 0.5 * (L[i * n + j] - s * L[j * n + i]);
    }
  } else if (g == GRP_SP) {
    int h = n / 2;
    /* (J L^T J)_{ij} = sum J_ia L_ba J_bj; J = [[0,I],[-I,0]] */
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
      int a = i < h ? i + h : i - h; double sa = i < h ? 1 : -1;
      int b = j < h ? j + h : j - h; double sb = j < h ? -1 : 1;
      T[i * n + j] = 0.5 * (L[i * n + j] + sa * sb * L[b * n + a]);
    }
  } else if (g == GRP_SE) {
    int d = n - 1;
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
      if (i == d) T[i * n + j] = 0;
      else if (j == d) T[i * n + j] = L[i * n + j];
      else T[i * n + j] = 0.5 * (L[i * n + j] - L[j * n + i]);
    }
  }
  mcopy(P, T, n);
}

/* Frobenius-orthonormal basis of the Lie algebra (Gram-Schmidt on projected E_ij). */
static double BASIS[NN * NN];
static int basis(int n, int g) {
  int d = 0;
  double E[NN], P[NN];
  for (int i = 0; i < n * n; i++) {
    for (int k = 0; k < n * n; k++) E[k] = 0;
    E[i] = 1;
    proj(P, E, n, g);
    for (int b = 0; b < d; b++) {
      double dot = 0; for (int k = 0; k < n * n; k++) dot += BASIS[b * NN + k] * P[k];
      for (int k = 0; k < n * n; k++) P[k] -= dot * BASIS[b * NN + k];
    }
    double nrm = normF(P, n * n);
    if (nrm > 1e-10) { for (int k = 0; k < n * n; k++) BASIS[d * NN + k] = P[k] / nrm; d++; }
  }
  return d;
}

/* ---------------------------------------------------------------- the method */

/* g0 = Y X^+ = (Y X^T)(X X^T)^{-1} for full row rank X */
static int least_squares(double *G, const double *X, const double *Y, int n, int m) {
  double C[NN], S[NN];
  for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
    double c = 0, s = 0;
    for (int k = 0; k < m; k++) { c += X[i * m + k] * X[j * m + k]; s += Y[i * m + k] * X[j * m + k]; }
    C[i * n + j] = c; S[i * n + j] = s;
  }
  if (!minv(C, C, n)) return 0;
  mmul(G, S, C, n);
  return 1;
}

static void lie_algebra(double *G, const double *G0in, int n, int g) {
  double L[NN];
  logm(L, G0in, n);
  proj(L, L, n, g);
  expm(G, L, n);
}

/* Gauss-Newton on the group: g <- exp(sum c_k B_k) g, with c minimizing ||Y - (I + sum c_k B_k) g X||.
 * Normal equations only need P = (gX)(gX)^T and Q = R (gX)^T. */
static int gauss_newton(double *G, const double *X, const double *Y, int n, int m, int g, int maxit) {
  int d = basis(n, g), it;
  static double GX[MAXN * MAXCOLS];
  double P[NN], Q[NN], A[NN * NN], b[NN], BtB[NN], D[NN], E[NN];
  for (it = 0; it < maxit; it++) {
    for (int i = 0; i < n; i++) for (int k = 0; k < m; k++) {
      double s = 0; for (int j = 0; j < n; j++) s += G[i * n + j] * X[j * m + k];
      GX[i * m + k] = s;
    }
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
      double p = 0, q = 0;
      for (int k = 0; k < m; k++) { p += GX[i * m + k] * GX[j * m + k]; q += (Y[i * m + k] - GX[i * m + k]) * GX[j * m + k]; }
      P[i * n + j] = p; Q[i * n + j] = q;
    }
    for (int a = 0; a < d; a++) {
      const double *Ba = BASIS + a * NN;
      double s = 0; for (int k = 0; k < n * n; k++) s += Ba[k] * Q[k];
      b[a] = s;
      for (int c = a; c < d; c++) {
        const double *Bc = BASIS + c * NN;
        /* <Ba P, Bc> = tr(P^T Ba^T Bc) = sum (Ba^T Bc) .* P */
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
          double t = 0; for (int k = 0; k < n; k++) t += Ba[k * n + i] * Bc[k * n + j];
          BtB[i * n + j] = t;
        }
        double v = 0; for (int k = 0; k < n * n; k++) v += BtB[k] * P[k];
        A[a * d + c] = A[c * d + a] = v;
      }
    }
    solve(A, b, d);
    for (int k = 0; k < n * n; k++) D[k] = 0;
    for (int a = 0; a < d; a++) for (int k = 0; k < n * n; k++) D[k] += b[a] * BASIS[a * NN + k];
    expm(E, D, n);
    mmul(G, E, G, n);
    if (normF(D, n * n) < 1e-14 * (1 + normF(G, n * n))) { it++; break; }
  }
  return it;
}

/* Run all three estimates on the data in BX, BY. Writes G0, GLA, GGN and returns 1 on success. */
EXPORT int align(int n, int m, int g, int gn_iters) {
  if (!least_squares(G0, BX, BY, n, m)) return 0;
  lie_algebra(GLA, G0, n, g);
  mcopy(GGN, GLA, n);
  INFO[0] = gauss_newton(GGN, BX, BY, n, m, g, gn_iters);
  return 1;
}

/* ---------------------------------------------------------------- diagnostics */

static double residual(const double *G, const double *X, const double *Y, int n, int m) {
  double s = 0;
  for (int i = 0; i < n; i++) for (int k = 0; k < m; k++) {
    double r = Y[i * m + k];
    for (int j = 0; j < n; j++) r -= G[i * n + j] * X[j * m + k];
    s += r * r;
  }
  return sqrt_(s);
}

static double det(const double *A, int n) {
  double M[NN]; mcopy(M, A, n);
  double d = 1;
  for (int c = 0; c < n; c++) {
    int p = c;
    for (int r = c + 1; r < n; r++) if (fabs_(M[r * n + c]) > fabs_(M[p * n + c])) p = r;
    if (M[p * n + c] == 0) return 0;
    if (p != c) { d = -d; for (int k = 0; k < n; k++) { double t = M[c * n + k]; M[c * n + k] = M[p * n + k]; M[p * n + k] = t; } }
    d *= M[c * n + c];
    for (int r = c + 1; r < n; r++) { double f = M[r * n + c] / M[c * n + c]; for (int k = c; k < n; k++) M[r * n + k] -= f * M[c * n + k]; }
  }
  return d;
}

/* How far G is from the group: ||G^T M G - M||_F for the defining form M, |det G - 1| for SL. */
static double defect(const double *G, int n, int g) {
  double Mf[NN], T[NN], U[NN];
  if (g == GRP_GL) return 0;
  if (g == GRP_SL) return fabs_(det(G, n) - 1);
  if (g == GRP_SE) {
    double R[NN]; int d = n - 1; double s = 0;
    for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) R[i * d + j] = G[i * n + j];
    mtrans(T, R, d); mmul(U, T, R, d);
    for (int i = 0; i < d; i++) U[i * d + i] -= 1;
    for (int j = 0; j < n; j++) { double e = G[d * n + j] - (j == d); s += e * e; }
    return sqrt_(normF(U, d * d) * normF(U, d * d) + s);
  }
  for (int i = 0; i < n * n; i++) Mf[i] = 0;
  if (g == GRP_SO || g == GRP_U) meye(Mf, n);
  if (g == GRP_LORENTZ) { Mf[0] = -1; for (int i = 1; i < n; i++) Mf[i * n + i] = 1; }
  if (g == GRP_SP) { int h = n / 2; for (int i = 0; i < h; i++) { Mf[i * n + i + h] = 1; Mf[(i + h) * n + i] = -1; } }
  mtrans(T, G, n); mmul(T, T, Mf, n); mmul(T, T, G, n);
  for (int i = 0; i < n * n; i++) T[i] -= Mf[i];
  return normF(T, n * n);
}

/* INFO = [gn iters, then for G0, GLA, GGN: rel error, residual, defect] */
EXPORT void diagnostics(int n, int m, int g) {
  const double *Gs[3] = {G0, GLA, GGN};
  double D[NN], nt = normF(GT, n * n);
  for (int e = 0; e < 3; e++) {
    for (int i = 0; i < n * n; i++) D[i] = Gs[e][i] - GT[i];
    INFO[1 + 3 * e] = normF(D, n * n) / nt;
    INFO[2 + 3 * e] = residual(Gs[e], BX, BY, n, m);
    INFO[3 + 3 * e] = defect(Gs[e], n, g);
  }
  INFO[10] = residual(GT, BX, BY, n, m);
}

/* Eigenvalues of a matrix in buffer `which` (0 = Gt, 1 = G0, 2 = GLA, 3 = GGN):
 * characteristic polynomial by Faddeev-LeVerrier, roots by Durand-Kerner. Writes EIG (re, im pairs). */
EXPORT void eigenvalues(int which, int n) {
  const double *A = which == 0 ? GT : which == 1 ? G0 : which == 2 ? GLA : GGN;
  double M[NN], AM[NN], c[MAXN + 1];
  c[n] = 1; meye(M, n);
  for (int k = 1; k <= n; k++) {
    mmul(AM, A, M, n);
    double tr = 0; for (int i = 0; i < n; i++) tr += AM[i * n + i];
    c[n - k] = -tr / k;
    mcopy(M, AM, n);
    for (int i = 0; i < n; i++) M[i * n + i] += c[n - k];
  }
  double zr[MAXN], zi[MAXN];
  for (int i = 0; i < n; i++) { double s, co; sincos_(0.4 + 2 * PI * i / n, &s, &co); zr[i] = 1.1 * co; zi[i] = 1.1 * s; }
  for (int it = 0; it < 500; it++) {
    double moved = 0;
    for (int i = 0; i < n; i++) {
      /* p(z) by Horner */
      double pr = 1, pi = 0;
      for (int k = n - 1; k >= 0; k--) { double t = pr * zr[i] - pi * zi[i] + c[k]; pi = pr * zi[i] + pi * zr[i]; pr = t; }
      double qr = 1, qi = 0;
      for (int j = 0; j < n; j++) if (j != i) {
        double dr = zr[i] - zr[j], di = zi[i] - zi[j];
        double t = qr * dr - qi * di; qi = qr * di + qi * dr; qr = t;
      }
      double den = qr * qr + qi * qi; if (den == 0) den = 1e-300;
      double wr = (pr * qr + pi * qi) / den, wi = (pi * qr - pr * qi) / den;
      zr[i] -= wr; zi[i] -= wi; moved += wr * wr + wi * wi;
    }
    if (moved < 1e-30) break;
  }
  for (int i = 0; i < n; i++) { EIG[2 * i] = zr[i]; EIG[2 * i + 1] = zi[i]; }
}

EXPORT double angle(double y, double x) { return atan2_(y, x); }

/* ---------------------------------------------------------------- data generators */

static void rand_group(double *G, int n, int g, double scale) {
  double A[NN];
  for (int i = 0; i < n * n; i++) A[i] = scale * randn();
  proj(A, A, n, g);
  expm(G, A, n);
}

/* Y = GT X + sigma * noise for the GT and X already in the buffers. */
EXPORT void noisy_outputs(int n, int m, int g, double sigma) {
  int rows = g == GRP_SE ? n - 1 : n;
  for (int i = 0; i < n; i++) for (int k = 0; k < m; k++) {
    double s = 0; for (int j = 0; j < n; j++) s += GT[i * n + j] * BX[j * m + k];
    BY[i * m + k] = s + (i < rows ? sigma * randn() : 0);
  }
}

/* Random group element GT and Y = GT X + sigma * noise for the X already in BX. */
EXPORT void gen_from_X(int n, int m, int g, double sigma, double scale) {
  rand_group(GT, n, g, scale);
  noisy_outputs(n, m, g, sigma);
}

/* Condition number of the input data: row i of X is scaled by ANISO^(i / (rows - 1)),
 * so ANISO = 1 gives isotropic data, X X^T close to m I. */
static double ANISO = 1;
EXPORT void set_aniso(double a) { ANISO = a; }
static int ISO_EXACT = 0;
EXPORT void set_isotropic(int on) { ISO_EXACT = on; }

/* Random group element GT, Gaussian data X with m columns, Y = GT X + sigma * noise. */
EXPORT void gen_data(int n, int m, int g, double sigma, double scale) {
  int rows = g == GRP_SE ? n - 1 : n;
  for (int i = 0; i < n; i++) {
    double f = rows > 1 && i < rows ? exp_(log_(ANISO) * i / (rows - 1)) : 1;
    for (int k = 0; k < m; k++) BX[i * m + k] = i < rows ? f * randn() : 1;
  }
  if (ISO_EXACT) {
    /* Gram-Schmidt on the rows, so X X^T = m I exactly (for SE, on the centred rows) */
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < i; j++) {
        double d = 0; for (int k = 0; k < m; k++) d += BX[i * m + k] * BX[j * m + k];
        for (int k = 0; k < m; k++) BX[i * m + k] -= d / m * BX[j * m + k];
      }
      if (g == GRP_SE) { double mu = 0; for (int k = 0; k < m; k++) mu += BX[i * m + k]; for (int k = 0; k < m; k++) BX[i * m + k] -= mu / m; }
      double nrm = 0; for (int k = 0; k < m; k++) nrm += BX[i * m + k] * BX[i * m + k];
      nrm = sqrt_(nrm / m);
      for (int k = 0; k < m; k++) BX[i * m + k] /= nrm;
    }
  }
  gen_from_X(n, m, g, sigma, scale);
}

/* Lorentz transformation from frame A to frame B: a boost with rapidity eta along the direction
 * (theta, phi), followed by a rotation by angle psi about the z axis. Coordinates (E, px, py, pz). */
EXPORT void make_lorentz(double eta, double theta, double phi, double psi) {
  double K[16] = {0}, R[16] = {0}, B[16], Q[16], st, ct, sp, cp;
  sincos_(theta, &st, &ct); sincos_(phi, &sp, &cp);
  double d[3] = {st * cp, st * sp, ct};
  for (int i = 0; i < 3; i++) { K[i + 1] = eta * d[i]; K[(i + 1) * 4] = eta * d[i]; }
  R[1 * 4 + 2] = -psi; R[2 * 4 + 1] = psi;
  expm(B, K, 4); expm(Q, R, 4);
  mmul(GT, Q, B, 4);
}

/* Particles measured by two observers. Each column of X is a four-momentum (E, p) in frame A,
 * with a mass drawn from {electron, muon, pion, kaon, proton} and momentum of about p_scale GeV.
 * Y = GT X in frame B. Both frames add Gaussian noise of sigma GeV to every component. */
static const double MASSES[5] = {0.000511, 0.10566, 0.13957, 0.49368, 0.93827};
EXPORT void gen_particles(int m, double sigma, double p_scale) {
  const int n = 4;
  for (int k = 0; k < m; k++) {
    double mass = MASSES[(int)(urand() * 5) % 5], p2 = 0;
    for (int i = 1; i < 4; i++) { double v = p_scale * randn(); BX[i * m + k] = v; p2 += v * v; }
    BX[k] = sqrt_(mass * mass + p2);
  }
  for (int i = 0; i < n; i++) for (int k = 0; k < m; k++) {
    double s = 0; for (int j = 0; j < n; j++) s += GT[i * n + j] * BX[j * m + k];
    BY[i * m + k] = s + sigma * randn();
  }
  for (int i = 0; i < n * m; i++) BX[i] += sigma * randn();
}

/* Self tests of the building blocks. Results go to INFO.
 * test_projection: INFO[0] = ||P(P L) - P L||, INFO[1] = |<L - P L, P L>|, INFO[2] = distance of exp(P L) from the group
 * test_explog:     INFO[0] = ||exp(log g) - g|| / ||g||, INFO[1] = ||proj(log g) - log g|| (the log stays in the algebra) */
EXPORT void test_projection(int n, int g) {
  double L[NN], P[NN], PP[NN], E[NN], dot = 0;
  for (int i = 0; i < n * n; i++) L[i] = randn();
  proj(P, L, n, g); proj(PP, P, n, g);
  for (int i = 0; i < n * n; i++) { dot += (L[i] - P[i]) * P[i]; PP[i] -= P[i]; }
  INFO[0] = normF(PP, n * n); INFO[1] = fabs_(dot);
  for (int i = 0; i < n * n; i++) P[i] *= 0.5;
  expm(E, P, n);
  INFO[2] = defect(E, n, g);
}
EXPORT void test_explog(int n, int g, double scale) {
  double G[NN], L[NN], P[NN], E[NN];
  rand_group(G, n, g, scale);
  logm(L, G, n); expm(E, L, n); proj(P, L, n, g);
  for (int i = 0; i < n * n; i++) { E[i] -= G[i]; P[i] -= L[i]; }
  INFO[0] = normF(E, n * n) / normF(G, n * n); INFO[1] = normF(P, n * n);
}

/* TRACK = G X for matrix `which` (0 = Gt, 1 = G0, 2 = GLA, 3 = GGN), used to draw transformed point clouds. */
EXPORT void transform(int which, int n, int m) {
  const double *G = which == 0 ? GT : which == 1 ? G0 : which == 2 ? GLA : GGN;
  for (int i = 0; i < n; i++) for (int k = 0; k < m; k++) {
    double s = 0; for (int j = 0; j < n; j++) s += G[i * n + j] * BX[j * m + k];
    TRACK[i * m + k] = s;
  }
}

/* Storage ring: one-turn map M = A R A^{-1} in Sp(4), coordinates (x, y, px, py),
 * turn-by-turn BPM data with relative noise sigma. X, Y hold consecutive turns. */
/* Quantum gate identification. A two-qubit gate U (4 x 4 complex) acts on m random probe states,
 * and the outputs are measured with complex Gaussian noise of size sigma per amplitude.
 * Each complex column x = a + ib is stored as the two real columns [a; b] and [-b; a] (that is, x and ix),
 * so the real 8 x 8 least squares fit is exactly the embedding of the complex one.
 * gate: 0 sqrt(SWAP), 1 controlled-S, 2 sqrt(iSWAP), 3 random U(4), 4 CNOT */
EXPORT void gen_quantum(int gate, int m, double sigma) {
  const int k = 4, n = 8;
  double Ur[16] = {0}, Ui[16] = {0};
  const double r = 0.7071067811865476;
  if (gate == 3) {
    double G[NN];
    rand_group(G, n, GRP_U, 1.0);
    for (int i = 0; i < k; i++) for (int j = 0; j < k; j++) { Ur[i * k + j] = G[i * n + j]; Ui[i * k + j] = G[(i + k) * n + j]; }
  } else {
    Ur[0] = 1; Ur[15] = 1;
    if (gate == 0) { Ur[5] = Ur[10] = 0.5; Ui[5] = Ui[10] = 0.5; Ur[6] = Ur[9] = 0.5; Ui[6] = Ui[9] = -0.5; }
    if (gate == 1) { Ur[5] = Ur[10] = 1; Ur[15] = 0; Ui[15] = 1; }
    if (gate == 2) { Ur[5] = Ur[10] = r; Ui[6] = Ui[9] = r; }
    if (gate == 4) { Ur[5] = 1; Ur[11] = Ur[14] = 1; Ur[15] = 0; }
  }
  for (int i = 0; i < k; i++) for (int j = 0; j < k; j++) {
    GT[i * n + j] = GT[(i + k) * n + j + k] = Ur[i * k + j];
    GT[(i + k) * n + j] = Ui[i * k + j]; GT[i * n + j + k] = -Ui[i * k + j];
  }
  int M = 2 * m;
  for (int p = 0; p < m; p++) {
    double xr[4], xi[4], yr[4], yi[4], nrm = 0;
    for (int i = 0; i < k; i++) { xr[i] = randn(); xi[i] = randn(); nrm += xr[i] * xr[i] + xi[i] * xi[i]; }
    nrm = sqrt_(nrm);
    for (int i = 0; i < k; i++) { xr[i] /= nrm; xi[i] /= nrm; }
    for (int i = 0; i < k; i++) {
      double a = 0, b = 0;
      for (int j = 0; j < k; j++) { a += Ur[i * k + j] * xr[j] - Ui[i * k + j] * xi[j]; b += Ur[i * k + j] * xi[j] + Ui[i * k + j] * xr[j]; }
      yr[i] = a + sigma * randn() * r; yi[i] = b + sigma * randn() * r;
    }
    for (int i = 0; i < k; i++) {
      BX[i * M + 2 * p] = xr[i]; BX[(i + k) * M + 2 * p] = xi[i];
      BX[i * M + 2 * p + 1] = -xi[i]; BX[(i + k) * M + 2 * p + 1] = xr[i];
      BY[i * M + 2 * p] = yr[i]; BY[(i + k) * M + 2 * p] = yi[i];
      BY[i * M + 2 * p + 1] = -yi[i]; BY[(i + k) * M + 2 * p + 1] = yr[i];
    }
  }
}

/* Gate fidelity |tr(U^dagger V)| / k between GT and matrix `which`, for embedded k x k complex matrices. */
EXPORT double gate_fidelity(int which, int n) {
  const double *V = which == 0 ? GT : which == 1 ? G0 : which == 2 ? GLA : GGN;
  int k = n / 2; double re = 0, im = 0;
  for (int i = 0; i < k; i++) for (int j = 0; j < k; j++) {
    double ur = GT[i * n + j], ui = GT[(i + k) * n + j], vr = V[i * n + j], vi = V[(i + k) * n + j];
    re += ur * vr + ui * vi; im += ur * vi - ui * vr;
  }
  return sqrt_(re * re + im * im) / k;
}

EXPORT void gen_accel(double Qx, double Qy, double coupling, int turns, double sigma) {
  const int n = 4;
  double R[16] = {0}, N[16] = {0}, C[16], A[16], Ai[16], T[16];
  double s, c;
  sincos_(2 * PI * Qx, &s, &c); R[0] = c; R[2] = s; R[8] = -s; R[10] = c;
  sincos_(2 * PI * Qy, &s, &c); R[5] = c; R[7] = s; R[13] = -s; R[15] = c;
  double bx = 12.0, ax = -1.1, by = 7.0, ay = 0.6;
  N[0] = sqrt_(bx); N[8] = -ax / sqrt_(bx); N[10] = 1 / sqrt_(bx);
  N[5] = sqrt_(by); N[13] = -ay / sqrt_(by); N[15] = 1 / sqrt_(by);
  rand_group(C, n, GRP_SP, coupling);
  mmul(A, C, N, n); minv(Ai, A, n);
  mmul(T, A, R, n); mmul(GT, T, Ai, n);

  int m = turns;
  double z[4] = {1e-3, 0.7e-3, 0, 0}, mean[4] = {0}, sq[4] = {0};
  for (int k = 0; k <= m; k++) {
    for (int i = 0; i < 4; i++) {
      TRACK[i * (m + 1) + k] = z[i]; mean[i] += z[i]; sq[i] += z[i] * z[i];
    }
    double w[4];
    for (int i = 0; i < 4; i++) { w[i] = 0; for (int j = 0; j < 4; j++) w[i] += GT[i * 4 + j] * z[j]; }
    for (int i = 0; i < 4; i++) z[i] = w[i];
  }
  for (int i = 0; i < 4; i++) {
    double mu = mean[i] / (m + 1), sd = sqrt_(sq[i] / (m + 1) - mu * mu);
    for (int k = 0; k <= m; k++) TRACK[i * (m + 1) + k] += sigma * sd * randn();
  }
  for (int i = 0; i < 4; i++) for (int k = 0; k < m; k++) {
    BX[i * m + k] = TRACK[i * (m + 1) + k];
    BY[i * m + k] = TRACK[i * (m + 1) + k + 1];
  }
}

/* Track an initial condition through `turns` applications of matrix `which`. Writes TRACK (n x turns). */
EXPORT void track(int which, int n, int turns, double z0, double z1, double z2, double z3) {
  const double *G = which == 0 ? GT : which == 1 ? G0 : which == 2 ? GLA : GGN;
  double z[MAXN] = {z0, z1, z2, z3}, w[MAXN];
  for (int k = 0; k < turns; k++) {
    for (int i = 0; i < n; i++) TRACK[i * turns + k] = z[i];
    for (int i = 0; i < n; i++) { w[i] = 0; for (int j = 0; j < n; j++) w[i] += G[i * n + j] * z[j]; }
    for (int i = 0; i < n; i++) z[i] = w[i];
  }
}

/* Monte Carlo: `trials` independent problems; ERRS holds (ls, la, gn) relative errors.
 * gn_iters = 0 skips Gauss-Newton. Same model as the WebGPU kernel. */
EXPORT void monte_carlo(int n, int m, int g, double sigma, double scale, int trials, int gn_iters) {
  if (trials > MAXTRIALS) trials = MAXTRIALS;
  double D[NN];
  for (int t = 0; t < trials; t++) {
    gen_data(n, m, g, sigma, scale);
    double nt = normF(GT, n * n);
    if (!least_squares(G0, BX, BY, n, m)) { ERRS[3 * t] = ERRS[3 * t + 1] = ERRS[3 * t + 2] = -1; continue; }
    lie_algebra(GLA, G0, n, g);
    for (int i = 0; i < n * n; i++) D[i] = G0[i] - GT[i];
    ERRS[3 * t] = (float)(normF(D, n * n) / nt);
    for (int i = 0; i < n * n; i++) D[i] = GLA[i] - GT[i];
    ERRS[3 * t + 1] = (float)(normF(D, n * n) / nt);
    if (gn_iters > 0) {
      mcopy(GGN, GLA, n);
      gauss_newton(GGN, BX, BY, n, m, g, gn_iters);
      for (int i = 0; i < n * n; i++) D[i] = GGN[i] - GT[i];
      ERRS[3 * t + 2] = (float)(normF(D, n * n) / nt);
    } else ERRS[3 * t + 2] = 0;
  }
}
