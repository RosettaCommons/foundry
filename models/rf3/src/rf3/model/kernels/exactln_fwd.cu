// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Anthropic, PBC. See NOTICE and LICENSE.
// exactln_fwd.cu — a bit-for-bit replica of ATen's CUDA layer_norm forward for fp32 (and bf16) rows of width C (C % 4 == 0, C <= 1024),
// scheduled one WARP per row (two rows per warp at C <= 64), many rows per CTA, rows read through arbitrary leading strides.
//
// What is replicated (torch 2.12.0, aten/src/ATen/native/cuda/layer_norm_kernel.cu, `vectorized_layer_norm_kernel<T, float>` — the kernel
// `LayerNormKernelImplInternal` launches when T is float/half/bf16, N % 4 == 0, N <= 2^24 and X/Y/gamma/beta are 16-byte (T=float) aligned;
// launch config `threads = (32, 4)`, one CTA per row (`launch_vectorized_layer_norm_kernel`); its fp32 arithmetic follows that source
// (vectorized_layer_norm_kernel<float, float>) as nvcc compiles it for sm_90 in torch 2.12.0+cu130:
//
//   virtual thread t in [0,128) (thrx = threadIdx.x + 32*threadIdx.y) reads the float4 vectors v = t, t+128, ... of the row in that order and runs
//   the ONLINE Welford update per element (cuWelfordOnlineSum):
//       new_count = count + 1.0f                (FADD)
//       delta     = val - mean                  (FADD)
//       coef      = rcp_rn(new_count)           (MUFU.RCP + Newton step + slow path == IEEE round-to-nearest reciprocal, i.e. __frcp_rn)
//       mean      = fma(delta, coef, mean)      (FFMA — the compiler contracted `mean + delta * coef`)
//       sigma2    = fma(delta, val - mean, sigma2)   (FADD for val - NEW mean, then FFMA)
//   then the warp tree `for offset in 16,8,4,2,1: wd = cuWelfordCombine(dataB = wd, dataA = shfl_down(wd, offset))` where the combine is
//       count = A.count + B.count               (FADD)
//       if (count > 0):                          (compiled form: B.count > -A.count — the same predicate for finite non-negative counts)
//         delta = B.mean - A.mean                (FADD)
//         coef  = rcp_rn(count)
//         nB    = coef * B.count                 (FMUL)
//         nA    = A.count * coef                 (FMUL)
//         mean  = fma(A.mean, nA, nB * B.mean)   (FMUL, FFMA)
//         sigma2 = fma(nB, A.count * (delta * delta), A.sigma2 + B.sigma2)   (FMUL, FMUL, FADD, FFMA)
//       else: mean = 0, sigma2 = 0
//   then the shared-memory tree over the 4 warps (blockDim.y = 4 > 1): W0 = C(W0, W2); W1 = C(W1, W3); W0 = C(W0, W1) with the same combine —
//   for C <= 512 the warps 1..3 hold data only when C > 128, but the combines with EMPTY partials (0,0,0) are still executed (they are identities
//   except for inf/NaN/-0.0 corner cases, so they are executed here too);
//   var  = div_rn(sigma2, float(C))            (IEEE division: MUFU.RCP + FFMA refinement + FCHK slow path)
//   rstd = rsqrtf(var + eps)                    (FADD; MUFU.RSQ behind the non-FTZ denormal scaling — `rsqrt.approx.f32`)
//   y    = fma(rstd * (x - mean), gamma, beta)  (FADD, FMUL, FFMA)   [gamma only: (rstd*(x-mean))*gamma; no affine: rstd*(x-mean)]
//   mean/rstd of the row are lane (0,0)'s values; eps is the double 1e-5 cast to float on the host (static_cast<T_ACC>(eps)).
//
// Every fp32 operation below is written with an explicit round-to-nearest intrinsic so that no compiler flag (fmad / contraction) can change it;
// compile WITHOUT --use_fast_math / --ftz (rsqrtf must keep its non-FTZ expansion, denormals must survive FADD/FMUL/FFMA as in ATen's build).
// The schedule (which lane computes which row, how many rows per CTA, strided row gather) changes no bits: each output element's arithmetic is the
// sequence above on the same operands.
//
// Layout served: row r of `rows`; the row's first element is X + (r / inner) * stride_outer + (r % inner) * stride_inner (element strides, 64-bit
// products), its C elements contiguous; Y is written contiguous (row r at Y + r*C) — exactly what layer_norm_cuda produces (it copies a
// non-contiguous input to a contiguous buffer first and allocates Y LEGACY_CONTIGUOUS).  TIN/TOUT: float or __nv_bfloat16 storage; the arithmetic
// is fp32 either way (acc_type<T> = float; bf16 -> float is exact; the bf16 store is ATen's round-to-nearest-even `static_cast<T>`).

namespace exactln {

struct WD { float mean, sigma2, count; };

__device__ __forceinline__ WD wd_empty() { WD w; w.mean = 0.f; w.sigma2 = 0.f; w.count = 0.f; return w; }

// cuWelfordOnlineSum<float, /*rms_norm=*/false> (same source file) as compiled for sm_90, torch 2.12.0+cu130
__device__ __forceinline__ WD online(WD w, float val) {
  const float new_count = __fadd_rn(w.count, 1.0f);
  const float delta = __fsub_rn(val, w.mean);
  const float coef = __frcp_rn(new_count);
  const float new_mean = __fmaf_rn(delta, coef, w.mean);
  const float diff = __fsub_rn(val, new_mean);
  WD r;
  r.mean = new_mean;
  r.sigma2 = __fmaf_rn(delta, diff, w.sigma2);
  r.count = new_count;
  return r;
}

// cuWelfordCombine<false>(dataB = self, dataA = other) as compiled
__device__ __forceinline__ WD combine(WD b, WD a) {
  WD r;
  const float count = __fadd_rn(a.count, b.count);
  r.count = count;
  if (count > 0.f) {
    const float delta = __fsub_rn(b.mean, a.mean);
    const float coef = __frcp_rn(count);
    const float nB = __fmul_rn(coef, b.count);
    const float nA = __fmul_rn(a.count, coef);
    r.mean = __fmaf_rn(a.mean, nA, __fmul_rn(nB, b.mean));
    const float d2 = __fmul_rn(delta, delta);
    const float t = __fmul_rn(a.count, d2);
    r.sigma2 = __fmaf_rn(nB, t, __fadd_rn(a.sigma2, b.sigma2));
  } else {
    r.mean = 0.f;
    r.sigma2 = 0.f;
  }
  return r;
}

template <int WIDTH>
__device__ __forceinline__ WD shfl_down_wd(WD w, int offset) {
  WD o;
  o.mean = __shfl_down_sync(0xffffffffu, w.mean, offset, WIDTH);
  o.sigma2 = __shfl_down_sync(0xffffffffu, w.sigma2, offset, WIDTH);
  o.count = __shfl_down_sync(0xffffffffu, w.count, offset, WIDTH);
  return o;
}

// bf16 storage without cuda_bf16.h (NVRTC has no include path here): the upper 16 bits of an fp32
struct bf16_t { unsigned short u; };
template <typename T> struct Vec4;
template <> struct alignas(16) Vec4<float> { float v[4]; };
template <> struct alignas(8) Vec4<bf16_t> { bf16_t v[4]; };

__device__ __forceinline__ float to_f32(float x) { return x; }
__device__ __forceinline__ float to_f32(bf16_t x) { return __uint_as_float(((unsigned int)x.u) << 16); }
template <typename T> __device__ __forceinline__ T from_f32(float x);
template <> __device__ __forceinline__ float from_f32<float>(float x) { return x; }
template <> __device__ __forceinline__ bf16_t from_f32<bf16_t>(float x) {
  // round-to-nearest-even fp32 -> bf16, NaN kept quiet (c10::BFloat16(float) / __float2bfloat16_rn semantics)
  unsigned int u = __float_as_uint(x);
  bf16_t r;
  if ((u & 0x7fffffffu) > 0x7f800000u) { r.u = (unsigned short)((u >> 16) | 0x0040u); return r; }
  const unsigned int lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  r.u = (unsigned short)(u >> 16);
  return r;
}

// AFFINE: 0 = none, 1 = gamma only, 2 = beta only, 3 = gamma and beta (ATen's four branches)
// C: row width (multiple of 4, <= 1024).  ROWS_PER_WARP: 1 (C >= 128) or 2 (C <= 64: lanes 0-15 / 16-31 hold one row each; ATen's data lanes
// of a 64-wide row are lanes 0..15 of warp 0, whose first tree step combines each with an EMPTY lane 16..31 partner — done explicitly here).
template <int C, int ROWS_PER_WARP>
struct Geo {
  static_assert(C % 4 == 0 && C >= 4 && C <= 1024, "C must be a multiple of 4 in [4, 1024]");
  static_assert(ROWS_PER_WARP == 1 || (ROWS_PER_WARP == 2 && C <= 64), "two rows per warp only for C <= 64");
  static constexpr int NVEC = C / 4;                 // float4 vectors per row (n_vec_to_read)
  static constexpr int LANES = 32 / ROWS_PER_WARP;   // lanes serving one row
  static constexpr int KMAX = (NVEC + 127) / 128;    // vectors per virtual thread (1 for C <= 512, 2 for C <= 1024); virtual thread t = 32*w + vl reads vectors t + 128*k
  __device__ static __forceinline__ bool has(int w, int k, int vl) { return (32 * w < NVEC) && (ROWS_PER_WARP == 1 || w == 0) && (128 * k + 32 * w + vl < NVEC); }
};

// gamma / beta: this lane's vectors (fp32 values; a bf16 gamma is widened exactly like static_cast<T_ACC>)
template <typename TPAR, int C, int AFFINE, int RPW>
__device__ __forceinline__ void load_params(const TPAR* __restrict__ G, const TPAR* __restrict__ Bt, int vl,
                                            float (&g)[4][Geo<C, RPW>::KMAX][4], float (&bt)[4][Geo<C, RPW>::KMAX][4]) {
  using GE = Geo<C, RPW>;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
#pragma unroll
    for (int k = 0; k < GE::KMAX; ++k) {
      const int v = 128 * k + 32 * w + vl;
#pragma unroll
      for (int j = 0; j < 4; ++j) { g[w][k][j] = 1.f; bt[w][k][j] = 0.f; }
      if (GE::has(w, k, vl)) {
        if (AFFINE & 1) { Vec4<TPAR> gv = reinterpret_cast<const Vec4<TPAR>*>(G)[v];
#pragma unroll
          for (int j = 0; j < 4; ++j) g[w][k][j] = to_f32(gv.v[j]); }
        if (AFFINE & 2) { Vec4<TPAR> bv = reinterpret_cast<const Vec4<TPAR>*>(Bt)[v];
#pragma unroll
          for (int j = 0; j < 4; ++j) bt[w][k][j] = to_f32(bv.v[j]); }
      }
    }
  }
}

// one row's load into registers: x[w][k][j] = fp32 value of vector (128k + 32w + vl), element j (zeros where the lane holds no data)
template <typename TIN, int C, int RPW>
__device__ __forceinline__ void load_row(const TIN* __restrict__ xrow, int vl, float (&x)[4][Geo<C, RPW>::KMAX][4]) {
  using GE = Geo<C, RPW>;
  const Vec4<TIN>* xv = reinterpret_cast<const Vec4<TIN>*>(xrow);
#pragma unroll
  for (int w = 0; w < 4; ++w) {
#pragma unroll
    for (int k = 0; k < GE::KMAX; ++k) {
      const int v = 128 * k + 32 * w + vl;
      if (GE::has(w, k, vl)) {
        Vec4<TIN> d = xv[v];
#pragma unroll
        for (int j = 0; j < 4; ++j) x[w][k][j] = to_f32(d.v[j]);
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) x[w][k][j] = 0.f;
      }
    }
  }
}

// ATen's layer_norm of the row held in registers (the online sums, the (32,4)-CTA trees, var, rstd, the affine pass), stored to yrow when `live`.
// All 32 lanes of the warp call this (full-mask shuffles); `sub` selects which of the warp's ROWS_PER_WARP rows this lane serves.
template <typename TOUT, int C, int AFFINE, int RPW>
__device__ __forceinline__ void ln_row(const float (&x)[4][Geo<C, RPW>::KMAX][4], const float (&g)[4][Geo<C, RPW>::KMAX][4],
                                       const float (&bt)[4][Geo<C, RPW>::KMAX][4], int vl, int sub, float eps, bool live, TOUT* __restrict__ yrow) {
  using GE = Geo<C, RPW>;
  WD part[4];
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    part[w] = wd_empty();
#pragma unroll
    for (int k = 0; k < GE::KMAX; ++k) {
      if (GE::has(w, k, vl)) {
#pragma unroll
        for (int j = 0; j < 4; ++j) part[w] = online(part[w], x[w][k][j]);
      }
    }
  }
  // the warp trees (virtual warps that hold no data reduce to EMPTY exactly and are skipped)
  WD R[4];
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    if (32 * w < GE::NVEC && (RPW == 1 || w == 0)) {
      WD wd = part[w];
      if (RPW == 1) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) wd = combine(wd, shfl_down_wd<32>(wd, off));
      } else {
        // C <= 64: ATen's offset-16 partner of every data lane is an EMPTY lane; then offsets 8..1 among the 16 lanes of this row
        wd = combine(wd, wd_empty());
#pragma unroll
        for (int off = GE::LANES / 2; off > 0; off >>= 1) wd = combine(wd, shfl_down_wd<GE::LANES>(wd, off));
      }
      R[w] = wd;
    } else {
      R[w] = wd_empty();
    }
  }
  // the block tree of the (32,4) CTA: offset 2 then offset 1 (virtual lane 0's values are the row's; every lane computes the same expressions)
  WD W0 = combine(R[0], R[2]);
  WD W1 = combine(R[1], R[3]);
  W0 = combine(W0, W1);
  // thread (0,0): meansigmabuf = {mean, sigma2 / float(N)}; broadcast from the row's virtual lane 0
  const int src = (RPW == 1) ? 0 : (sub * GE::LANES);
  const float mean = __shfl_sync(0xffffffffu, W0.mean, src);
  const float sigma2 = __shfl_sync(0xffffffffu, W0.sigma2, src);
  const float var = __fdiv_rn(sigma2, static_cast<float>(C));
  const float rstd = rsqrtf(__fadd_rn(var, eps));
  if (!live) return;
  Vec4<TOUT>* yv = reinterpret_cast<Vec4<TOUT>*>(yrow);
#pragma unroll
  for (int w = 0; w < 4; ++w) {
#pragma unroll
    for (int k = 0; k < GE::KMAX; ++k) {
      const int v = 128 * k + 32 * w + vl;
      if (GE::has(w, k, vl)) {
        Vec4<TOUT> out;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float u = __fmul_rn(rstd, __fsub_rn(x[w][k][j], mean));
          float y;
          if (AFFINE == 3)      y = __fmaf_rn(u, g[w][k][j], bt[w][k][j]);
          else if (AFFINE == 1) y = __fmul_rn(u, g[w][k][j]);
          else if (AFFINE == 2) {
            // ATen's beta-only branch as nvcc compiled it for sm_90: element 0 of each float4 is FMUL then FADD (two roundings), elements
            // 1..3 are the contracted FFMA rstd*(x-mean)+beta (one rounding), as nvcc contracts that source line for sm_90.
            if (j == 0) y = __fadd_rn(u, bt[w][k][j]);
            else        y = __fmaf_rn(rstd, __fsub_rn(x[w][k][j], mean), bt[w][k][j]);
          }
          else                  y = u;
          out.v[j] = from_f32<TOUT>(y);
        }
        yv[v] = out;
      }
    }
  }
}

// a row index r in [0, rows) -> element offset of that row in a tensor whose rows form an (outer, inner) stride pair (plan()'s collapse)
__device__ __forceinline__ long long row_off(long long r, int inner, long long stride_outer, long long stride_inner) {
  const long long o = r / inner, i = r - o * inner;
  return o * stride_outer + i * stride_inner;
}

// TIN: x storage; TPAR: gamma/beta storage (== TIN for a plain call; float with a bf16 x for an autocast call, whose widening of x is fused
// into the load); TOUT: y storage (TIN for a plain call, float for the autocast call; bf16 with a float x = the caller's `.to(bfloat16)` fused).
// Row r of `rows`: read at X + row_off(r, inner, stride_outer, stride_inner), written contiguous at Y + r*C.
template <typename TIN, typename TPAR, typename TOUT, int C, int AFFINE, int ROWS_PER_WARP>
__global__ void __launch_bounds__(256) exactln_fwd(const TIN* __restrict__ X, const TPAR* __restrict__ G, const TPAR* __restrict__ Bt,
                                                   TOUT* __restrict__ Y, int rows, int inner, long long stride_outer, long long stride_inner,
                                                   double eps_d, int row_step) {
  using GE = Geo<C, ROWS_PER_WARP>;
  const float eps = static_cast<float>(eps_d);
  const int lane = threadIdx.x & 31;
  const int vl = (ROWS_PER_WARP == 1) ? lane : (lane & (GE::LANES - 1));   // virtual lane
  const int sub = (ROWS_PER_WARP == 1) ? 0 : (lane / GE::LANES);           // which of the warp's rows this lane serves
  const int warp_in_block = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const long long first = ((long long)blockIdx.x * warps_per_block + warp_in_block) * ROWS_PER_WARP + sub;

  float g[4][GE::KMAX][4], bt[4][GE::KMAX][4];
  load_params<TPAR, C, AFFINE, ROWS_PER_WARP>(G, Bt, vl, g, bt);

  // Lanes whose row is out of range still take part in the shuffles of their warp (full-mask shuffles): they run the loop on a clamped row and skip the store.
  const long long warp_first = first - sub;                      // the warp's first row (sub 0)
  for (long long rw = warp_first; rw < rows; rw += row_step) {
    const long long r_raw = rw + sub;
    const bool live = r_raw < rows;
    const long long r = live ? r_raw : (rows - 1);
    float x[4][GE::KMAX][4];
    load_row<TIN, C, ROWS_PER_WARP>(X + row_off(r, inner, stride_outer, stride_inner), vl, x);
    ln_row<TOUT, C, AFFINE, ROWS_PER_WARP>(x, g, bt, vl, sub, eps, live, Y + r * (long long)C);
  }
}

} // namespace exactln
