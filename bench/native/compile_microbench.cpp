// compile_microbench.cpp — pure-C++ de-risk for Emily M6.
//
// Question: on an Apple Silicon GPU, does `mlx::core::compile` speed up
// a transformer block (Qwen3-0.6B-shaped) enough to clear the 20% gate
// the PLAN sets? If not, no amount of BEAM integration can rescue M6.
//
// Method: build a single transformer block (RMSNorm, attention, RMSNorm,
// SwiGLU FFN) using plain MLX ops. Run it 1000 times after a 50-iteration
// warmup, both uncompiled and wrapped in `mlx::core::compile`. Report
// min/median/p95 per iteration and compiled/uncompiled speedup ratio.
//
// All weights and inputs are passed as tracer-input `array`s through the
// `std::vector<array>` arg of the compile lambda — matching how Emily's
// Phase-3 Defn walk will present operands (every Defn parameter is a
// traced input, not a compile-time constant).

#include <mlx/mlx.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace mx = mlx::core;

// --- Qwen3-0.6B-ish block dims ---------------------------------------
//
// Qwen3-0.6B: hidden 1024, 16 Q heads / 8 KV heads (GQA), head_dim 128,
// FFN intermediate 3072. GQA adds a broadcast step but the fusion
// surface (RMSNorm chains, SwiGLU, attention softmax neighbourhood) is
// what we want to measure; ungrouped attention with head_dim=64 captures
// the same op vocabulary at a cheaper iteration cost. Seq kept small
// so a single block fits one iteration per ~1ms — tight enough that
// 1000 iterations is fast to run yet long enough that fusion matters.
constexpr int kBatch = 1;
static int kSeq = 128;                    // overridable via --seq
constexpr int kHidden = 1024;
constexpr int kHeads = 16;
constexpr int kHeadDim = 64;             // kHeads * kHeadDim = 1024 = kHidden
constexpr int kInter = 2816;              // common SwiGLU intermediate

// Input vector layout for the block fn:
//   0: x           [batch, seq, hidden]
//   1: w_rms1      [hidden]
//   2: w_q         [hidden, hidden]
//   3: w_k         [hidden, hidden]
//   4: w_v         [hidden, hidden]
//   5: w_o         [hidden, hidden]
//   6: w_rms2      [hidden]
//   7: w_gate      [hidden, inter]
//   8: w_up        [hidden, inter]
//   9: w_down      [inter, hidden]
// Output: one array — the block output [batch, seq, hidden]

static mx::array rms_norm(const mx::array& x, const mx::array& w, float eps = 1e-6f) {
  // rms = sqrt(mean(x^2) + eps); out = x / rms * w
  auto sq = mx::square(x);
  auto m = mx::mean(sq, /*axis=*/-1, /*keepdims=*/true);
  auto inv_rms = mx::rsqrt(mx::add(m, mx::array(eps)));
  auto normed = mx::multiply(x, inv_rms);
  return mx::multiply(normed, w);
}

static mx::array silu(const mx::array& x) {
  return mx::multiply(x, mx::sigmoid(x));
}

static std::vector<mx::array> block(const std::vector<mx::array>& in) {
  const auto& x = in[0];
  const auto& w_rms1 = in[1];
  const auto& w_q = in[2];
  const auto& w_k = in[3];
  const auto& w_v = in[4];
  const auto& w_o = in[5];
  const auto& w_rms2 = in[6];
  const auto& w_gate = in[7];
  const auto& w_up = in[8];
  const auto& w_down = in[9];

  // --- Pre-attention norm + projections -----------------------------
  auto h = rms_norm(x, w_rms1);

  auto q = mx::matmul(h, w_q);
  auto k = mx::matmul(h, w_k);
  auto v = mx::matmul(h, w_v);

  // Reshape to [batch, seq, heads, head_dim] then transpose to
  // [batch, heads, seq, head_dim]
  auto reshape_heads = [](const mx::array& t) {
    return mx::transpose(
        mx::reshape(t, {kBatch, kSeq, kHeads, kHeadDim}),
        {0, 2, 1, 3});
  };
  q = reshape_heads(q);
  k = reshape_heads(k);
  v = reshape_heads(v);

  // --- Scaled dot-product attention ---------------------------------
  // scores = q @ k^T / sqrt(head_dim)
  // attn = softmax(scores)
  // out = attn @ v
  auto k_t = mx::transpose(k, {0, 1, 3, 2});
  auto scores = mx::matmul(q, k_t);
  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
  scores = mx::multiply(scores, mx::array(scale));
  auto attn = mx::softmax(scores, std::vector<int>{-1}, /*precise=*/false);
  auto ctx = mx::matmul(attn, v);

  // [batch, heads, seq, head_dim] -> [batch, seq, hidden]
  ctx = mx::reshape(mx::transpose(ctx, {0, 2, 1, 3}),
                    {kBatch, kSeq, kHidden});

  auto attn_out = mx::matmul(ctx, w_o);
  auto h1 = mx::add(x, attn_out);

  // --- Pre-FFN norm + SwiGLU ----------------------------------------
  auto h2 = rms_norm(h1, w_rms2);

  auto gate = silu(mx::matmul(h2, w_gate));
  auto up = mx::matmul(h2, w_up);
  auto ffn = mx::matmul(mx::multiply(gate, up), w_down);

  auto out = mx::add(h1, ffn);
  return {out};
}

// ---------------------------------------------------------------------

struct Stats {
  double min_ms;
  double median_ms;
  double p95_ms;
};

static Stats summarise(std::vector<double>& xs) {
  std::sort(xs.begin(), xs.end());
  size_t n = xs.size();
  Stats s;
  s.min_ms = xs.front();
  s.median_ms = xs[n / 2];
  s.p95_ms = xs[std::min(n - 1, static_cast<size_t>(n * 0.95))];
  return s;
}

static std::vector<mx::array> make_inputs() {
  // Distinct PRNG key per weight so we don't end up with a rank-deficient
  // weight tensor from any accidental reuse.
  auto key = mx::random::key(42);

  auto x      = mx::random::uniform(-1.0f, 1.0f, {kBatch, kSeq, kHidden}, mx::float32, key);
  auto w_rms1 = mx::random::uniform(0.5f, 1.5f, {kHidden}, mx::float32, key);
  auto w_q    = mx::random::uniform(-0.1f, 0.1f, {kHidden, kHidden}, mx::float32, key);
  auto w_k    = mx::random::uniform(-0.1f, 0.1f, {kHidden, kHidden}, mx::float32, key);
  auto w_v    = mx::random::uniform(-0.1f, 0.1f, {kHidden, kHidden}, mx::float32, key);
  auto w_o    = mx::random::uniform(-0.1f, 0.1f, {kHidden, kHidden}, mx::float32, key);
  auto w_rms2 = mx::random::uniform(0.5f, 1.5f, {kHidden}, mx::float32, key);
  auto w_gate = mx::random::uniform(-0.1f, 0.1f, {kHidden, kInter}, mx::float32, key);
  auto w_up   = mx::random::uniform(-0.1f, 0.1f, {kHidden, kInter}, mx::float32, key);
  auto w_down = mx::random::uniform(-0.1f, 0.1f, {kInter, kHidden}, mx::float32, key);

  std::vector<mx::array> in = {
      x, w_rms1, w_q, w_k, w_v, w_o, w_rms2, w_gate, w_up, w_down};

  // Materialise the initial values so first-iteration timing isn't
  // dominated by weight init + download.
  for (auto& a : in) mx::eval(a);
  mx::synchronize();
  return in;
}

template <typename Fn>
static Stats time_runs(Fn&& fn, int warmup, int iters) {
  // Warmup: includes the compile trace on the first call when fn is
  // the compiled closure.
  for (int i = 0; i < warmup; ++i) {
    auto out = fn();
    mx::eval(out[0]);
    mx::synchronize();
  }

  std::vector<double> samples;
  samples.reserve(iters);

  for (int i = 0; i < iters; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    auto out = fn();
    mx::eval(out[0]);
    mx::synchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    samples.push_back(ms);
  }

  return summarise(samples);
}

static void run_on_device(mx::Device::DeviceType dev_type, const char* label,
                          int warmup, int iters) {
  mx::set_default_device(mx::Device(dev_type));

  auto inputs = make_inputs();

  // Uncompiled baseline: call block() directly on each iteration.
  auto uncompiled_fn = [&inputs]() -> std::vector<mx::array> {
    return block(inputs);
  };

  // Compiled: wrap block in mx::compile. First call of the returned
  // closure does the trace; subsequent calls hit the fused tape.
  auto compiled_closure = mx::compile(block, /*shapeless=*/false);
  auto compiled_fn = [&compiled_closure, &inputs]() -> std::vector<mx::array> {
    return compiled_closure(inputs);
  };

  std::printf("\n=== %s ===\n", label);
  std::printf("  shape: x[%d,%d,%d], heads=%d x head_dim=%d, inter=%d\n",
              kBatch, kSeq, kHidden, kHeads, kHeadDim, kInter);
  std::printf("  warmup: %d iters, measured: %d iters\n\n", warmup, iters);

  auto uncompiled = time_runs(uncompiled_fn, warmup, iters);
  auto compiled = time_runs(compiled_fn, warmup, iters);

  std::printf("  uncompiled  min=%.3f ms  median=%.3f ms  p95=%.3f ms\n",
              uncompiled.min_ms, uncompiled.median_ms, uncompiled.p95_ms);
  std::printf("  compiled    min=%.3f ms  median=%.3f ms  p95=%.3f ms\n",
              compiled.min_ms, compiled.median_ms, compiled.p95_ms);

  double median_speedup = uncompiled.median_ms / compiled.median_ms;
  double min_speedup = uncompiled.min_ms / compiled.min_ms;
  std::printf("  speedup (median) = %.3fx    (min) = %.3fx\n",
              median_speedup, min_speedup);

  bool passes = median_speedup >= 1.20;
  std::printf("  gate (>=1.20x on median): %s\n", passes ? "PASS" : "FAIL");
}

// ---------------------------------------------------------------------
// Sanity check: pure elementwise chain that is maximally fusable.
// If compile doesn't win here, the benchmark harness itself is broken.
// ---------------------------------------------------------------------

constexpr int kSanitySize = 1 << 20;  // 1M elements, f32 -> 4 MB each

static std::vector<mx::array> sanity_fn(const std::vector<mx::array>& in) {
  const auto& a = in[0];
  const auto& b = in[1];
  const auto& c = in[2];
  // 8 chained elementwise ops — a prime fusion target: the whole thing
  // should become a single Metal kernel under compile, vs ~8 kernel
  // launches uncompiled.
  auto t1 = mx::multiply(a, b);
  auto t2 = mx::add(t1, c);
  auto t3 = mx::multiply(t2, t2);
  auto t4 = mx::add(t3, a);
  auto t5 = mx::exp(t4);
  auto t6 = mx::multiply(t5, b);
  auto t7 = mx::add(t6, c);
  auto t8 = mx::multiply(t7, a);
  return {t8};
}

static void sanity_on_device(mx::Device::DeviceType dev_type, const char* label,
                             int warmup, int iters) {
  mx::set_default_device(mx::Device(dev_type));

  auto key = mx::random::key(7);
  auto a = mx::random::uniform(-1.0f, 1.0f, {kSanitySize}, mx::float32, key);
  auto b = mx::random::uniform(-1.0f, 1.0f, {kSanitySize}, mx::float32, key);
  auto c = mx::random::uniform(-1.0f, 1.0f, {kSanitySize}, mx::float32, key);
  mx::eval(a); mx::eval(b); mx::eval(c);
  mx::synchronize();

  std::vector<mx::array> inputs = {a, b, c};

  auto uncompiled_fn = [&inputs]() { return sanity_fn(inputs); };
  auto compiled_closure = mx::compile(sanity_fn, /*shapeless=*/false);
  auto compiled_fn = [&compiled_closure, &inputs]() {
    return compiled_closure(inputs);
  };

  std::printf("\n--- sanity: 8-op elementwise chain, %d elements, %s ---\n",
              kSanitySize, label);

  auto uncompiled = time_runs(uncompiled_fn, warmup, iters);
  auto compiled = time_runs(compiled_fn, warmup, iters);

  std::printf("  uncompiled  min=%.3f ms  median=%.3f ms  p95=%.3f ms\n",
              uncompiled.min_ms, uncompiled.median_ms, uncompiled.p95_ms);
  std::printf("  compiled    min=%.3f ms  median=%.3f ms  p95=%.3f ms\n",
              compiled.min_ms, compiled.median_ms, compiled.p95_ms);
  std::printf("  speedup (median) = %.3fx    (min) = %.3fx\n",
              uncompiled.median_ms / compiled.median_ms,
              uncompiled.min_ms / compiled.min_ms);
}

int main(int argc, char** argv) {
  int warmup = 50;
  int iters = 1000;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      warmup = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      iters = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--seq") == 0 && i + 1 < argc) {
      kSeq = std::atoi(argv[++i]);
    }
  }

  std::printf("Emily / M6 de-risk: mlx::core::compile microbenchmark\n");

  // Explicitly enable compile in case env disables it.
  mx::enable_compile();

  // Sanity first — if this doesn't win, harness is broken.
  sanity_on_device(mx::Device::gpu, "GPU (Metal)", warmup, iters);
  sanity_on_device(mx::Device::cpu, "CPU", warmup, iters);

  // Transformer block — the real question.
  run_on_device(mx::Device::gpu, "GPU (Metal)", warmup, iters);
  run_on_device(mx::Device::cpu, "CPU", warmup, iters);

  return 0;
}
