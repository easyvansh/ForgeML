#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_OK(x) do { \
  cudaError_t e = (x); \
  if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s:%d code=%d name=%s description=%s\n", \
      __FILE__, __LINE__, (int)e, cudaGetErrorName(e), cudaGetErrorString(e)); \
    return 2; \
  } \
} while (0)

template <int TILE>
__global__ void tiled_attention(const float* q, const float* k, const float* v,
                                float* out, int B, int H, int T, int D) {
  extern __shared__ float smem[];
  float* sk = smem;
  float* sv = smem + TILE * D;
  int row = blockIdx.x;
  int total = B * H * T;
  if (row >= total) return;
  int d = threadIdx.x;
  bool active = d < D;
  int t = row % T;
  int bh = row / T;
  int qbase = (bh * T + t) * D;
  float m = -INFINITY;
  float l = 0.0f;
  float acc = 0.0f;
  float scale = rsqrtf((float)D);

  for (int start = 0; start <= t; start += TILE) {
    int width = min(TILE, t - start + 1);
    for (int x = d; x < width * D; x += blockDim.x) {
      int j = x / D;
      int c = x % D;
      int src = (bh * T + start + j) * D + c;
      sk[j * D + c] = k[src];
      sv[j * D + c] = v[src];
    }
    __syncthreads();
    for (int j = 0; j < width; ++j) {
      if (active) {
        float score = 0.0f;
        for (int c = 0; c < D; ++c) score += q[qbase + c] * sk[j * D + c];
        score *= scale;
        float next = fmaxf(m, score);
        float old = isinf(m) ? 0.0f : expf(m - next);
        float p = expf(score - next);
        acc = old * acc + p * sv[j * D + d];
        l = old * l + p;
        m = next;
      }
    }
    __syncthreads();
  }
  if (active) out[qbase + d] = acc / l;
}

static void reference(const std::vector<float>& q, const std::vector<float>& k,
                      const std::vector<float>& v, std::vector<float>& out,
                      int B, int H, int T, int D) {
  const float scale = 1.0f / std::sqrt((float)D);
  for (int bh = 0; bh < B * H; ++bh) {
    for (int t = 0; t < T; ++t) {
      std::vector<float> scores(t + 1);
      float max_score = -INFINITY;
      for (int j = 0; j <= t; ++j) {
        float score = 0.0f;
        for (int c = 0; c < D; ++c)
          score += q[(bh * T + t) * D + c] * k[(bh * T + j) * D + c];
        scores[j] = score * scale;
        max_score = std::max(max_score, scores[j]);
      }
      float denom = 0.0f;
      for (float score : scores) denom += std::exp(score - max_score);
      for (int d = 0; d < D; ++d) {
        float value = 0.0f;
        for (int j = 0; j <= t; ++j)
          value += std::exp(scores[j] - max_score) * v[(bh * T + j) * D + d];
        out[(bh * T + t) * D + d] = value / denom;
      }
    }
  }
}

int main(int argc, char** argv) {
  int T = argc > 1 ? std::atoi(argv[1]) : 128;
  int B = 1, H = 2, D = 32;
  size_t n = (size_t)B * H * T * D;
  std::vector<float> q(n), k(n), v(n), cpu(n), gpu(n);
  for (size_t i = 0; i < n; ++i) {
    q[i] = std::sin(i * .017f);
    k[i] = std::cos(i * .013f);
    v[i] = std::sin(i * .009f);
  }
  reference(q, k, v, cpu, B, H, T, D);

  float *dq, *dk, *dv, *dout;
  CUDA_OK(cudaMalloc(&dq, n * sizeof(float)));
  CUDA_OK(cudaMalloc(&dk, n * sizeof(float)));
  CUDA_OK(cudaMalloc(&dv, n * sizeof(float)));
  CUDA_OK(cudaMalloc(&dout, n * sizeof(float)));
  CUDA_OK(cudaMemcpy(dq, q.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dk, k.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dv, v.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  int blocks = B * H * T;
  int threads = std::max(D, 128);
  size_t shared = 2 * (size_t)32 * D * sizeof(float);
  tiled_attention<32><<<blocks, threads, shared>>>(dq, dk, dv, dout, B, H, T, D);
  CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaMemcpy(gpu.data(), dout, n * sizeof(float), cudaMemcpyDeviceToHost));
  float max_error = 0.0f;
  for (size_t i = 0; i < n; ++i) max_error = std::max(max_error, std::fabs(cpu[i] - gpu[i]));

  cudaEvent_t begin, end;
  CUDA_OK(cudaEventCreate(&begin));
  CUDA_OK(cudaEventCreate(&end));
  for (int i = 0; i < 10; ++i)
    tiled_attention<32><<<blocks, threads, shared>>>(dq, dk, dv, dout, B, H, T, D);
  CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaEventRecord(begin));
  for (int i = 0; i < 50; ++i)
    tiled_attention<32><<<blocks, threads, shared>>>(dq, dk, dv, dout, B, H, T, D);
  CUDA_OK(cudaEventRecord(end));
  CUDA_OK(cudaEventSynchronize(end));
  float ms;
  CUDA_OK(cudaEventElapsedTime(&ms, begin, end));
  std::printf("{\"context\":%d,\"max_abs_error\":%.8g,\"median_like_ms\":%.6f,\"launches\":50,\"kernel\":\"shared_memory_tiled_reference\",\"shared_bytes\":%zu}\n",
              T, max_error, ms / 50.0f, shared);
  return max_error < 2e-5f ? 0 : 1;
}
