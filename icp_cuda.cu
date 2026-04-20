// icp_cuda_kernels.cu
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#define CUDA_CHECK(x) do {                              \
  cudaError_t err = (x);                                \
  if (err != cudaSuccess) {                             \
    printf("CUDA error %s:%d: %s\n",                    \
      __FILE__, __LINE__, cudaGetErrorString(err));     \
  }                                                     \
} while(0)

// NN: O(N*M) (일단 유지). 나중에 grid/kd-tree로 바꾸면 더 빨라짐.
__global__ void findNearestNeighbors2D(
    const float* __restrict__ src,   // [N*3] but z ignored
    const float* __restrict__ tgt,   // [M*3]
    int N, int M,
    int* __restrict__ indices)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  float sx = src[3*i + 0];
  float sy = src[3*i + 1];

  float best = 1e30f;
  int best_j = 0;

  for (int j = 0; j < M; ++j) {
    float tx = tgt[3*j + 0];
    float ty = tgt[3*j + 1];
    float dx = sx - tx;
    float dy = sy - ty;
    float d2 = dx*dx + dy*dy;
    if (d2 < best) {
      best = d2;
      best_j = j;
    }
  }

   float max_corr_dist_sq = 0.1;
   
   if(best <= max_corr_dist_sq){
       indices[i] = best_j;
   }else{
       indices[i] = -1;
   }
  
}

// Pass A: sum_src(x,y), sum_tgt(x,y)
__global__ void reduceSums2D(
    const float* __restrict__ src,
    const float* __restrict__ tgt,
    const int* __restrict__ idx,
    int N,
    float* __restrict__ out_sum) // out_sum[4] = {sum_sx,sum_sy,sum_tx,sum_ty}
{
  __shared__ float sh_sx[256];
  __shared__ float sh_sy[256];
  __shared__ float sh_tx[256];
  __shared__ float sh_ty[256];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;

  float sx=0, sy=0, tx=0, ty=0;
  if (i < N) {
    int j = idx[i];
    sx = src[3*i + 0];
    sy = src[3*i + 1];
    tx = tgt[3*j + 0];
    ty = tgt[3*j + 1];
  }

  sh_sx[tid] = sx;
  sh_sy[tid] = sy;
  sh_tx[tid] = tx;
  sh_ty[tid] = ty;
  __syncthreads();

  // block reduction
  for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sh_sx[tid] += sh_sx[tid + stride];
      sh_sy[tid] += sh_sy[tid + stride];
      sh_tx[tid] += sh_tx[tid + stride];
      sh_ty[tid] += sh_ty[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(&out_sum[0], sh_sx[0]);
    atomicAdd(&out_sum[1], sh_sy[0]);
    atomicAdd(&out_sum[2], sh_tx[0]);
    atomicAdd(&out_sum[3], sh_ty[0]);
  }
}

// Pass B: H = Σ (p-μp)(q-μq)^T  (2x2)
// out_H[4] = {H00,H01,H10,H11}
__global__ void reduceH2D(
    const float* __restrict__ src,
    const float* __restrict__ tgt,
    const int* __restrict__ idx,
    int N,
    float mu_sx, float mu_sy,
    float mu_tx, float mu_ty,
    float* __restrict__ out_H)
{
  __shared__ float sh_H00[256];
  __shared__ float sh_H01[256];
  __shared__ float sh_H10[256];
  __shared__ float sh_H11[256];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;

  float h00=0, h01=0, h10=0, h11=0;
  if (i < N) {
    int j = idx[i];
    float px = src[3*i + 0] - mu_sx;
    float py = src[3*i + 1] - mu_sy;
    float qx = tgt[3*j + 0] - mu_tx;
    float qy = tgt[3*j + 1] - mu_ty;

    h00 = px*qx;
    h01 = px*qy;
    h10 = py*qx;
    h11 = py*qy;
  }

  sh_H00[tid] = h00;
  sh_H01[tid] = h01;
  sh_H10[tid] = h10;
  sh_H11[tid] = h11;
  __syncthreads();

  for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sh_H00[tid] += sh_H00[tid + stride];
      sh_H01[tid] += sh_H01[tid + stride];
      sh_H10[tid] += sh_H10[tid + stride];
      sh_H11[tid] += sh_H11[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(&out_H[0], sh_H00[0]);
    atomicAdd(&out_H[1], sh_H01[0]);
    atomicAdd(&out_H[2], sh_H10[0]);
    atomicAdd(&out_H[3], sh_H11[0]);
  }
}

// d_src에 2D rigid transform 적용
__global__ void applyTransform2D(
    float* __restrict__ src,
    int N,
    float c, float s,
    float tx, float ty)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  float x = src[3*i + 0];
  float y = src[3*i + 1];

  float nx = c*x - s*y + tx;
  float ny = s*x + c*y + ty;

  src[3*i + 0] = nx;
  src[3*i + 1] = ny;
  // z는 그대로
}

// 외부에서 호출할 C API
extern "C" void launchNearestNeighborKernel(const float* d_src, const float* d_tgt, int N, int M, int* d_indices) {
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  findNearestNeighbors2D<<<blocks, threads>>>(d_src, d_tgt, N, M, d_indices);
}

extern "C" void launchReduceSums2D(const float* d_src, const float* d_tgt, const int* d_idx, int N, float* d_sum4) {
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  reduceSums2D<<<blocks, threads>>>(d_src, d_tgt, d_idx, N, d_sum4);
}

extern "C" void launchReduceH2D(const float* d_src, const float* d_tgt, const int* d_idx, int N,
                                float mu_sx, float mu_sy, float mu_tx, float mu_ty, float* d_H4) {
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  reduceH2D<<<blocks, threads>>>(d_src, d_tgt, d_idx, N, mu_sx, mu_sy, mu_tx, mu_ty, d_H4);
}

extern "C" void launchApplyTransform2D(float* d_src, int N, float c, float s, float tx, float ty) {
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  applyTransform2D<<<blocks, threads>>>(d_src, N, c, s, tx, ty);
}