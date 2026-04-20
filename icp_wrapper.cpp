#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

extern "C" void launchNearestNeighborKernel(const float* d_src, const float* d_tgt, int N, int M, int* d_indices);
extern "C" void launchReduceSums2D(const float* d_src, const float* d_tgt, const int* d_idx, int N, float* d_sum4);
extern "C" void launchReduceH2D(const float* d_src, const float* d_tgt, const int* d_idx, int N,
                                float mu_sx, float mu_sy, float mu_tx, float mu_ty, float* d_H4);
extern "C" void launchApplyTransform2D(float* d_src, int N, float c, float s, float tx, float ty);

#define CUDA_CHECK(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " " \
              << cudaGetErrorString(err) << std::endl; \
  } \
} while(0)

struct ICPWorkspace {
  int capN = 0;
  int capM = 0;
  float* d_src = nullptr;
  float* d_tgt = nullptr;
  int*   d_idx = nullptr;
  float* d_sum4 = nullptr; // {sum_sx,sum_sy,sum_tx,sum_ty}
  float* d_H4   = nullptr; // {H00,H01,H10,H11}

  void ensure(int N, int M) {
    if (N <= capN && M <= capM) return;

    release();
    capN = N;
    capM = M;

    CUDA_CHECK(cudaMalloc(&d_src,  sizeof(float) * capN * 3));
    CUDA_CHECK(cudaMalloc(&d_tgt,  sizeof(float) * capM * 3));
    CUDA_CHECK(cudaMalloc(&d_idx,  sizeof(int)   * capN));
    CUDA_CHECK(cudaMalloc(&d_sum4, sizeof(float) * 4));
    CUDA_CHECK(cudaMalloc(&d_H4,   sizeof(float) * 4));
  }

  void release() {
    if (d_src)  cudaFree(d_src);
    if (d_tgt)  cudaFree(d_tgt);
    if (d_idx)  cudaFree(d_idx);
    if (d_sum4) cudaFree(d_sum4);
    if (d_H4)   cudaFree(d_H4);
    d_src = d_tgt = d_sum4 = d_H4 = nullptr;
    d_idx = nullptr;
    capN = capM = 0;
  }

  ~ICPWorkspace(){ release(); }
};

// 전역(또는 static)으로 유지해서 프레임 간 재사용
static ICPWorkspace g_ws;

// 2D Kabsch closed-form: theta = atan2(H01 - H10, H00 + H11)
static inline void computeRt2D_from_H_mu(
    float H00, float H01, float H10, float H11,
    float mu_sx, float mu_sy,
    float mu_tx, float mu_ty,
    float& c, float& s, float& tx, float& ty)
{
  float theta = std::atan2(H01 - H10, H00 + H11);
  c = std::cos(theta);
  s = std::sin(theta);

  // t = mu_tgt - R * mu_src
  float rxs = c*mu_sx - s*mu_sy;
  float rys = s*mu_sx + c*mu_sy;

  tx = mu_tx - rxs;
  ty = mu_ty - rys;
}

Eigen::Matrix4f runICPCUDA(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& src_cloud,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& tgt_cloud,
    int max_iterations,
    Eigen::Matrix4f init_guess)
{
  const int N = (int)src_cloud->size();
  const int M = (int)tgt_cloud->size();

  if (N <= 0 || M <= 0) {
    return Eigen::Matrix4f::Identity();
  }

  // Host -> float array
  std::vector<float> h_src(N * 3);
  std::vector<float> h_tgt(M * 3);
  for (int i = 0; i < N; ++i) {
    h_src[3*i+0] = src_cloud->points[i].x;
    h_src[3*i+1] = src_cloud->points[i].y;
    h_src[3*i+2] = src_cloud->points[i].z;
  }
  for (int i = 0; i < M; ++i) {
    h_tgt[3*i+0] = tgt_cloud->points[i].x;
    h_tgt[3*i+1] = tgt_cloud->points[i].y;
    h_tgt[3*i+2] = tgt_cloud->points[i].z;
  }

  // Workspace 확보 + 초기 업로드(프레임당 1회)
  g_ws.ensure(N, M);
  CUDA_CHECK(cudaMemcpy(g_ws.d_src, h_src.data(), sizeof(float)*N*3, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(g_ws.d_tgt, h_tgt.data(), sizeof(float)*M*3, cudaMemcpyHostToDevice));

  Eigen::Matrix4f total = init_guess;

  float c = init_guess(0,0);
  float s = init_guess(1,0);
  float tx = init_guess(0,3);
  float ty = init_guess(1,3);

  // 작은 결과만 받아올 host 버퍼
  float h_sum4[4];
  float h_H4[4];

  for (int iter = 0; iter < max_iterations; ++iter) {
    // 1) NN
    launchNearestNeighborKernel(g_ws.d_src, g_ws.d_tgt, N, M, g_ws.d_idx);
    CUDA_CHECK(cudaGetLastError());

    // 2) sums
    CUDA_CHECK(cudaMemset(g_ws.d_sum4, 0, sizeof(float)*4));
    launchReduceSums2D(g_ws.d_src, g_ws.d_tgt, g_ws.d_idx, N, g_ws.d_sum4);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_sum4, g_ws.d_sum4, sizeof(float)*4, cudaMemcpyDeviceToHost));
    float mu_sx = h_sum4[0] / N;
    float mu_sy = h_sum4[1] / N;
    float mu_tx = h_sum4[2] / N;
    float mu_ty = h_sum4[3] / N;

    // 3) H
    CUDA_CHECK(cudaMemset(g_ws.d_H4, 0, sizeof(float)*4));
    launchReduceH2D(g_ws.d_src, g_ws.d_tgt, g_ws.d_idx, N, mu_sx, mu_sy, mu_tx, mu_ty, g_ws.d_H4);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_H4, g_ws.d_H4, sizeof(float)*4, cudaMemcpyDeviceToHost));

    // 4) R,t (CPU에서 매우 가벼움)
    //float c,s,tx,ty;
    computeRt2D_from_H_mu(h_H4[0], h_H4[1], h_H4[2], h_H4[3], mu_sx, mu_sy, mu_tx, mu_ty, c, s, tx, ty);

    // 5) d_src에 적용(GPU)
    launchApplyTransform2D(g_ws.d_src, N, c, s, tx, ty);
    CUDA_CHECK(cudaGetLastError());

    // 6) 누적 transform 업데이트
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T(0,0)=c; T(0,1)=-s;
    T(1,0)=s; T(1,1)= c;
    T(0,3)=tx;
    T(1,3)=ty;

    total = T * total;

    float dtrans = std::sqrt(tx*tx + ty*ty);
    float drot = std::atan2(s, c);

    float trans_tol = 0.002f;
    float rot_tol = 0.001f;
    if (dtrans < trans_tol && std::fabs(drot) < rot_tol) break;
  }

  return total;
}