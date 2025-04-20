#include <cooperative_groups.h>

#include "reduction.hpp"
#include "cuda_utility.hpp"

namespace cg = cooperative_groups;


__global__ void reduce_gpu_kernel_v1(
    const int* d_values, size_t num_elements, int* d_out
) {
  cg::thread_block cta = cg::this_thread_block();
  // 動的共有メモリ
  extern __shared__ int shared[];
  int tid = threadIdx.x;
  int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_id < num_elements)
    shared[tid] = d_values[global_id];
  else
    shared[tid] = 0;

  cg::sync(cta);

  for (int neighbor = 1; neighbor < blockDim.x; neighbor *= 2)
  {
    // 最初はtidが2の倍数のとき、tid番の隣のブロックをtid番に足す
    // 次はtidが4の倍数のとき、tid番の2つとなりのブロックをtid番に足す
    if (tid % (2 * neighbor) == 0)
      shared[tid] += shared[tid + neighbor];
    cg::sync(cta);

  if (tid == 0)
    d_out[blockIdx.x] = shared[0];
  }
}

__global__ void reduce_gpu_kernel_v2(
    const int* d_values, size_t num_elements, int* d_out
) {
  cg::thread_block cta = cg::this_thread_block();
  int tid = threadIdx.x;
  int global_id = blockDim.x * blockIdx.x + threadIdx.x;

  extern __shared__ int shared[];
  
  if (global_id < num_elements)
    shared[tid] = d_values[global_id];

  cg::sync(cta);
  
  int next_id = 1;
  while (tid + next_id < blockDim.x) {
    // next_id * 2 represents next_next_id
    if (tid % (next_id * 2) == 0)
      shared[tid] += shared[tid + next_id];
    next_id *= 2;

    cg::sync(cta);
  }

  // 結果を書き込む
  if (tid == 0)
    d_out[blockIdx.x] = shared[0];
}

__global__ void reduce_gpu_kernel_v3(
    const int* d_values, size_t num_elements, int* d_out
) {
  cg::thread_block cta = cg::this_thread_block();
  int tid = threadIdx.x;
  int global_id = blockDim.x * blockIdx.x + threadIdx.x;

  extern __shared__ int shared[];
  
  if (global_id < num_elements)
    shared[tid] = d_values[global_id];

  cg::sync(cta);
  
  int limit = blockDim.x;
  int nid = limit / 2;
  while (0 < nid && tid + nid < limit) {
    shared[tid] += shared[tid + nid];
    
    limit /= 2;
    nid = limit / 2;
    cg::sync(cta);
  }

  // 結果を書き込む
  if (tid == 0)
    d_out[blockIdx.x] = shared[0];
}

int call_reduction(
    const int* d_input,
    size_t num_elements,
    int* d_out,
    int* h_out,
    int num_blocks,
    int num_threads,
    int kernel_version = 1
) {
  switch (kernel_version) {
    case 1:
      reduce_gpu_kernel_v1<<<num_blocks, num_threads,
	      sizeof(int) * num_threads>>>(d_input, num_elements, d_out);
      break;
    case 2:
      reduce_gpu_kernel_v2<<<num_blocks, num_threads,
	      sizeof(int) * num_threads>>>(d_input, num_elements, d_out);
      break;
    case 3:
      reduce_gpu_kernel_v3<<<num_blocks, num_threads,
          sizeof(int) * num_threads>>>(d_input, num_elements, d_out);
      break;
    default:
      throw std::invalid_argument("Invalid kernel version.");
  }
  checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(int) * num_blocks,
                             cudaMemcpyDefault));

  // 最後のブロック数分はCPUでまとめる
  int result = reduce(h_out, num_blocks);
  return result;
}
