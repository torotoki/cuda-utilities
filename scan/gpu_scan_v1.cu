#include <cassert>
#include <iostream>
#include <cooperative_groups.h>
#include "../cuda_utility.hpp"

namespace cg = cooperative_groups;
using uint = unsigned int;
const int SECTION_LIMITATION = 1024;

/////////////////////////////////////////
// Inclusive Scan with Kogge-Stone algorithm
// Shared memory を使わずにglobal memoryですべて完結させる
/////////////////////////////////////////
__global__ void scan_gpu_kernel_v1(
  int num_elements,
  const uint* values,
  uint* prefix_sum
) {
  cg::grid_group cga = cg::this_grid();
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < num_elements)
    prefix_sum[global_idx] = values[global_idx];
  else
    prefix_sum[global_idx] = 0;

  for (int stride = 1; stride <= global_idx; stride *= 2) {
    cg::sync(cga);
    uint temp;
    temp = prefix_sum[global_idx] + prefix_sum[global_idx - stride];
    cg::sync(cga);
    prefix_sum[global_idx] = temp;
  }
}

void launch_kernel_scan_v1(
	const int num_elements,
  uint* d_values,
  uint* d_prefix_sum,
  int num_blocks,
  int num_threads
) {
  void* args[] = {(void*)&num_elements, (void*)&d_values, (void*)&d_prefix_sum};
  checkCudaErrors(cudaLaunchCooperativeKernel(
      (void*)scan_gpu_kernel_v1,
      num_blocks,
      num_threads,
      args
  ));
  cudaDeviceSynchronize();
}


/////////////////////////////////////////
// Inclusive Scan with Kogge-Stone algorithm with shared memory
// block内で計算して、後で結果をまとめ上げる
// TODO:
//   * block内だけで計算する-> Sという中間出力の配列に書く
//   * まとめ上げる処理を書く（もう一度Sにscanを通す）
//   * 定数を足すカーネルを作る
/////////////////////////////////////////
__global__ void scan_gpu_kernel_v2_first_phase(
  int num_elements,
  const uint* values,
  uint* prefix_sum,
  uint* sections
) {
  cg::thread_block cta = cg::this_thread_block();
  int thread_idx = threadIdx.x;
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ int local_prefix_sum[SECTION_LIMITATION];

  if (thread_idx < num_elements)
    local_prefix_sum[thread_idx] = values[thread_idx];
  else
    local_prefix_sum[thread_idx] = 0;

  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    cg::sync(cta);
    float temp;
    if (thread_idx >= stride)
      temp = local_prefix_sum[thread_idx] + local_prefix_sum[thread_idx - stride];
    cg::sync(cta);
    if (thread_idx >= stride)
      local_prefix_sum[thread_idx] = temp;
  }
  if (thread_idx < num_elements)
    prefix_sum[global_idx] = local_prefix_sum[thread_idx];

  cg::sync(cta);
  if (thread_idx == blockDim.x - 1) {
    sections[blockIdx.x] = prefix_sum[thread_idx];
  } 
}

__global__ void scan_gpu_kernel_v2_second_phase(
  const int num_sections,
  uint* sections
) {
  cg::thread_block cta = cg::this_thread_block();
  
  for (uint stride = 1; stride <= num_sections; stride *= 2) {
    cta.sync();
    uint temp = sections[threadIdx.x] + sections[threadIdx.x - stride];
    cta.sync();
    sections[threadIdx.x] = temp;
  }
}

__global__ void scan_gpu_kernel_v2_third_phase(
  const int num_elements,
  const uint* sections,
  uint* prefix_sum
) {
  int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  prefix_sum[global_idx] += sections[blockIdx.x];
}


void launch_kernel_scan_v2(
	const int num_elements,
  uint* d_values,
  uint* d_prefix_sum,
  int num_blocks,
  int num_threads
) {
  uint* d_sections;
  size_t sections_size = num_blocks * sizeof(uint);
  checkCudaErrors(cudaMalloc(&d_sections, sections_size));
  uint* d_sections_compressed;
  uint num_sections_compressed = 
    (num_blocks + SECTION_LIMITATION - 1) / SECTION_LIMITATION;
  size_t sections_compressed_size = 
    num_sections_compressed * sizeof(uint);
  checkCudaErrors(cudaMalloc(&d_sections_compressed, sections_compressed_size));

  // First CUDA kernel: scan on each block
  scan_gpu_kernel_v2_first_phase<<<num_blocks, num_threads, sections_size>>>(
      num_elements,
      d_values,
      d_sections,
      d_prefix_sum
  );
  
  // Second CUDA kernel: scan on block-wise sections
  std::cout << num_blocks << std::endl;
  assert(SECTION_LIMITATION <= num_blocks);
  assert(num_blocks <= SECTION_LIMITATION * SECTION_LIMITATION);
  scan_gpu_kernel_v2_first_second_phase
    <<<num_sections_compressed, num_blocks, sections_compressed_size>>>
  (
      num_blocks,
      d_sections,
      d_sections_compressed
  );
  scan_gpu_kernel_v2_second_phase<<<1, num_blocks>>>(
      num_blocks,
      d_sections
  );

  // Third CUDA kernel: adding a section element as a base
  scan_gpu_kernel_v2_third_phase<<<num_blocks, num_threads>>>(
      num_elements,
      d_sections,
      d_prefix_sum
  );
}
