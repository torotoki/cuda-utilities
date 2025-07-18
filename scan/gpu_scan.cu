#include <cassert>
#include <iostream>
#include <cooperative_groups.h>
#include "../common/cuda_utility.hpp"

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
// 処理順:
//   * block内だけで計算する-> Sという中間出力の配列に書く
//   * まとめ上げる処理を書く（もう一度Sにscanを通す）
//   * 定数を足すカーネルを作る
// 計算量:
//   * 入力長をnとする
//   * それぞれのthreadでO(log n)回の演算を行う
//   * n thread 必要なので、全体の計算量はO(n log n)になる
/////////////////////////////////////////
__global__ void scan_gpu_kernel_v2_first_phase(
    int num_elements,
    const uint* values,
    uint* prefix_sum,
    uint* sections
) {
  cg::thread_block cta = cg::this_thread_block();
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint local_prefix_sum[SECTION_LIMITATION];

  if (threadIdx.x < num_elements)
    local_prefix_sum[threadIdx.x] = values[threadIdx.x];
  else
    local_prefix_sum[threadIdx.x] = 0;

  for (uint stride = 1; stride < blockDim.x; stride *= 2) {
    cta.sync();
    uint temp;
    if (threadIdx.x >= stride)
      temp = local_prefix_sum[threadIdx.x] + local_prefix_sum[threadIdx.x - stride];
    cta.sync();
    if (threadIdx.x >= stride)
      local_prefix_sum[threadIdx.x] = temp;
  }
  if (global_idx < num_elements)
    prefix_sum[global_idx] = local_prefix_sum[threadIdx.x];

  cg::sync(cta);
  if (threadIdx.x == blockDim.x - 1) {
    sections[blockIdx.x] = local_prefix_sum[threadIdx.x];
  } 
}

__global__ void scan_gpu_kernel_v2_second_phase(
    const int num_sections,
    uint* sections
) {
  cg::thread_block cta = cg::this_thread_block();

  if (threadIdx.x >= num_sections)
    sections[threadIdx.x] = 0;
  
  for (uint stride = 1; stride <= blockDim.x; stride *= 2) {
    cta.sync();
    uint temp;
    if (threadIdx.x >= stride)
      temp = sections[threadIdx.x] + sections[threadIdx.x - stride];
    cta.sync();
    if (threadIdx.x >= stride)
      sections[threadIdx.x] = temp;
  }
}

__global__ void scan_gpu_kernel_v2_third_phase(
  const int num_elements,
  const uint* sections,
  uint* prefix_sum
) {
  int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (0 < blockIdx.x && global_idx < num_elements)
    prefix_sum[global_idx] += sections[blockIdx.x - 1];
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
  // uint* d_sections2;
  // checkCudaErrors(cudaMalloc(&d_sections2, sections_size));
  // uint* d_sections_compressed;
  // uint num_sections_compressed = 
  //   (num_blocks + SECTION_LIMITATION - 1) / SECTION_LIMITATION;
  // size_t sections_compressed_size = 
  //   num_sections_compressed * sizeof(uint);
  // checkCudaErrors(cudaMalloc(&d_sections_compressed, sections_compressed_size));

  // First CUDA kernel: scan on each block
  scan_gpu_kernel_v2_first_phase
    <<<num_blocks, num_threads, SECTION_LIMITATION * sizeof(uint)>>>(
      num_elements,
      d_values,
      d_prefix_sum,
      d_sections
  );
  
  // Second CUDA kernel: scan on block-wise sections
  assert(num_blocks <= SECTION_LIMITATION);
  // scan_gpu_kernel_v2_first_phase
  //   <<<num_sections_compressed, num_blocks, sections_compressed_size>>>
  // (
  //     num_blocks,
  //     d_sections,
  //     d_sections_compressed,
  //     d_sections2
  // );

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


/////////////////////////////////////////
// Inclusive Scan with Brent-Kung algorithm with shared memory
// block内で計算して、後で結果をまとめ上げる
// 処理順:
//   * block内だけで計算する-> Sという中間出力の配列に書く
//   * 残りの処理はBrent-Kung algorithmと同じ
// メモ:
//   * 本だとbranch divergenceを避けるために複雑になっている
//   * SECTION_LIMITAION (= 1024)だが、1024threadsで2048要素計算することもできる
//     （ここではしていない）
/////////////////////////////////////////
__global__ void scan_gpu_kernel_v3_first_phase(
    const int num_elements,
    const uint* values,
    uint* prefix_sum,
    uint* sections
) {
  __shared__ int local_prefix_sum[SECTION_LIMITATION];
  cg::thread_block cta = cg::this_thread_block();
  uint global_idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (global_idx < num_elements)
    local_prefix_sum[threadIdx.x] = values[global_idx];
  
  for (uint stride = 1; stride <= blockDim.x; stride *= 2) {
    cta.sync();
    if ((threadIdx.x + 1) % (2 * stride) == 0)
      local_prefix_sum[threadIdx.x] += local_prefix_sum[threadIdx.x - stride];
  }

  for (uint stride = SECTION_LIMITATION / 4; stride > 0; stride /= 2) {
    cta.sync();
    if ((threadIdx.x + 1) % (2 * stride) == 0)
      local_prefix_sum[threadIdx.x + stride] += local_prefix_sum[threadIdx.x];
  }
  cta.sync();

  if (global_idx < num_elements)
    prefix_sum[global_idx] = local_prefix_sum[threadIdx.x];

  if (threadIdx.x == blockDim.x - 1)
    sections[blockIdx.x] = local_prefix_sum[threadIdx.x];
}

void launch_kernel_scan_v3(
	const int num_elements,
  uint* d_values,
  uint* d_prefix_sum,
  int num_blocks,
  int num_threads
) {
  uint* d_sections;
  size_t sections_size = num_blocks * sizeof(uint);
  checkCudaErrors(cudaMalloc(&d_sections, sections_size));

  // First CUDA kernel: scan on each block
  scan_gpu_kernel_v3_first_phase
    <<<num_blocks, num_threads, SECTION_LIMITATION * sizeof(uint)>>>(
      num_elements,
      d_values,
      d_prefix_sum,
      d_sections
  );
  
  // For the rest of the phases, we reuse v2 kernels

  // Second CUDA kernel: scan on block-wise sections
  assert(num_blocks <= SECTION_LIMITATION);
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


/////////////////////////////////////////
// Inclusive Scan with Brent-Kung algorithm with shared memory
// block内で計算して、後で結果をまとめ上げる
// 処理順:
//   * block内だけで計算する-> Sという中間出力の配列に書く
//   * 残りの処理はKogge-Stone algorithmと同じ
//   * branch divergence を減らしたバージョン
// メモ:
//   * SECTION_LIMITAION (= 1024)だが、1024threadsで2048要素計算することもできる
//     （ここではしていない）
/////////////////////////////////////////
__global__ void scan_gpu_kernel_v4_first_phase(
    const int num_elements,
    const uint* values,
    uint* prefix_sum,
    uint* sections
) {
  __shared__ int local_prefix_sum[SECTION_LIMITATION];
  cg::thread_block cta = cg::this_thread_block();
  uint global_idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (global_idx < num_elements)
    local_prefix_sum[threadIdx.x] = values[global_idx];
  
  for (uint stride = 1; stride <= blockDim.x; stride *= 2) {
    cta.sync();
    uint target_idx = (threadIdx.x + 1) * 2 * stride - 1;
    if (target_idx < SECTION_LIMITATION)
      local_prefix_sum[target_idx] += local_prefix_sum[target_idx - stride];
  }

  for (uint stride = SECTION_LIMITATION / 4; stride > 0; stride /= 2) {
    cta.sync();
    uint target_idx = (threadIdx.x + 1) * 2 * stride - 1;
    if (target_idx < SECTION_LIMITATION)
      local_prefix_sum[target_idx + stride] += local_prefix_sum[target_idx];
  }
  cta.sync();

  if (global_idx < num_elements)
    prefix_sum[global_idx] = local_prefix_sum[threadIdx.x];

  if (threadIdx.x == blockDim.x - 1)
    sections[blockIdx.x] = local_prefix_sum[threadIdx.x];
}

void launch_kernel_scan_v4(
	const int num_elements,
  uint* d_values,
  uint* d_prefix_sum,
  int num_blocks,
  int num_threads
) {
  uint* d_sections;
  size_t sections_size = num_blocks * sizeof(uint);
  checkCudaErrors(cudaMalloc(&d_sections, sections_size));

  // First CUDA kernel: scan on each block
  scan_gpu_kernel_v4_first_phase
    <<<num_blocks, num_threads, SECTION_LIMITATION * sizeof(uint)>>>(
      num_elements,
      d_values,
      d_prefix_sum,
      d_sections
  );
  
  // For the rest of the phases, we reuse v2 kernels

  // Second CUDA kernel: scan on block-wise sections
  assert(num_blocks <= SECTION_LIMITATION);
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

