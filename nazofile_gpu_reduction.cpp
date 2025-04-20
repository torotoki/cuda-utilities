#include <stdexcept>

__global__ void reduce_gpu_kernel_v1(
    const int* d_input, size_t size, int* d_out
) {
  cg::thread_block cta = cg::this_thread_block();
  __shared__ int shared[];
  int tid = threadIdx.x;
  int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_id < n)
    shared[tid] = d_input[i];
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

int call_reduction(
    const int* d_input,
    size_t size,
    int* d_out,
    int* h_tmp_out,
    int num_blocks,
    int num_threads,
    int kernel_version = 1
) {
  switch (kernel_version) {
    case 1:
      reduce_gpu_kernel_v1<<<num_blocks, num_threads,
	      sizeof(int) * num_threads>>>(d_input, size, d_out);
      break;
    case 2:
      reduce_gpu_kernel_v2<<<num_blocks, num_threads,
	      sizeof(int) * num_threads>>>(d_input, size, d_out);
      break;
    default:
      throw std::invalid_argument("Invalid kernel version.");
  }
 
  checkCudaErrors(cudaMemcpy(h_tmp_out, d_tmp_out, sizeof(int) * n_blocks,
                             cudaMemcpyDefault));

  // 最後のブロック数分はCPUでまとめる
  int result = reduce(h_tmp_out, n_blocks);
  return result;
}
