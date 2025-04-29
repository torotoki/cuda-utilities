
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void histogram_kernel_v2(
    const uint* h_assignments,
    const uint num_elements,
    uint* h_histogram,
    const uint num_bins
) {
  cg::thread_block cta = cg::this_thread_block();
  extern __shared__ uint shared_histogram[];
  int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_id = threadIdx.x;

  if (thread_id < num_bins)
    shared_histogram[thread_id] = 0;

  cg::sync(cta);

  if (thread_id < num_elements) {
    int bin_idx = h_assignments[global_id];
    // 以下は `h_histogram[bin_idx]++` の並列版
    atomicAdd(shared_histogram + bin_idx, 1);
  }

  cg::sync(cta);

  if (thread_id < num_bins) {
    uint sub_sum = shared_histogram[thread_id];
    atomicAdd(h_histogram + thread_id, sub_sum);
  }
}
