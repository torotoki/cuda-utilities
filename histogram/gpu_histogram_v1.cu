__global__ void histogram_kernel_v1(
    const uint* h_assignments,
    const uint num_elements,
    uint* h_histogram,
    const uint num_bins
) {
  int global_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_id > num_elements)
    return;

  int bin_idx = h_assignments[global_id];
  // 以下は `h_histogram[bin_idx]++` の並列版
  atomicAdd(h_histogram + bin_idx, 1);
}
