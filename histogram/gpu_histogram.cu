__global__ histogram_kernel_v1(
    const int* h_assignments,
    const uint num_elements,
    uint* h_histogram,
    const uint num_bins
) {
  int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_id = threadIdx.x;

  if (global_id > num_elements)
    return;

  int bin_idx = h_assignments[global_id];
  h_histogram[bin_idx]++;
}
