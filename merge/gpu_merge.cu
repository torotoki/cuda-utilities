__global__ merge_kernel_v1(
    const int num_elements,
    const int* values1,
    const int* values2,
    int* sorted
) {
  int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
}
