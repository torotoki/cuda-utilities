#include <math.h>

__device__ int find_input_segments(
    const int search,
    const int m, const int* values1,
    const int n, const int* values2
) {
  int i_low = max(0, search - n);
  int i = min(search, m);
  int j_low = max(0, search - m);
  int j = search - i;
  int delta;
  bool active = true;
  while (active) {
    if (0 < i && j < n && values1[i - 1] > values2[j]) {
      // guess is too high
      delta = ceilf((i - i_low) / 2);
      j_low = j;
      j = j + delta;
      i = i - delta;
    } else if (0 < j && i < m && values2[j - 1] >= values1[i]) {
      // guess is too low
      delta = ceilf((j - j_low) / 2);
      i_low = i;
      i = i + delta;
      j = j - delta;
    } else {
      // guess is correct
      active = false;
    }
  }
  //printf("i: %d, j: %d\n", i, j);
  return i;
}

__device__ void merge_sequential(
    const int m,
    const int* values1,
    const int n,
    const int* values2,
    int* sorted
) {
  int i = 0;  // Index into values1
  int j = 0;  // Index into values2
  int k = 0;  // Index into sorted
  while (i < m && j < n) {
    if (values1[i] <= values2[j]) {
      sorted[k++] = values1[i++];
    } else {
      sorted[k++] = values2[j++];
    }
  }

  while (i < m) {
    sorted[k++] = values1[i++];
  }
  while (j < n) {
    sorted[k++] = values2[j++];
  }
}

__global__ void merge_kernel_v1(
    const int num_elements,
    const int m,
    const int* values1,
    const int n,
    const int* values2,
    int* sorted
) {
  int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  int elements_per_thread = ceilf(num_elements / (blockDim.x * gridDim.x));

  // start output index
  int k_curr = global_idx * elements_per_thread;
  int k_next = min((threadIdx.x + 1) * elements_per_thread, m + n);
  int i_curr = find_input_segments(k_curr, m, values1, n, values2);
  int i_next = find_input_segments(k_next, m, values1, n, values2);
  int j_curr = k_curr - i_curr;
  int j_next = k_next - i_next;
  merge_sequential(
      i_next - i_curr,
      &values1[i_curr],
      j_next - j_curr,
      &values2[j_curr],
      &sorted[k_curr]
  );
  
  /**
   * 
   */

}
