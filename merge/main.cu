#include <iostream>
#include <vector>
#include "cuda_utility.hpp"
#include "timer.cpp"
#include "host_utils.cpp"

#include "gpu_merge.cuh"

using uint = unsigned int;

double benchmarkOnCPU(
  const uint num_elements,
  const int* values1,
  const int* values2,
  int* sorted,
  const bool verbose = true
) {
  Stopwatch stopwatch = Stopwatch();
  stopwatch.start();

  uint i = 0, j = 0;
  uint cur = 0;
  while (i < num_elements && j < num_elements) {
    if (values1[i] < values2[j])
      sorted[cur++] = values1[i++];
    else
      sorted[cur++] = values2[j++];
  }

  while (i < num_elements)
    sorted[cur++] = values1[i++];

  while (j < num_elements)
    sorted[cur++] = values2[j++];

  stopwatch.stop();
  
  assert(i == num_elements);
  assert(j == num_elements);
  assert(cur == num_elements * 2);

  if (verbose)
    stopwatch.pprint();
  return stopwatch.get_elapsed_time_msec();
}

double benchmarkOnGPU(
  const uint num_elements,
  const int* h_values1,
  const int* h_values2,
  int* h_sorted,
  const int num_trials,
  const bool verbose = true
) {
  const int num_total_elements = 2 * num_elements;
  Stopwatch stopwatch_total = Stopwatch();
  Stopwatch stopwatch_compute = Stopwatch();
  stopwatch_total.start();

  int* d_values1;
  int* d_values2;
  int* d_sorted;

  const size_t values_size = num_elements * sizeof(int);
  checkCudaErrors(cudaMalloc(&d_values1, values_size));
  checkCudaErrors(cudaMalloc(&d_values2, values_size));
  checkCudaErrors(cudaMalloc(&d_sorted, 2 * values_size));

  checkCudaErrors(cudaMemcpy(d_values1, h_values1, values_size, cudaMemcpyDefault));
  checkCudaErrors(cudaMemcpy(d_values2, h_values2, values_size, cudaMemcpyDefault));

  // Computation
  stopwatch_compute.start();
  const int elements_per_block = 1024;
  assert(num_total_elements % elements_per_block == 0);
  const int num_blocks = num_total_elements / elements_per_block;
  const int threads_per_block = 512;
  for (uint i = 0; i < num_trials; ++i) {
    merge_kernel_v1<<<num_blocks, threads_per_block>>>(
        2 * num_elements,
        num_elements,
        d_values1,
        num_elements,
        d_values2,
        d_sorted
    );
    cudaDeviceSynchronize();
    cout << "GPU computed" << endl;
  }
  stopwatch_compute.stop();

  checkCudaErrors(cudaMemcpy(h_sorted, d_sorted, 2 * values_size, cudaMemcpyDefault));
  stopwatch_total.stop();

  stopwatch_compute.pprint();
  stopwatch_total.pprint();

  checkCudaErrors(cudaFree(d_values1));
  checkCudaErrors(cudaFree(d_values2));
  checkCudaErrors(cudaFree(d_sorted));

  return stopwatch_compute.get_elapsed_time_msec();
}

int main() {
  // 1 << 26
  const uint num_total_elements = 2 * (1 << 15);
  const uint num_elements = num_total_elements / 2;
  const int num_trials = 10;
  InputGenerator input_generator = InputGenerator();
  cout << "Generate inputs... ";
  vector<int> h_values1 =
    input_generator.generateSortedVector<int>(num_elements);
  vector<int> h_values2 =
    input_generator.generateSortedVector<int>(num_elements);
  cout << "done." << endl;
  vector<int> h_expected_sorted(num_total_elements);
  vector<int> h_actual_sorted(num_total_elements);

  benchmarkOnCPU(
      num_elements,
      h_values1.data(),
      h_values2.data(),
      h_expected_sorted.data()
  );

  benchmarkOnGPU(
      num_elements,
      h_values1.data(),
      h_values2.data(),
      h_actual_sorted.data(),
      num_trials
  );

  cout << "Check the result validation..." << endl;
  for (int i = 0; i < 10; ++i) {
    cout << "i=" << i << ": " << h_actual_sorted[i] << " ";
    cout << "i=" << i << ": " << h_expected_sorted[i] << endl;
  }
  cout << "... " << h_expected_sorted.back() << endl;
  for (int i = 0; i < h_expected_sorted.size(); ++i) {
    if (h_actual_sorted[i] != h_expected_sorted[i]) {
      cout << "i=" << i << ": " << h_actual_sorted[i] << " " << h_expected_sorted[i] << endl;
    }
  }
  assert(h_expected_sorted == h_actual_sorted);
}
