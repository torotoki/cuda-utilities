#include <iostream>
#include <random>
#include <cassert>
#include <stdexcept>
#include <chrono>

#include <cuda_runtime.h>
#include "../common/cuda_utility.hpp"
#include "gpu_scan.cuh"

using uint = unsigned int;
using namespace std;

double benchmarkOnGPU(
    const int num_elements,
    const uint* h_values,
    uint* d_values,
    uint* h_prefix_sum,
    uint* d_prefix_sum,
    const int kernel_version,
    const int num_trials = 10
) {
  assert(num_elements % 1024 == 0 && num_elements >= 1024);

  const size_t size = num_elements * sizeof(uint);
  chrono::system_clock::time_point start, end;

  checkCudaErrors(cudaMalloc(&d_values, size));
  checkCudaErrors(cudaMalloc(&d_prefix_sum, size));
  checkCudaErrors(cudaMemcpy(d_values, h_values, size, cudaMemcpyDefault));

  int num_blocks, num_threads;
  start = chrono::system_clock::now();
  switch (kernel_version) {
    case 1:
      num_blocks = num_elements / 1024;
      num_threads = 1024;
      launch_kernel_scan_v1(
          num_elements,
          d_values,
          d_prefix_sum,
          num_blocks,
          num_threads
      );
      break;
    case 2:
      num_blocks = num_elements / 1024;
      num_threads = 1024;
      launch_kernel_scan_v2(
          num_elements,
          d_values,
          d_prefix_sum,
          num_blocks,
          num_threads
      );
      break;
    case 3:
      num_blocks = num_elements / 1024;
      num_threads = 1024;
      launch_kernel_scan_v3(
          num_elements,
          d_values,
          d_prefix_sum,
          num_blocks,
          num_threads
      );
      break;
    case 4:
      num_blocks = num_elements / 1024;
      num_threads = 1024;
      launch_kernel_scan_v4(
          num_elements,
          d_values,
          d_prefix_sum,
          num_blocks,
          num_threads
      );
      break;
    default:
      throw std::invalid_argument("Invalid kernel version.");
  }
  end = chrono::system_clock::now();

  double elapsed_time_msec =
      chrono::duration<double, std::milli>(end - start).count();
  checkCudaErrors(cudaMemcpy(h_prefix_sum, d_prefix_sum, size, cudaMemcpyDefault));
  cout << h_prefix_sum[num_elements - 1] << endl;
  checkCudaErrors(cudaFree(d_values));
  checkCudaErrors(cudaFree(d_prefix_sum));

  return elapsed_time_msec;
}

//
// Inclusive Scan
//
double benchmarkOnCPU(
    const int num_elements,
    const uint* values,
    uint* prefix_sum,
    const int num_trials = 10
) {
  assert(num_elements > 1);
  
  chrono::system_clock::time_point start, end;
  double elapsed_time_msec;

  start = chrono::system_clock::now();
  for (int trial_id = 0; trial_id < num_trials; ++trial_id) {
    prefix_sum[0] = values[0];
    for (uint i = 1; i < num_elements; ++i) {
      prefix_sum[i] = values[i] + prefix_sum[i-1];
    }
  }
  end = chrono::system_clock::now();

  elapsed_time_msec =
    chrono::duration<double, std::milli>(end - start).count();
  return elapsed_time_msec;
}

int main() {
  const int kernel_version = 4;
  std::mt19937 gen(42);
  std::uniform_int_distribution<> dist(0, 32767);

  // 1 * 2^20 * 4 MB
  const uint num_elements = 1 * (1<<20);
  vector<uint> h_values(num_elements);
  vector<uint> h_expected_prefix_sum(num_elements);
  vector<uint> h_actual_prefix_sum(num_elements);
  uint* d_values;
  uint* d_prefix_sum;
  
  cout << "num_elements: " << num_elements << endl;
  cout << "kernel version: " << kernel_version << endl;

  for (uint i = 0; i < num_elements; ++i) {
    // NOTE: easily overflows
    // h_values[i] = dist(gen);
    h_values[i] = 2;
  }

  benchmarkOnCPU(
      num_elements,
      h_values.data(),
      h_expected_prefix_sum.data()
  );

  double elapsed_time_msec;
  switch (kernel_version) {
    case 0:
      elapsed_time_msec = benchmarkOnCPU(
          num_elements,
          h_values.data(),
          h_actual_prefix_sum.data()
      );
      break;
    case 1:
    case 2:
    case 3:
    case 4:
      elapsed_time_msec = benchmarkOnGPU(
          num_elements,
          h_values.data(),
          d_values,
          h_actual_prefix_sum.data(),
          d_prefix_sum,
          kernel_version
      );
      break;
    default:
      throw std::invalid_argument("Invalid kernel version.");
  }

  cout << "10th elem: " << h_actual_prefix_sum[9] << endl;
  cout << "(expected): "
    << h_expected_prefix_sum[9] << endl;
  cout << "Last element: " << h_actual_prefix_sum[num_elements - 1] << endl;
  cout << "Last element (expected): "
    << h_expected_prefix_sum[num_elements - 1] << endl;
  assert(h_expected_prefix_sum == h_actual_prefix_sum);

  cout << "Time: " << elapsed_time_msec << " msec" << endl;
}
