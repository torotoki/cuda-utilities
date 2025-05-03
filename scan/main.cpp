#include <iostream>
#include <random>
#include <cassert>
#include <stdexcept>
#include <chrono>

#include <cuda_runtime.h>
#include "../cuda_utility.hpp"

using uint = unsigned int;
using namespace std;

double benchmarkOnGPU(
    const int num_elements,
    const uint* h_values,
    uint* d_values,
    uint* h_prefix_sum,
    uint* d_prefix_sum,
    const int kernel_version = 1,
    const int num_trials = 10
) {
  const size_t size = num_elements * sizeof(uint);
  chrono::system_clock::time_point start, end;
  checkCudaErrors(cudaMalloc(&d_values, h_values, size));

  start = chrono::system_clock::now();
  switch (kernel_version) {
    case 1:
      launch_kernel_scan_v1(d_values, d_prefix_sum);
      break;
    default:
      throw std::invalid_argument("Invalid kernel version.");
  }
  end = chrono::system_clock::now();

  double elapsed_time_msec =
      chrono::duration<double, std::milli>(end - start).count();
  checkCudaErrors(cudaMalloc(&h_prefix_sum, d_prefix_sum, size));

  return elapsed_time_msec;
}
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
    for (uint i = 1; i < num_elements; ++i) {
      prefix_sum[i] = prefix_sum[i-1] + values[i-1];
    }
  }
  end = chrono::system_clock::now();

  elapsed_time_msec =
    chrono::duration<double, std::milli>(end - start).count();
  return elapsed_time_msec;
}

int main() {
  const int kernel_version = 0;
  std::mt19937 gen(42);
  std::uniform_int_distribution<> dist(0, 32767);

  // 13 * 2^22 * 4 = 4 * 52 MB
  const uint num_elements = 13 * (1<<22);
  vector<uint> h_values(num_elements);
  vector<uint> h_expected_prefix_sum(num_elements);
  vector<uint> h_actual_prefix_sum(num_elements);
  uint* d_values;
  uint* d_prefix_sum;

  for (uint i = 0; i < num_elements; ++i) {
    h_values[i] = dist(gen);
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
      elapsed_time_msec = benchmarkOnGPU(
          num_elements,
          h_values.data(),
      );
      break;
    default:
      throw std::invalid_argument("Invalid kernel version.");
  }

  assert(h_expected_prefix_sum == h_actual_prefix_sum);

  cout << "kernel version: " << kernel_version << endl;
  cout << "Time: " << elapsed_time_msec << " msec" << endl;
}
