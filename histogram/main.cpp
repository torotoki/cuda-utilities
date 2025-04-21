#include <iostream>
#include <random>
#include <vector>
#include <cassert>
#include <stdint.h>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cpu_histogram.hpp"
#include "cuda_utility.hpp"

using namespace std;

const float eps = 0.0001f;

/**
time = histogramBenchmarkOnCPU(
    h_assign.data(),
    num_elements,
    num_bins,
    h_expected_bin_counts.data()
);
*/
using uint = unsigned int;

double histogramBenchmarkOnCPU(
    const int* assignments,
    const uint num_elements,
    const uint num_bins,
    uint* histogram
) {
  for (uint i=0; i < num_bins; ++i)
    histogram[i] = 0;

  for (uint i=0; i < num_elements; ++i) {
    const int bin_idx = assignments[i];
    histogram[bin_idx]++;
  }
}


double reduceBenchmarkOnCPU(
  const size_t size,
  const int* values,
  const int num_trials,
  const int expected_value,
  bool print_log
) {
  chrono::system_clock::time_point start, end;
  double elapsed_time_msec;
  int actual_value;

  start = chrono::system_clock::now();
  for (int i = 0; i < num_trials; ++i) {
    actual_value = reduce(values, size);
  }
  end = chrono::system_clock::now();

  if (print_log) {
    cout << "actual value: " << actual_value << endl;
  }
  assert(abs(actual_value - expected_value) < eps);
  elapsed_time_msec =
      static_cast<double>(
          chrono::duration_cast<chrono::microseconds>
	  (end - start).count() / 1000.0
      );
  if (print_log) {
    // bandwidth (GB/s)
    double bandwidth_gbs = (size * num_trials) / elapsed_time_msec / 1e6;
    cout << "bandwidth (GB/s) without memcpy: " << bandwidth_gbs << endl;
  }
  return elapsed_time_msec;
}

double reduceBenchmarkOnGPU(
  const size_t num_elements,
  const int* values,
  const int num_blocks,
  const int num_threads,
  const int num_trials,
  const int expected_value,
  const bool print_log,
  const int kernel_version = 1
) {
  assert(num_elements > 0);
  chrono::system_clock::time_point start, end;
  double elapsed_time_msec;
  int actual_value = -100;

  int size = sizeof(int) * num_elements;
  int* h_out = (int*)malloc(size);

  int* d_values = nullptr;
  int* d_out = nullptr;
  checkCudaErrors(cudaMalloc(&d_values, size));
  checkCudaErrors(cudaMalloc(&d_out, size));
  checkCudaErrors(cudaMemcpy(d_values, values, size, cudaMemcpyDefault));
  
  call_reduction(
    d_values, size, d_out,
    h_out, num_blocks, num_threads,
    kernel_version
  );

  // Reset returned value
  h_out[0] = -100;
  
  start = chrono::system_clock::now();
  for (int i = 0; i < num_trials; ++i) {
    actual_value = call_reduction(
      d_values, size, d_out,
      h_out, num_blocks, num_threads,
      kernel_version
    );
    // Make sure to wait until the CUDA kernel is computed
    cudaDeviceSynchronize();
  }
  end = chrono::system_clock::now();

  if (print_log)
    cout << "actual value: " << actual_value << endl;
  assert(abs(actual_value - expected_value) < eps);
  elapsed_time_msec = 
    static_cast<double>(
      chrono::duration_cast<chrono::microseconds>(end - start).count() /
      1000.0
    );
  
  if (print_log) {
    // bandwidth (GB/s)
    double bandwidth_gbs = (size * num_trials) / elapsed_time_msec / 1e6;
    cout << "bandwidth (GB/s) without memcpy: " << bandwidth_gbs << endl;
  }
  cudaFree(d_values);
  cudaFree(d_out);
  free(h_out);

  return elapsed_time_msec;
}

int main(int argc, char* argv[]) {
  const uint max_func_id = 4;  // 要らないかも
  const uint num_bins = 256;
  const uint log_num_elements = 28;
  const uint num_elements = 1 << 28;  // 256M elements

  const uint n_trials = 10;
  vector<int> h_assignments(num_elements);
  vector<uint> h_expected_bin_counts(n_bins);
  vector<uint> h_actual_bin_counts(n_bins);
  std::mt19937 random_generator(42);
  std::uniform_int_distribution<> dist(0, n_bins - 1);

  const int kernel_version = 0;
  
  cout << "#elements: " << num_elements << endl;
  cout << "#bins: " << num_bins << endl;

  for (int i = 0; i < n; ++i) {
    h_assignments[i] = dist(random_generator);
  }

  double time;
  if (kernel_version == 0) {
    /*
    time = reduceBenchmarkOnCPU(
        h_input.size(),
        h_input.data(),
        num_trials,
        expected_value,
        true
    );*/
    time = histogramBenchmarkOnCPU(
        h_assignments.data(),
        num_elements,
        num_bins,
        h_expected_bin_counts.data()
    );
  } else {
    // GPU:
    // time = ...
  }

  cout << "Time (" << num_trials << " iterations) on version " 
      << kernel_version << ": " << time << " msec" << endl;
}
