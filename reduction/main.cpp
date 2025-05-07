#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>

#include "reduction.hpp"
#include "gpu_reduction.cuh"
#include "common/cuda_utility.hpp"

using namespace std;

const float eps = 0.0001f;


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
  const int kernel_version = 3;
  const int num_trials = 100;
  const int num_blocks = 1 << 16;
  const int num_threads = 512;
  const int n = num_blocks * num_threads;
  const size_t array_size = sizeof(int) * n;
  const size_t tmp_size = sizeof(int) * num_blocks;

  cout << "n: " << n << endl;
  vector<int> h_input(n);
  vector<int> h_tmp_out(num_blocks);

  for (int i = 0; i < n; ++i) {
    h_input[i] = 1;
  }
  assert(h_input[0] == 1);

  int expected_value = n;
  cout << "input size: " << h_input.size() << endl;
  double time;
  if (kernel_version == 0) {
    time = reduceBenchmarkOnCPU(
        h_input.size(),
        h_input.data(),
        num_trials,
        expected_value,
        true
    );
  } else {
    time = reduceBenchmarkOnGPU(
        h_input.size(),
        h_input.data(),
        num_blocks,
        num_threads,
        num_trials,
        expected_value,
        true,
        kernel_version
    );
  }

  cout << "Time (" << num_trials << " iterations) on version " 
      << kernel_version << ": " << time << " msec" << endl;
}
