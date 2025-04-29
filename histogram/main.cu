#include <iostream>
#include <random>
#include <vector>
#include <cassert>
#include <stdint.h>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../cuda_utility.hpp"
//#include "cpu_histogram.hpp"
#include "gpu_histogram.cuh"

using namespace std;


using uint = unsigned int;

bool checkEqualVectors(vector<uint> v1, vector<uint> v2) {
  if (v1.size() != v2.size())
    return false;

  size_t size = v1.size();
  for (size_t i=0; i < size; ++i) {
    if (v1[i] != v2[i]) {
      cout << "i=" << i << ", v1[i]=" << v1[i] << ", v2[i]=" << v2[i] << endl;
      return false;
    }
  }
  return true;
}

double histogramBenchmarkOnCPU(
    const int num_trials,
    const int* assignments,
    const uint num_elements,
    const uint num_bins,
    uint* histogram,
    bool print_log = false
) {
  chrono::system_clock::time_point start, end;
  double elapsed_time_msec;

  start = chrono::system_clock::now();
  for (int trial_id = 0; trial_id < num_trials; ++trial_id) {
    for (uint i=0; i < num_bins; ++i)
      histogram[i] = 0;

    for (uint i=0; i < num_elements; ++i) {
      const int bin_idx = assignments[i];
      histogram[bin_idx]++;
    }
  }
  end = chrono::system_clock::now();

  elapsed_time_msec = 
      static_cast<double>(
          chrono::duration_cast<chrono::microseconds>
          (end - start).count() / 1000.0);

  return elapsed_time_msec;
}

double histogramBenchmarkOnGPU(
    const int num_trials,
    const uint num_elements,
    const int* h_assignments,
    const uint num_bins,
    uint* h_histogram,
    const int kernel_version,
    bool print_log = false
) {
  chrono::system_clock::time_point start, end;
  double elapsed_time_msec;

  uint* d_assignments;  // input
  uint* d_histogram;    // output

  // Transfer data to the device memory
  checkCudaErrors(cudaMalloc(&d_assignments, num_elements*sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_histogram, num_bins*sizeof(uint)));
  
  checkCudaErrors(cudaMemcpy(
          d_assignments,
          h_assignments,
          num_elements * sizeof(int),
          cudaMemcpyDefault)
  );
  for (uint i=0; i < num_bins; i++) {
    h_histogram[i] = 0;
  }

  // compute
  start = chrono::system_clock::now();
  for (int trial_id = 0; trial_id < num_trials; ++trial_id) {
    // virtual one-dimentional CUDA kernel launching
    const int num_threads = 1024;
    const int num_blocks = (num_elements + num_threads - 1) / num_threads;   
  
    checkCudaErrors(cudaMemcpy(
            d_histogram,
            h_histogram,
            num_bins * sizeof(uint),
            cudaMemcpyDefault)
    );

    switch (kernel_version) {
      case 1:
        histogram_kernel_v1<<<num_blocks, num_threads>>>(
            d_assignments,
            num_elements,
            d_histogram,
            num_bins
        );
        break;
      case 2:
        histogram_kernel_v2<<<num_blocks, num_threads>>>(
            d_assignments,
            num_elements,
            d_histogram,
            num_bins
        );
        break;
      default:
        throw std::invalid_argument("Invalid kernel version.");
    }
  }
  end = chrono::system_clock::now();

  checkCudaErrors(cudaMemcpy(
        h_histogram,
        d_histogram,
        num_bins * sizeof(uint),
        cudaMemcpyDefault)
  );
  elapsed_time_msec = chrono::duration<double, std::milli>(end - start).count();
  
  return elapsed_time_msec;
}

int main(int argc, char* argv[]) {
  const uint num_bins = 256;
  const uint num_elements = 1 << 28;  // 256M elements

  const uint num_trials = 10;
  vector<int> h_assignments(num_elements);
  vector<uint> h_expected_histogram(num_bins);
  vector<uint> h_actual_histogram(num_bins);
  std::mt19937 random_generator(42);
  std::uniform_int_distribution<> dist(0, num_bins - 1);

  const int kernel_version = 2;
  
  cout << "#elements: " << num_elements << endl;
  cout << "#bins: " << num_bins << endl;

  for (int i = 0; i < num_elements; ++i) {
    h_assignments[i] = dist(random_generator);
  }

  // Running on CPU for verifying the results
  histogramBenchmarkOnCPU(
        num_trials,
        h_assignments.data(),
        num_elements,
        num_bins,
        h_expected_histogram.data()
  );

  double time;
  if (kernel_version == 0) {
    time = histogramBenchmarkOnCPU(
        num_trials,
        h_assignments.data(),
        num_elements,
        num_bins,
        h_actual_histogram.data()
    );
  } else {
    // GPU:
    time = histogramBenchmarkOnGPU(
        num_trials,
        num_elements,
        h_assignments.data(),
        num_bins,
        h_actual_histogram.data(),
        kernel_version
    );
    assert(checkEqualVectors(h_actual_histogram, h_expected_histogram));
  }

  cout << "Time (" << num_trials << " iterations) on version " 
      << kernel_version << ": " << time << " msec" << endl;
}
