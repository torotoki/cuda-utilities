#include <iostream>
#include <random>
#include <vector>
#include <cassert>
#include <stdint.h>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>

//#include "cpu_histogram.hpp"
#include "../cuda_utility.hpp"

using namespace std;

const float eps = 0.0001f;

using uint = unsigned int;

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
    const int* h_assignments,
    const uint num_elements,
    const uint num_bins,
    uint* h_histogram,
    bool print_log = false
) {
  chrono::system_clock::time_point start, end;
  double elapsed_time_msec;

  uint* d_assignemnts;  // input
  uint* d_histogram;    // output

  // Transfer data to the device memory
  checkCudaErrors(cudaMalloc(&d_assignments, num_elements*sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_histogram, num_bins*sizeof(uint)));
  
  checkCudaErrors(cudaMemcpy(
          d_assignments,
          h_assignments,
          num_elements*sizeof(int),
          cudaMemcpyDefault)
  );
  checkCudaErrors(cudaMemcpy(
          d_histogram,
          h_histogram,
          num_elements*sizeof(uint),
          cudaMemcpyDefault)
  );

  // compute
  start = chrono::system_clock::now();
  for (int trial_id = 0; trial_id < num_trial; ++trial_id) {
    // virtual one-dimentional CUDA kernel launching
    const int num_threads = 1024;
    const int num_blocks = (num_elements + num_threads - 1) / num_threads;   
    histogram_kernel_v1<<<num_blocks, num_threads>>>(
        d_assignments,
        num_elements,
        d_histogram,
        num_bins
    );
  }
}

int main(int argc, char* argv[]) {
  const uint max_func_id = 4;  // 要らないかも
  const uint num_bins = 256;
  const uint log_num_elements = 28;
  const uint num_elements = 1 << 28;  // 256M elements

  const uint num_trials = 10;
  vector<int> h_assignments(num_elements);
  vector<uint> h_expected_bin_counts(num_bins);
  vector<uint> h_actual_bin_counts(num_bins);
  std::mt19937 random_generator(42);
  std::uniform_int_distribution<> dist(0, num_bins - 1);

  const int kernel_version = 0;
  
  cout << "#elements: " << num_elements << endl;
  cout << "#bins: " << num_bins << endl;

  for (int i = 0; i < num_elements; ++i) {
    h_assignments[i] = dist(random_generator);
  }

  double time;
  if (kernel_version == 0) {
    time = histogramBenchmarkOnCPU(
        num_trials,
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
