#pragma once

__global__ void reduce_gpu_kernel(const int*, size_t, int*);
int call_reduction(const int*, size_t, int*, int*, int, int, int);

