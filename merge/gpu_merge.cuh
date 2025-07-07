#pragma once


__global__ void merge_kernel_v1(
    const int num_elements,
    const int m,
    const int* values1,
    const int n,
    const int* values2,
    int* sorted
);
