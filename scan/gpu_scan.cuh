#pragma once

using uint = unsigned int;

void launch_kernel_scan_v1(
    const int num_elements,
    uint* d_values,
    uint* d_prefix_sum,
    const int num_blocks,
    const int num_threads
);

void launch_kernel_scan_v2(
    const int num_elements,
    uint* d_values,
    uint* d_prefix_sum,
    const int num_blocks,
    const int num_threads
);
