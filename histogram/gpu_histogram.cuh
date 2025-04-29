#pragma once

using uint = unsigned int;


__global__ void histogram_kernel_v1(const uint*, const uint, uint*, const uint);
__global__ void histogram_kernel_v2(const uint*, const uint, uint*, const uint);
