#include <cuda.h>
#include <cstdio>
#include <cstdlib>

CUmodule loadPTX(const char* filename) {
  CUmodule module;
  CUresult res = cuModuleLoad(&module, filename);
  if (res != CUDA_SUCCESS) {
    printf("Failed to load PTX module\n");
    exit(EXIT_FAILURE);
  }
  return module;
}

int main() {
  // CUDA Driver API
  cuInit(0);

  CUdevice device;
  cuDeviceGet(&device, 0);

  CUcontext context;
  cuCtxCreate(&context, 0, device);

  CUmodule module = loadPTX("hello.ptx");
  CUfunction kernel;
  cuModuleGetFunction(&kernel, module, "hello_ptx_kernel");

  void* args[] = { nullptr };

  // 1 block, 1 thread で kernel を起動
  // CUresult cuLaunchKernel ( CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams, void** extra )
  cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, 0, args, 0);
  cuCtxSynchronize();

  printf("PTX kernel executed successfully!\n");

  cuModuleUnload(module);
  cuCtxDestroy(context);

  return 0;
}

