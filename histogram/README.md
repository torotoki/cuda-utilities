# 参考

## 公式サンプルの出力

```
[[histogram]] - Starting...
GPU Device 0: "Ada" with compute capability 8.9

CUDA device [NVIDIA GeForce RTX 4060 Ti] has 34 Multi-Processors, Compute 8.9
Initializing data...
...allocating CPU memory.
...generating input data
...allocating GPU memory and copying input data

Starting up 64-bin histogram...

Running 64-bin GPU histogram for 67108864 bytes (16 runs)...

histogram64() time (average) : 0.00029 sec, 233574.4606 MB/sec

histogram64, Throughput = 233574.4606 MB/s, Time = 0.00029 s, Size = 67108864 Bytes, NumDevsUsed = 1, Workgroup = 64

Validating GPU results...
 ...reading back GPU results
 ...histogram64CPU()
 ...comparing the results...
 ...64-bin histograms match

Shutting down 64-bin histogram...


Initializing 256-bin histogram...
Running 256-bin GPU histogram for 67108864 bytes (16 runs)...

histogram256() time (average) : 0.00026 sec, 257801.1606 MB/sec

histogram256, Throughput = 257801.1606 MB/s, Time = 0.00026 s, Size = 67108864 Bytes, NumDevsUsed = 1, Workgroup = 192

Validating GPU results...
 ...reading back GPU results
 ...histogram256CPU()
 ...comparing the results
 ...256-bin histograms match

Shutting down 256-bin histogram...


Shutting down...

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.

[histogram] - Test Summary
Test passed
```
