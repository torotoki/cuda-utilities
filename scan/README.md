# Prefix Sum (aka. Scan)


CPU上で計算（逐次計算）:
```
➜  scan git:(main) ✗ ./main
num_elements: 1048576
kernel version: 0
10th elem: 10
(expected): 10
Last element: 1048576
Last element (expected): 1048576
Time: 19.709 msec
```

GPU上で計算（cooperative_groups + global memory, エラー）:
```
➜  scan git:(main) ✗ ./main
num_elements: 1048576
kernel version: 1
gpu_scan_v1.cu(44) : CUDA Runtime API error 720: too many blocks in cooperative launch.
```

GPU上で計算（shared memory + hierarchical scan）
```
➜  scan git:(main) ✗ ./main
num_elements: 1048576
kernel version: 2
1048576
10th elem: 10
(expected): 10
Last element: 1048576
Last element (expected): 1048576
Time: 4.055 msec
```
CPUよりも5倍程度早い。

```
➜  scan git:(main) ✗ ./main
num_elements: 1048576
kernel version: 2
1048576
10th elem: 10
(expected): 10
Last element: 1048576
Last element (expected): 1048576
Time: 3.6838 msec
➜  scan git:(main) ✗ ./main
num_elements: 1048576
kernel version: 2
1048576
10th elem: 10
(expected): 10
Last element: 1048576
Last element (expected): 1048576
Time: 4.13 msec
```
