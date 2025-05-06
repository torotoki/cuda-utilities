# Prefix Sum (aka. Scan)


v0: CPU上で計算（逐次計算）:
```
$ ./main
num_elements: 1048576
kernel version: 0
10th elem: 10
(expected): 10
Last element: 1048576
Last element (expected): 1048576
Time: 19.709 msec
```

v1: GPU上で計算（cooperative_groups + global memory, エラー）:
```
$ ./main
num_elements: 1048576
kernel version: 1
gpu_scan_v1.cu(44) : CUDA Runtime API error 720: too many blocks in cooperative launch.
```

v2: GPU上で計算（shared memory + hierarchical scan）
```
$ ./main
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
$ ./main
num_elements: 1048576
kernel version: 2
1048576
10th elem: 10
(expected): 10
Last element: 1048576
Last element (expected): 1048576
Time: 3.6838 msec

$ ./main
num_elements: 1048576
kernel version: 2
1048576
10th elem: 10
(expected): 10
Last element: 1048576
Last element (expected): 1048576
Time: 4.13 msec
```

v3: GPU上で計算（Brent-Kung kernel + shared memory）
```
./main
num_elements: 1048576
kernel version: 3
2097152
10th elem: 20
(expected): 20
Last element: 2097152
Last element (expected): 2097152
Time: 4.3736 msec
```

v4: GPU上で計算（Brent-Kung kernel + shared memory + branch divergence 削減）
```
./main
num_elements: 1048576
kernel version: 4
2097152
10th elem: 20
(expected): 20
Last element: 2097152
Last element (expected): 2097152
Time: 2.8254 msec
./main
num_elements: 1048576
kernel version: 4
2097152
10th elem: 20
(expected): 20
Last element: 2097152
Last element (expected): 2097152
Time: 3.9828 msec
```

けっこうムラがあるが、3msec台を付けることも多くv3より早いことが多かった

