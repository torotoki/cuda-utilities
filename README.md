CPU (baseline)
```
hkomatsu in ll-workload-landlord2-vszzd in hkomatsu/reduction/reduction took 3s 
> ./main
n: 33554432
input size: 33554432
actual value: 33554432
Time (100 iterations) on GPU: 7064.84
```

GPU(v1 kernel):
```
hkomatsu in ll-workload-landlord2-vszzd in hkomatsu/reduction/reduction took 25s 
> ./main
n: 33554432
input size: 33554432
actual value: 33554432
Time (100 iterations) on GPU: 129.46
(114~149くらい)
```

GPU(v2 kernel, 自作):
```
hkomatsu in ll-workload-landlord2-vszzd in hkomatsu/reduction/reduction took 25s 
> ./main
n: 33554432
input size: 33554432
actual value: 33554432
Time (100 iterations) on version 2: 124.856 msec
(105~146くらい。速い！)
```

GPU(v3 kernel):
```
hkomatsu in ll-workload-landlord2-vszzd in hkomatsu/reduction/reduction took 3s 
> ./main
n: 33554432
input size: 33554432
actual value: 33554432
Time (100 iterations) on version 3: 78.747 msec
hkomatsu in ll-workload-landlord2-vszzd in hkomatsu/reduction/reduction 
> ./main
n: 33554432
input size: 33554432
actual value: 33554432
Time (100 iterations) on version 3: 58.053 msec
(めちゃ早い！)
```

