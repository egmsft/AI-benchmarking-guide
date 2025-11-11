# Azure ND GB300 v6 Benchmark Results

## System Specifications

| GPU           | NVIDIA GB300 288GB |
|---------------|-------------------|
| CPU           | ARM Neoverse-V2 |
| Ubuntu        |   24.04  |
| CUDA          |   13.0  |
| NVIDIA Driver | 580.82.07  |
| VBIOS         | 97.10.4A.00.05 |
| NCCL          |    2.25.1 |


## Microbenchmarks
### GEMM CuBLASLt (FP8)

The results shown below are with random initialization (best representation of real-life workloads), FP8, and 10,000 warmup iterations.

|     m     |     n     |     k     | ND GB300 v6 (TFLOPS)      |
|-----------|-----------|-----------|------------|
|   1024    |   1024    |   1024    | 553.5     |
|   2048    |   2048    |   2048    | 2288.9    |
|   4096    |   4096    |   4096    | 3257.0    |
|   8192    |   8192    |   8192    | 3286.3    |
|  16384    |  16384    |  16384    | 3308.5    |
|  32768    |  32768    |  32768    | 3271.3    |
| \---------- | \-------- | \------- | \--------------------- |
|   1024    |   2145    |   1024    | 1115.1    |
|   6144    |  12288    |  12288    | 3279.5    |
| 802816    |   192     |   768     | 1717.1    |


### GEMM CuBLASLt (FP4)

The results shown below are with random initialization (best representation of real-life workloads), FP4, and 10,000 warmup iterations.

|     m     |     n     |     k     | ND GB300 v6 (TFLOPS)      |
|-----------|-----------|-----------|------------|
|   1024    |   1024    |   1024    |  316.4    |
|   2048    |   2048    |   2048    |  1678.0  |
|   4096    |   4096    |   4096    |  5417.0   |
|   8192    |   8192    |   8192    |  8155.9   |
|  16384    |  16384    |  16384    |  8640.3   |
|  32768    |  32768    |  32768    |  8555.1   |
| \---------- | \-------- | \------- | \--------------------- |
|   1024    |   2145    |   1024    |  549.0   |
| 802816    |   192     |   768     |   2220.7  |


### Flash Attention 2.0

The performance (in TFLOPS), in table below, represents the performance for a head dimension of 128, a batch size of 2, and a sequence length of 8192.

|       | ND GB300 v6 (TFLOPS) |
| ----- | ----------------- |
| Standard Attention(PyTorch)  | 249.5   |
| Flash Attention 2.0   | 399.7  |

### NV Bandwidth

|                       | ND GB300 v6 (GB/s) |
| --------------------- | ----------------- |
| Host to Device        | 211                |
| Device to Host        | 193                |
| Device to Device read |  1530              |


### NCCL Bandwidth

The values (in GB/s) are the bus bandwidth values obtained from the NCCL AllReduce (Ring algorithm) tests in-place operations, varying from 1KB to 8GB of data.

| Message Size (Bytes) | ND GB300 v6 (GB/s) |
|--------------|-----------------|
| 1K           | 0.10            |
| 2K           | 0.19            |
| 4K           | 0.38            |
| 8K           | 0.77            |
| 16K          | 1.52            |
| 32K          | 2.91            |
| 65K          | 5.92            |
| 132K         | 11.28           |
| 256K         | 21.72           |
| 524K         | 38.14           |
| 1M           | 74.79           |
| 2M           | 116.31          |
| 4M           | 150.21          |
| 8M           | 257.00          |
| 16M          | 335.19          |
| 33M          | 417.65          |
| 67M          | 550.71          |
| 134M         | 575.25          |
| 268M         | 593.70          |
| 536M         | 624.45          |
| 1G           | 646.58          |
| 2G           | 669.40          |
| 4G           | 674.40          |
| 8G           | 680.45          |
