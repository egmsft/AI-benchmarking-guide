# Azure ND H200 v5 Benchmark Results

## System Specifications

| GPU           | NVIDIA H200 141GB |
|---------------|-------------------|
| CPU           | Intel(R) Xeon(R) Platinum 8480C |
| Ubuntu        |   22.04  |
| CUDA          |   12.6  |
| NVIDIA Driver | 560.28.03  |
| VBIOS         | 96.00.9D.00.02 |
| NCCL          |    2.20.5  |
| PyTorch       |    2.5.1   |


## Microbenchmarks
### GEMM CuBLASLt 

The results shown below are with random initialization (best representation of real-life workloads), FP8, and 10,000 warmup iterations.

| m           | n         | k        | ND H200 V5 (TFLOPS)    |
| ----------- | --------- | -------- | ---------------------- |
| 1024        | 1024      | 1024     | 290.8                  |
| 2048        | 2048      | 2048     | 1013.4                 |
| 4096        | 4096      | 4096     | 1249.1                 |
| 8192        | 8192      | 8192     | 1300.6                 |
| 16384       | 16384     | 16384    | 1366.3                 |
| 32768       | 32768     | 32768    | 1349.5                 |
| \---------- | \-------- | \------- | \--------------------- |
| 1024        | 2145      | 1024     | 407.9                  |
| 6144        | 12288     | 12288    | 1318.0                 |
| 802816      | 192       | 768      | 717.5                  |

### HBM Bandwidth

|       | ND H200 V5 (TB/s) |
| ----- | ----------------- |
| Copy  | 4.01              |
| Mul   | 4.00              |
| Add   | 4.15              |
| Triad | 4.15              |
| Dot   | 4.48              |


### Flash Attention 2.0

The performance (in TFLOPS), in table below, represents the performance for a head dimension of 128, a batch size of 2, and a sequence length of 8192.

|       | ND H200 V5 (TFLOPS) |
| ----- | ----------------- |
| Standard Attention(PyTorch)  | 161.5   |
| Flash Attention 2.0   | 329.3  |

### NV Bandwidth

|                       | ND H200 V5 (GB/s) |
| --------------------- | ----------------- |
| Host to Device        | 52                |
| Device to Host        | 52                |
| Device to Device read | 671               |


### FIO Tests

| Test             | Batch Size(Bytes) | ND H200 V5 (GB/s) |
| ---------------- | ----------------- | ----------------- |
| Sequential read  | 1M                | 55.4              |
| Sequential read  | 512k              | 55.6              |
| Random read      | 1k                | 1.26              |
| Sequential read  | 1k                | 1.56              |
| Sequential write | 1M                | 33.8              |
| Sequential write | 512k              | 33.9              |
| Random write     | 1k                | 0.37              |
| Sequential write | 1k                | 0.39              |


## NCCL Bandwidth

The values (in GB/s), in the table 6 and figure 5 below, are the bus bandwidth values obtained from the NCCL AllReduce (NVLS algorithm) tests in-place operations, varying from 1KB to 8GB of data.

| Message Size (Bytes) | ND H200 V5 (GB/s) |
| -------------------- | ----------------- |
| 1K                   | 0.06              |
| 2K                   | 0.12              |
| 4K                   | 0.23              |
| 8K                   | 0.47              |
| 16K                  | 0.93              |
| 32K                  | 1.84              |
| 65K                  | 3.72              |
| 132K                 | 7.39              |
| 256K                 | 14.75             |
| 524K                 | 29.38             |
| 1M                   | 52.96             |
| 2M                   | 88.78             |
| 4M                   | 138.44            |
| 8M                   | 181.94            |
| 16M                  | 240.41            |
| 33M                  | 300.91            |
| 67M                  | 368.11            |
| 134M                 | 412.45            |
| 268M                 | 432.24            |
| 536M                 | 442.72            |
| 1G                   | 467.58            |
| 2G                   | 472.57            |
| 4G                   | 477.41            |
| 8G                   | 479.98            |

## End-to-End Inference Workloads - TensorRT-LLM



### LLAMA 3.1 (8B)

Performance results for LLAMA 3.1 (8B) with FP8 quantization, 1000 requests.

| tp size | input len | output len | throughput(tokens/sec) |
|---------|-----------|------------|------------------------|
| 1       | 128       | 128        | 17853                  |
| 1       | 128       | 1024       | 23880                  |
| 1       | 128       | 2048       | 21255                  |
| 1       | 500       | 2000       | 18357                  |
| 1       | 1024      | 1024       | 15043                  |
| 1       | 2048      | 2048       | 9736                   |

### LLAMA 3 (70B)

Performance results for LLAMA 3 (70B) with FP8 quantization, 1000 requests.

| tp size | input len | output len | throughput(tokens/sec) |
|---------|-----------|------------|------------------------|
| 8       | 128       | 128        | 11973                  |
| 8       | 128       | 1024       | 18369                  |
| 8       | 128       | 2048       | 18219                  |
| 8       | 500       | 2000       | 15911                  |
| 8       | 1024      | 1024       | 11302                  |
| 8       | 2048      | 2048       | 10371                  |

### LLAMA 3 (405B)

Performance results for LLAMA 3 (405B) with FP8 quantization, 1000 requests.

| tp size | input len | output len | throughput(tokens/sec) |
|---------|-----------|------------|------------------------|
| 8       | 128       | 128        | 3148                   |
| 8       | 128       | 1024       | 4981                   |
| 8       | 128       | 2048       | 5098                   |
| 8       | 500       | 2000       | 4447                   |
| 8       | 1024      | 1024       | 3050                   |
| 8       | 2048      | 2048       | 2859                   |
