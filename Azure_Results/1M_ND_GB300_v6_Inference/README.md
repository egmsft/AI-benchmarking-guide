# MLPerf Inference v5.1 on Azure ND_GB300_v6



## Quick Start
`git clone https://github.com/Azure/AI-benchmarking-guide.git`: clone AI Benchmarking guide repo 
`cd AI-benchmarking-guide/Azure_Results`: this will be our **working directory**

#### Download the models & datasets

create `models`, `data` , and `preprocessed_data` directories

- Download the [Llama 2 70B model](https://github.com/mlcommons/inference/tree/master/language/llama2-70b#get-model) inside the `models` directory. 
- Download the [datasets](https://github.com/mlcommons/inference/tree/master/language/llama2-70b#get-dataset) inside `data` directory
- [Prepare the datasets](https://github.com/mlcommons/inference_results_v5.1/tree/main/closed/Azure/code/llama2-70b/tensorrt#download-and-prepare-data)

#### Setup container

`mkdir build && cd build` inside the **working directory**

`git clone https://github.com/NVIDIA/TensorRT-LLM.git TRTLLM`: clone TensorRT-LLM and enter the directory

Edit `TRTLLM/docker/Makefile` lines 135 and 136:
- `SOURCE_DIR        ?= AI-benchmarking-guide/Azure_Results` (make sure it is an absolute path to the **working directory**)
- `CODE_DIR          ?= /work`

`make -C docker build` this will build a container tagged tensorrt_llm/devel:latest

`make -C docker run` this will launch a container. Once inside, run:

`cd 1M_ND_GB300_v6_Inference/build/TRTLLM`

`python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt --benchmarks --cuda_architectures "103-real" --no-venv --clean`: build trtllm wheel

`pip install build/tensorrt_llm-1.1.0rc6-cp312-cp312-linux_aarch64.whl`

Go back to the `1M_ND_GB300_v6_Inference` dir

`make clone_loadgen && make build_loadgen`: build loadgen

`git clone https://github.com/NVIDIA/mitten.git ./build/mitten && pip install build/mitten`: build mitten

`pip install -r docker/common/requirements/requirements.llm.txt`

`export MLPERF_SCRATCH_PATH=/work`

`make link_dirs`

`export SYSTEM_NAME=ND_GB300_v6`

`make run_llm_server RUN_ARGS="--core_type=trtllm_endpoint --benchmarks=llama2-70b --scenarios=Offline"`

`make run_harness RUN_ARGS="--core_type=trtllm_endpoint --benchmarks=llama2-70b --scenarios=Offline"`
