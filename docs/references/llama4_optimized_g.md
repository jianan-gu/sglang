# Optimized Llama 4 models Get Started on Intel Gaudi 3 AI accelerators

Intel Gaudi 3 AI accelerators are designed from the ground up for AI workloads and benefit from Tensor cores, and eight large Matrix Multiplication Engines compared to many small matrix multiplication units that a GPU has.
This leads to reduced data transfers and greater energy efficiency. 
The new Llama 4 Maverick model can be run on a single Gaudi 3 node with 8 accelerators.  

## Performance Benchmark Overview
For this performance evaluation, we will be using vLLM's Online Serving mode. Below is a set of sample commands to help you get started with running vLLM alongside the Llama 4 models.
### Usage Instructions
The following guide outlines how to deploy various Llama 4 models on Gaudi 3 using vLLM's online serving capabilities.
#### Pre-requisites:
Before getting started, ensure:
- You have Intel Gaudi 3 (8 cards) system and Habana driver setup	
- Pull pre-built docker image with SynapseAI installed
```
docker pull vault.habana.ai/gaudi-docker/1.20.0/ubuntu24.04/habanalabs/pytorch-installer-2.6.0:latest
```

##### Download Llama models through Hugging face: Llama 4 - [a meta-llama Collection](https://huggingface.co/collections/meta-llama/llama-4-67f0c30d9fe03840bc9d0164)
```
huggingface-cli login
huggingface-cli download meta-llama/Llama-4-Scout-17B-16E --local-dir $LLAMA_SCOUT
huggingface-cli download meta-llama/Llama-4-Maverick-17B-128E-Instruct --local-dir $LLAMA_MAVERICK
```
##### Prepare the software dependencies
```
git clone https://github.com/HabanaAI/vllm-fork -b llama4
cd vllm-fork; pip install -r requirements-hpu.txt; VLLM_TARGET_DEVICE=hpu pip install -e . --no-build-isolation;
pip uninstall vllm-hpu-extension; pip install git+https://github.com/HabanaAI/vllm-hpu-extension.git@145c63d

# install llama 4 dependencies
pip install pydantic msgspec cachetools cloudpickle psutil zmq blake3 py-cpuinfo aiohttp openai uvloop fastapi uvicorn watchfiles partial_json_parser python-multipart gguf llguidance prometheus_client numba compressed_tensors

# install Intel Neural Compress for Llama 4 branch support
pip uninstall -y neural-compressor neural-compressor-pt
git clone -b dev/llama4_launch https://github.com/intel/neural-compressor.git
cd neural-compressor
pip install -e .
```

##### Step 1: Start Docker container
To run on Gaudi 3, run the docker as shown below:
```
docker run -it \
     --runtime=habana \
     --name llama4-vllm \
     -v /software:/software \
     -e HABANA_VISIBLE_DEVICES=all \
     -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
     --cap-add=sys_nice \
     --ipc=host \
     --net=host \
     vault.habana.ai/gaudi-docker/1.20.1/ubuntu24.04/habanalabs/pytorch-installer-2.6.0:latest /bin /bash
```
##### Step 2: Running Inference using benchmark script
```
cd vllm-fork/llama4-scripts/
./benchmark/benchmark-vllm-online.sh # For 1k/1k benchmark on Scout BF16 Model
./benchmark/benchmark-vllm-online-Maverick-FP8.sh # For 1k/1k benchmark on Maverick FP8 Model
```
