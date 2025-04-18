# Optimized SGlang Llama 4 models Get Started on CPUs
Optimized SGlang provides dedicated optimizations for running Llama 4 models faster on Xeon CPUs.

## 1. Environment Setup
```sh

# Create conda ENV
conda create -n sglang python=3.10
conda activate sglang

# GCC >= 11 is needed

# Install SGLang
git clone https://github.com/jianan-gu/sglang -b llama4_optimzed_cpu
cd sglang
pip install -e "python[all_cpu]"
cd ..

# Install some dependencies
conda install -y libsqlite=3.48.0
pip uninstall -y triton # uninstall incorrect version
pip uninstall transformers
pip install transformers==4.51.1
pip install triton==3.1
conda install -y gperftools -c conda-forge
pip install intel-openmp==2024.2.0
pip install transformers

# Build and install vllm for CPU following https://docs.vllm.ai/en/latest/getting_started/installation/cpu/index.html
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.6.4.post1
sudo apt-get install libnuma-dev # optional: conda install -y libnuma numactl
pip install cmake==3.31.2 wheel packaging ninja "setuptools-scm>=8" numpy nvidia-ml-py
pip install -v -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
VLLM_TARGET_DEVICE=cpu python setup.py develop
cd ..

# Build sgl-kernel
cd PATH/TO/SGlang
pip uninstall torch torchvision # uninstall incorrect version
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu # PT 2.6 is needed to build sgl-kernel
cd sgl-kernel/
python setup.py install
cd ..

```

## 2. Benchmark

### 2.1 Notes

The following command lines are for demonstration purposes. Modify the arguments and thread binding according to your requirements and CPU type.

#### 2.1.1  Core binding and tensor parallel degree setup

The env variable `SGLANG_CPU_OMP_THREADS_BIND` specifies the CPU cores dedicated to the OpenMP threads, and argument `--tp` sets the TP size (tensor parallel degree). Below examples are running with TP = 6. By changing `--tp` and `SGLANG_CPU_OMP_THREADS_BIND` accordingly, you could set TP size to other values, and specifiy the core binding for each rank. Please be aware that cross NUMA node memory access needs to be avoided when setting `SGLANG_CPU_OMP_THREADS_BIND`, or the performance would be severrely impacted.

#### 2.1.2 Preload libraries

Preload iomp and tcmalloc for better performance.

```sh
export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libtcmalloc.so
```

### 2.2 Download Llama4 models from Huggingface
```sh
# using huggingface-cli to download origin BF16 models
mkdir ./origin_model_path
huggingface-cli download meta-llama/Llama-4-Scout-17B-16E-Instruct --local-dir ./origin_model_path/Llama-4-Scout-17B-16E-Instruct
huggingface-cli download meta-llama/Llama-4-Maverick-17B-128E-Instruct --local-dir ./origin_model_path/Llama-4-Maverick-17B-128E-Instruct
```
### 2.3 Quantization

We can use [AutoRound](https://github.com/intel/auto-round) which is a novel quantization approach minimizing the accuracy loss.

```sh
git clone https://github.com/intel/auto-round -b enable_llama4_int8_baseline
cd auto-round/
pip install -e .[cpu]
sh run_llama4_quant.sh {origin_model_path}  {quant_model_dir} # the quantized model folder will be in {quant_model_dir}

# Note:
# 1. example: sh run_llama4_quant.sh ./model_dir/Llama-4-Maverick-17B-128E-Instruct ./quant_model_dir
# 2. quantized model is under {quant_model_dir}, like: ./quant_model_dir/Llama-4-Maverick-17B-128E-Instruct-w8g-1
```
### 2.4 Performance
#### 2.4.1 Throughout 
```sh
# TP = 6, 43 OpenMP threads of rank0 are bound on 0-42 CPU cores, and the OpenMP threads of rank1 are bound on 43-85 CPU cores, etc.

# download prompts files first
wget https://intel-extension-for-pytorch.s3.us-east-1.amazonaws.com/miscellaneous/llm/prompt_llama4.json

# {Quantized_model_path} is based on the above Quantization stage, example: ./quant_model_dir/Llama-4-Maverick-17B-128E-Instruct-w8g-1
Command:
MOE_QUANT_ONLY=1 SGLANG_CPU_OMP_THREADS_BIND="0-42|43-85|86-127|128-170|171-213|214-255" python -m sglang.bench_one_batch --batch-size 1 --input 1024 --output 1024 --model {Quantized_model_path} --trust-remote-code --device cpu --mem-fraction-static 0.8 --tp=6  --quantization w8a8_int8 --max-total-tokens 65536 --prompt-filename prompt_llama4.json

# Notes: to get max throughput, "--batch-size " needs to be further tuned.
```

#### 2.4.2 Multimodel vision use case
**Server**

launch SGlang server engine:
```sh
# TP = 6, 43 OpenMP threads of rank0 are bound on 0-42 CPU cores, and the OpenMP threads of rank1 are bound on 43-85 CPU cores, etc.
# {Quantized_model_path} is based on above Quantization stage, example: ./quant_model_dir/Llama-4-Maverick-17B-128E-Instruct-w8g-1
Command:
MOE_QUANT_ONLY=1 SGLANG_CPU_OMP_THREADS_BIND="0-42|43-85|86-127|128-170|171-213|214-255" python -m sglang.launch_server --model {Quantized_model_path} --trust-remote-code --device cpu   --tp 6 --mem-fraction-static 0.8 --max-total-tokens 65536   --chat-template llama-4 --quantization w8a8_int8
```

**Client**

Refer to [this SGlang doc](https://docs.sglang.ai/backend/openai_api_vision.html#Multiple-Image-Inputs) to use OpenAI API to launch the client request.

Example python script:
```
from openai import OpenAI
client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="None")
response = client.chat.completions.create(
    model="Llama-4-Maverick-17B-128E-Instruct-w8g-1",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png",
                    },
                },
                {
                    "type": "text",
                    "text": "I have two very different images. They are not related at all. "
                    "Please describe the first image in one sentence, and then describe the second image in another sentence.",
                },
            ],
        }
    ],
    temperature=0.6,
    top_p=0.9,
    max_tokens=128,
)
print(response.choices[0].message.content)
```

