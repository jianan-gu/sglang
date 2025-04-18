## Prepare the environment to run SGLang on CPU

**GCC >= 11 is needed**

```sh
# Create conda ENV
conda create -n sglang python=3.10
conda activate sglang

# Install SGLang
git clone https://github.com/jianan-gu/sglang -b jianan/llama4_v0
cd sglang
pip install -e "python[all_cpu]"

conda install -y libsqlite=3.48.0
pip uninstall -y triton
pip install triton==3.1

conda install -y gperftools -c conda-forge
pip install intel-openmp==2024.2.0

cd ..

# Build and install vllm for CPU following https://docs.vllm.ai/en/latest/getting_started/installation/cpu/index.html
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.6.4.post1

sudo apt-get install libnuma-dev
# TODO: check the case where this conda install does not work
# conda install -y libnuma numactl
pip install cmake==3.31.2 wheel packaging ninja "setuptools-scm>=8" numpy nvidia-ml-py
pip install -v -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu

VLLM_TARGET_DEVICE=cpu python setup.py develop

# Build sgl-kernel
# PT 2.6 is needed to build sgl-kernel
pip uninstall torch torchvision
pip3 install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu

cd ../sglang/
cd sgl-kernel
python setup.py install

cd ..
```

## Example command lines
**The following command lines are for demonstration purposes. Modify the arguments and thread binding according to your requirements and CPU type.**

**Please avoid cross NUMA node memory access when setting SGLANG_CPU_OMP_THREADS_BIND.**

`SGLANG_CPU_OMP_THREADS_BIND` specifies the CPU cores dedicated to the OpenMP threads. `--tp` sets the TP size. Below are the example of running without TP and with TP = 6. By changing `--tp` and `SGLANG_CPU_OMP_THREADS_BIND` accordingly, you could set TP size to other values and specifiy the core binding for each rank.

**Preload iomp and tcmalloc for better performance.**
```sh
export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libtcmalloc.so
```
### Quantization
```sh
cd auto-round/
pip install -e .[cpu]
# modify model_dir and output_dir in "run_llama4_quant.sh"
sh run_llama4_quant.sh
```
### Benchmark
```sh
# TP = 6, 43 OpenMP threads of rank0 are bound on 0-42 CPU cores, and the OpenMP threads of rank1 are bound on 43-85 CPU cores, etc.
SGLANG_CPU_OMP_THREADS_BIND="0-42|43-85|86-127|128-170|171-213|214-255" python -m sglang.bench_one_batch --batch-size 1 --input 1024 --output 1024 --model {PATH} --trust-remote-code --device cpu --mem-fraction-static 0.8 --tp=6  --quantization w8a8_int8 --max-total-tokens 65536 
```


