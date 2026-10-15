#!/bin/bash -l 
#SBATCH --job-name=arrayfire   # Name of job 
#SBATCH --account=def-taolu   # adjust this to match the accounting group you are using 


#SBATCH --time=0-00:20          # 2 hours 
#SBATCH --cpus-per-task=6         # CPU cores/threads 
#SBATCH --gpus-per-node=a100:1 
#SBATCH --mem=16G 
 
 
 
# Load modules 
module load StdEnv/2020 gcc/9.3.0 cuda/12.2 fmt/9.1.0 spdlog/1.9.2 arrayfire/3.9.0 rust/1.70.0 python/3.11.2   

 
 
nvidia-smi 
 
 
 
rm -rf /home/yilind/scratch/raybnn/arrayfire/
mkdir /home/yilind/scratch/raybnn/arrayfire/
 
 
export AF_JIT=12 
 
export AF_JIT_KERNEL_CACHE_DIRECTORY=/home/yilin/Downloads/arrayfire/
 
export AF_OPENCL_MAX_JIT_LEN=$AF_JIT 
 
export AF_CUDA_MAX_JIT_LEN=$AF_JIT 
 
export AF_CPU_MAX_JIT_LEN=$AF_JIT 
 
export AF_DISABLE_GRAPHICS=1 
 
export RUSTFLAGS="-Awarnings -C target-cpu=native" 
 
 
 
cd /home/yilind/scratch/raybnn/
cargo clean 
 
 
#cargo run --example figure1b --release 
#cargo run --example figure2a --release 
#cargo run --example figure2b --release 
#cargo run --example figure2c --release 
#cargo run --example figure2d --release 
#cargo run --example figure2e --release 
#cargo run --example figure2f --release 
#cargo run --example figure3a --release 
#cargo run --example figure3b --release 
#cargo run --example figure3d --release 
#cargo run --example figure4_raybnn_sp --release
#cargo run --example figure4_raybnn --release
#cargo run --example figure4_raybnn_or --release
#cargo run --example SNN_test1 --release
cargo run --example SNN_test4 --release
cargo run --example SNN_test5    --release  ./test_data  ./  42

 
#for i in {0..56}       
#do 
#    cargo run --example figure6_raybnn2   --release  $i 
#done

cargo clean 
#cd /scratch/brosnany/ 

 
 

