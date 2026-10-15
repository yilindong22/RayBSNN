# RayBNN + SNN
A 3-D Biological Neural Network model with forward propagation and backward propagation from Spiking Neural network



# System Requirements

* RTX 3090 or more powerful with at least 24GB VRAM
* 32GB RAM
* 20 GB of disk space
* Docker (https://www.docker.com/)
* Rust (https://www.rust-lang.org/)
* Arrayfire (https://github.com/arrayfire/arrayfire)
* Arrayfire Rust  (https://github.com/arrayfire/arrayfire-rust)
* Pytorch (https://pytorch.org/)
* Pytorch geometric (https://github.com/pyg-team/pytorch_geometric)
* Matlab



# Different Functions from Original RayBNN 

Tracing Function 
`RayBNN/src/neural/network_snn_f32.rs`
- `find_path_backward_snn()` : Walks through all neurons generated to discover which neurons still have a path to any output neuron (directly or un directly), Removed          the parameters of UAF from RayBNN's `raybnn::graph::path_f32::find_path_backward_group2`.  

Neural Network Training Algorithms

`RayBNN/src/neural/network_snn_f32.rs`
- `state_space_forward_batch_snn()`: Forward pass using CSR weighted adjacency sparse matrices and LIF. (Hard reset, the membrane potential will be set to the reset voltage after firing). This works better compare the soft reset. 
- `state_space_forward_batch_snn_soft()` Forward pass using CSR weighted adjacency sparse matrices and LIF. (Soft reset, the membrane potential will decrease the threshold potential after firing)
- `state_space_backward_batch()`: Backward pass using CSR weighted adjacency sparse matrices . Generates the gradients of the sparse weighted adjacency matrix
- `state_space_backward_batch2()` : same backward logrithm, but having an extra parameter (ce_spike_rate_global has different input shape compare to mse_spike_rate_grad)
Loss Functions 
`RayBNN/src/optimal/snnloss_f32.rs`
- `mse_spike_rate()` : MSE loss fucntion based on spikes 
- `mse_spike_rate_grad()` : Gradient of the MSE loss fucntion 
- `ce_spike_rate_global()`: Gradient of cross entropy loss fucntion 

Dataload Function:
- `file_to_hash_cpu_batch_blocks()`: Data load function which reads times series dataset which length > 1

# Installation Guide (Same tools required for RayBNN)

## 1. On the Host Machine. Place RayBNN.zip into $RAYBNN_DIR and unzip it.

Set the $RAYBNN_DIR enviromental variable to a folder that will store all the RayBNN files

For example, setting $RAYBNN_DIR to /opt/
```
export RAYBNN_DIR=/opt/
```

Place RayBNN.zip into $RAYBNN_DIR and unzip it to produce:

* $RAYBNN_DIR/RayBNN/src/
* $RAYBNN_DIR/RayBNN/examples/
* $RAYBNN_DIR/RayBNN/matlab_plot/
* $RAYBNN_DIR/RayBNN/python_verify/


## 2. Make Sure Matlab is Installed On the Host Machine. 

Make sure Matlab is installed on the host system. Tested on Matlab 2023a


## 3. On the Host Machine. Download CUDA Docker Container from Nvidia

This will download a CUDA Docker Container and link $RAYBNN_DIR directory in the Host Machine to the /workspace/ directory in the Docker Container.

```
docker run --name raybnn \
--gpus all \
-v $RAYBNN_DIR:/workspace \
-w /workspace  \
-it  nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04  bash
```


## 4. Inside the Docker Container, Verify the GPU is detected and CUDA is 12.1

Note that you need a GPU with 24 GB or more VRAM to run all of the code.
RayBNN was tested on RTX3090. Verify the GPU is detected

```
nvidia-smi
```


## 5. Inside the Docker Container, Install Dependencies and Install RayBNN

`./install.sh` installs all of the dependencies inside the docker container.


```
cd /workspace/RayBNN

chmod 700 ./install.sh

bash ./install.sh

exit
```
## 6. Install the Docker Container to run other models


Download pytorch docker container
Tested on RTX 3090 with i5-8400

```
docker run --name othermodel \
--gpus all \
-v $RAYBNN_DIR:/workspace \
-w /workspace  \
-it  pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel  bash

apt update

apt install wget git  curl git-lfs build-essential


pip install scikit-learn matplotlib pandas pytorch-lightning==1.9.0

pip install torch_geometric

pip install pyg_lib torch_scatter \
torch_sparse torch_cluster \
torch_spline_conv \
-f https://data.pyg.org/whl/torch-1.13.0+cu116.html



exit
```


## Reproducing inidividual results

On the Host Machine, Restart the Docker Container
```
docker restart raybnn

docker exec -it raybnn bash

cd /workspace/RayBNN
```



### Running RayBNN+SNN for the Mnist Dataset

Mnist dataset used :
https://www.kaggle.com/datasets/hojjatk/mnist-dataset

The dataset's features has been normalized to {0,1}, and the label has been applied one hot encoded

Related scripts at 
- `RayBNN/examples/SNN_test4.rs`


Run the SNN Test for the Mnist Dataset, and generates a file names `target_vs_prediction784.csv`
Where the csv file has shape of [nsamples,2] where column 0 = ground truth, column 1 = prediction

```
cd /workspace/RayBNN
cargo run --example SNN_test4
```


### Running RayBNN+SNN for the Gunpoint Dataset

Gunpoint dataset used :
https://www.timeseriesclassification.com/description.php?Dataset=GunPoint

The dataset's features has been normalized to {0,1}, and the label has been applied one hot encoded

Related scripts at 
- `RayBNN/examples/SNN_test6.rs`


Run the SNN Test for the Gunpoint Dataset
Where the csv file has shape of [nsamples,2] where column 0 = ground truth, column 1 = prediction

```
cd /workspace/RayBNN
cargo run --example SNN_test4
```

## Results for the EEG dataset

### The 210 GB EEG dataset

http://gigadb.org/dataset/100542

### The preprocessing steps

Preprocessing steps to obtain the processed `KU_mi_smt.h5` dataset

https://github.com/zhangks98/eeg-adapt

### Running Deep4Net_RayBNN_SNN

Running Deep4Net_RayBNN_SNN for a specific fold number 42 (this fold number is for preprocess the EEG dataset) and GPU number 1

cargo run --example SNN_test5    --release  ./test_data  ./  42
mv *.csv ./python_verify/EEG/Deep4Net_RayBNN/
cd python_verify/EEG/Deep4Net_RayBNN/
python3 ./getresult.py 
```



### Running CSP_LDA

Running CSP LDA for a specific fold number 42 and GPU number 1

Input:  /path_to/KU_mi_smt.h5         Dataset
Output:  ./CSP_info_*.txt             Containing Accuracy, F1, AUC ROC

```
cd /workspace/RayBNN/python_verify/EEG/CSP_LDA/
python3 ./train_CSP_LDA.py   /path_to/KU_mi_smt.h5   ./   -fold  42 -gpu 1
python3 ./parseData.py 
```


### Running CSP_LR

Running CSP LR for a specific fold number 42 and GPU number 1

Input:  /path_to/KU_mi_smt.h5         Dataset
Output:  ./CSP_info_*.txt             Containing Accuracy, F1, AUC ROC

```
cd /workspace/RayBNN/python_verify/EEG/CSP_LR/
python3 ./train_CSP_LR.py   /path_to/KU_mi_smt.h5   ./   -fold  42 -gpu 1
python3 ./parseData.py 
```



### Running Xdawn MDM

Running Xdawn MDM for a specific fold number 42 and GPU number 1

Input:  /path_to/KU_mi_smt.h5         Dataset
Output:  ./MDM_info_*.txt             Containing Accuracy, F1, AUC ROC

```
cd /workspace/RayBNN/python_verify/EEG/Xdawn_MDM/
python3 ./train_Xdawn_MDM.py   /path_to/KU_mi_smt.h5   ./   -fold  42 -gpu 1
python3 ./parseData.py 
```






### Running Xdawn LR

Running Xdawn LR for a specific fold number 42 and GPU number 1

Input:  /path_to/KU_mi_smt.h5         Dataset
Output:  ./LR_info_*.txt              Containing Accuracy, F1, AUC ROC

```
cd /workspace/RayBNN/python_verify/EEG/Xdawn_LR/
python3 ./train_Xdawn_LR.py   /path_to/KU_mi_smt.h5   ./   -fold  42 -gpu 1
python3 ./parseData.py 
```



### Running Deep4Net

Running Deep4Net for a specific fold number 42 and GPU number 1

Input:  /path_to/KU_mi_smt.h5         Dataset
Output:  ./CNN_info_*.txt             Containing Accuracy, F1, AUC ROC

```
cd /workspace/RayBNN/python_verify/EEG/Deep4Net/
python3 ./train_base_Deep4Net.py   /path_to/KU_mi_smt.h5   ./   -fold  42 -gpu 1
python3 ./parseData.py 
```


### Running Xdawn_Deep4Net_MLP

Running Xdawn_Deep4Net_MLP for a specific fold number 42 and GPU number 1

Input:  /path_to/KU_mi_smt.h5         Dataset
Output:  ./info_*.txt                 Containing Accuracy, F1, AUC ROC

```
cd /workspace/RayBNN/python_verify/EEG/Xdawn_Deep4Net_MLP/
python3 ./preprocess_Xdawn_Deep4Net.py   /path_to/KU_mi_smt.h5   ./   -fold  42 -gpu 1
python3 ./MLPEEG.py  ./  42 
python3 ./getresult.py 
```




### Running Deep4Net_RayBNN

Running Deep4Net_RayBNN for a specific fold number 42 and GPU number 1

Input:  /path_to/KU_mi_smt.h5         Dataset
Output:  ./info_*.txt                 Containing Accuracy, F1, AUC ROC

```
cd /workspace/RayBNN/python_verify/EEG/Deep4Net_RayBNN/
python3 ./preprocess_Deep4Net_RayBNN.py   /path_to/KU_mi_smt.h5   ./   -fold  42 -gpu 1
cargo run --example table1_deep4net_raybnn    --release  ./  ./  42
python3 ./getresult.py 
```




### Running Xdawn_Deep4Net_RayBNN

Running Xdawn_Deep4Net_RayBNN for a specific fold number 42 and GPU number 1

Input:  /path_to/KU_mi_smt.h5         Dataset
Output:  ./info_*.txt                 Containing Accuracy, F1, AUC ROC

```
cd /workspace/RayBNN/python_verify/EEG/Deep4Net_RayBNN/
python3 ./preprocess_Deep4Net_RayBNN.py   /path_to/KU_mi_smt.h5   ./   -fold  42 -gpu 1
cargo run --example table1_deep4net_raybnn    --release  ./  ./  42
cp ./* /workspace/RayBNN/python_verify/EEG/Xdawn_Deep4Net_RayBNN/



cd /workspace/RayBNN/python_verify/EEG/Xdawn_Deep4Net_RayBNN/
python3 ./preprocess_Xdawn_Deep4Net_RayBNN.py   /path_to/KU_mi_smt.h5   ./   -fold  42 -gpu 1
cargo run --example table1_transfer_xdawn_deep4net_raybnn    --release  ./  ./  42
python3 ./getresult.py 
```



