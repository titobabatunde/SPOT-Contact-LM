SPOT-Contact-Single: *Improving Single-Sequence-Based Prediction of Protein Contact Map using a Transformer Language Model, Large Training Set and Ensembled Deep Learning.*
====
The standalone version / CLI-tool of SPOT-Contact-Single available for public use for research purposes.

Contents
----
  * [Introduction](#introduction)
  * [Results](#results)
  * [System Requirments](#system-requirments)

# SPOT-Contact-Single

Introduction
----

Results
----

System Requirments
----

**Hardware Requirments:**
SPOT-Contact-single predictor has been tested on standard ubuntu 18 computer with approximately 32 GB RAM to support the in-memory operations.

* [Python3.7](https://docs.python-guide.org/starting/install3/linux/)
* [Anaconda](https://anaconda.org/anaconda/virtualenv)
* [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive) (Optional if using GPU)
* [cuDNN (>= 7.4.1)](https://developer.nvidia.com/cudnn) (Optional if using GPU)

Installation
----

To install SPOT-Contact-Single and it's dependencies following commands can be used in terminal:

1. `git clone https://github.com/jas-preet/SPOT-Contact-Single.git`
2. `cd SPOT-Contact-Single`

To download the model check points from the dropbox use the following commands in the terminal:

3. 
4. `tar -xvf contact_jits.tar.xz`

To install the dependencies and create a conda environment use the following commands

5. `conda create -n spot_contact_sgl python=3.7`
6. `conda activate spot_contact_sgl`

if GPU computer:
7. `conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch`

for CPU only 
7. `conda install pytorch==1.6.0 torchvision==0.7.0 cpuonly -c pytorch`

8. `pip install fair-esm`

9. `conda install pandas=1.1.1`

10. `conda install tqdm`

Input dependency
----
SPOT-Contact-Single uses the output of **SPOT-1D-Single**. Please generate the prediction of SPOT-1D-Single from the Stand-alone version available at `https://github.com/jas-preet/SPOT-1D-Single`.
Copy the prediction for the required proteins to `data_files/out_spot_1d_sgl/`

Execute
----
To run SPOT-Contact-Single use the following command

`python -W ignore spot_contact_single.py --file_list file_lists/file_list_casp14.txt --save_path results/ --device cuda:0 --esm_device cuda:0`

or 

`python -W ignore spot_contact_single.py --file_list file_lists/file_list_casp14.txt --save_path results/ --device cpu --esm_device cpu` 
