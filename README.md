# petals-to-the-metal
A rotationally symmetric/equivariant ConvNet applied to a real world dataset.
## Prologue
**Symmetry** is a term adored by physicists, meaning that one does something to a system while the result stays apparently the same. In the machine learning community, a more precise term, **equivariance**, is used, which means when one does something to a system, the result changes *accordingly*. From now on, I will follow the machine learning jargons.

Convolutional Neural Networks (CNN) are by construction translationally equivariant. CNNs use filters that are spacially small compared to a whole image and slide filters over an image to identify local features. No matter where an eye appears in an image, it is eventually picked up by filters through scanning.

On the other hand, rotational equivariance is not built into CNN's mathematical framework. The consequence is, for example, a lower accuracy in identifying upside down cats. To address the issue caused by object orientations, images are rotated by random angles before they are fed to train the neural nets. This is a kind of *data augmentation*.

It is aesthetically enticing to construct CNNs with rotational equivariance built into their mathematical framework. Rotational equivariance imposes constraints on the size of the function space of filters. This potentially can improve the performance or parameter efficiency of neural networks, just like weight sharing greatly reduced the model complexity and took part in the breakthrough of superhuman performance.

A large amount of papers have been published about the construction of a rotationally equivariant CNN. The variant I am testing out in this repository is [e2cnn](https://github.com/QUVA-Lab/e2cnn).
## A rotationally equivariant CNN, E(2)-steerable CNN
[Amsterdam Machine Learning Lab](https://amlab.science.uva.nl/), a group led by [Prof. Max Welling](https://staff.fnwi.uva.nl/m.welling/), has been applying group theory to the mathematical framework of neural networks for years with impactful publications.

It is anecdotally interesting to note that Prof. Welling got his PhD degree under the instruction of Prof. Gerard 't Hooft, a renowned theoretical physicist who proved the renormalizability of gauge theories in his PhD work and received the Nobel Prize in Physics with it in 1999.

## Data used for this investigation
The data used for this investigation comes from one of the Kaggle competitions called [Petals to the Metal - Flower Classification on TPU](https://www.kaggle.com/c/tpu-getting-started). This dataset is selected simply because the competition has no end date. Therefore, I can be sure that the dataset will be available for a long time. Besides, due to the radial symmetry of flowers, this might exhibit some interesting effects on the results.
### Data preprocessing
The data is provided by Kaggle in the TFRecord format readily used by TensorFlow. However, since I am using PyTorch for this study, I have to convert the TFRecord files into the input data format PyTorch uses. To achieve this, I have written a simple script which can be found in [this repository](https://github.com/kaikai581/tfrecord-io-test).

The original TFRecord files can be downloaded from [this link](https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/21154/1243559/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1614893308&Signature=Q%2FHtFG2qA6ITd1mFpWFnCFbJrpyKWCzcgqLpfGk3AK063ZgPNtbRbIe6yB6JG8g%2BN%2FDLQ%2BzgoTfg81%2BpZkXDHjNp7d41EuXwY1dMwnBNnPSBD26zJhbaV%2Fr%2FDCUiLHhBKM%2BhXZwi1cV35TJM1L4KmQi77gTCUBKV86nU6k%2B5AaqZ6eb5bQtk95dPORleoYUD3p4KPtE3gcG93ij2rTDWA8cCC%2B39jFgz4XLEoFy34%2FpCy9KIVnp1waDSCSULylIxnYki4OktGLEGsOueTxUR3ruaTKsnrS17T%2F1Au4pvm%2FUzctx1B6jIoHoyMl1sc37nHGsQq9R5%2FbpQkf9S%2BYDzTg%3D%3D&response-content-disposition=attachment%3B+filename%3Dtpu-getting-started.zip). After data conversion, one should see a directory tree structure like this.
```
$ tree imagefolder-jpeg-224x224 -L 1
imagefolder-jpeg-224x224
├── test
├── train
└── val
```
For convenience, the converted images can be downloaded from [here](https://www.kaggle.com/shihkailin/imagefolderjpeg224x224/download).

## Install e2cnn in an Anaconda environment with GPU acceleration
[Anaconda](https://www.anaconda.com/) has been my choice of python distribution for all projects. Follow the [instruction](https://docs.anaconda.com/anaconda/install/) to install Anaconda on your system, and make sure that the `conda` command is available to your command line.

To install e2cnn to your Anaconda environment, first create an environment.
```
$ conda create -n <env_name>
$ conda activate <env_name>
```
Now, install PyTorch that is compatible to your GPU driver installed. To make sure the compatibility, use the `nvidia-smi` command. For example, on my system,
```
$ nvidia-smi
Thu Mar  4 15:34:31 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.87.00    Driver Version: 418.87.00    CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 108...  Off  | 00000000:65:00.0 Off |                  N/A |
| 25%   27C    P8    11W / 250W |      0MiB / 11178MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
In the top right corner, the CUDA Version listed is 10.1. Since CUDA toolkit is backward compatible, all PyTorch versions earlier than 10.1 would work. Therefore, I would install cudatoolkit 10.1 as an example. Look up the correct command on the [PyTorch front page](https://pytorch.org/).
```
$ conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```
Finally, it's time to install e2cnn into the current environment.
```
$ pip install e2cnn
```

## Models used in this study
Since e2cnn is more like a mathematical framework, the author has provided several predefined variants for easy use. I have tried some of them, and found the variant, `wrn16_8_stl_d8d4d1`, to be very performant.

A brief explanation on the model name. "wrn" is "wide ResNet", suggesting the network structure is modeled after this CNN architecture, 16 is the depth, 8 is the so-called "widen factor", "stl" means the model is built for the STL dataset, and d8d4d1 means that the model's blocks are respectively [D₈, D₄ and D₁](https://en.wikipedia.org/wiki/Dihedral_group) equivariant.

Find [here](https://github.com/kaikai581/petals-to-the-metal/blob/master/train_models/wrn16_8_stl_d8d4d1_weight_decay/train_model.py) for the script I use to train the model. The trained weights used in this study can be downloaded [here](https://drive.google.com/file/d/1lwMc9DizSwmD0TEXbuqIY2xUBGCqIeVM/view?usp=sharing).
