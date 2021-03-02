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

## Install e2cnn in an Anaconda environment
Anaconda has been my choice of python distribution for all projects.
