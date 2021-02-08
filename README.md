# petals-to-the-metal
A rotationally symmetric/equivariant ConvNet applied to a real world dataset.
## Prologue
**Symmetry** is a term adored by physicists, meaning that one does something to a system while the system stays apparently the same. In the machine learning community, a more precise term, **equivariance**, is used, which means when one does something to a system, the result changes *accordingly*. From now on, I will follow the machine learning jargons.

Convolutional Neural Networks (CNN) are by construction translationally equivariant. CNNs use filters that are spacially small compared to a whole image and slide filters over an image to identify local features. No matter where an eye appears in an image, it is eventually picked up by filters through scanning.

On the other hand, rotational equivariance is not built into CNN's mathematical framework. The consequence is, for example, a lower accuracy in identifying upside down cats. To address the issue caused by object orientations, images are rotated by random angles before they are fed to train the neural nets. This is a kind of *data augmentation*.
## A rotationally equivariant CNN, E(2)-steerable CNN
[Amsterdam Machine Learning Lab](https://amlab.science.uva.nl/), a group led by [Prof. Max Welling](https://staff.fnwi.uva.nl/m.welling/), has been applying group theory to the mathematical framework of neural networks for years with impactful publications.
