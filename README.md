# SCNet
The official PyTorch implementation of CVPR 2020 paper ["Improving Convolutional Networks with Self-Calibrated Convolutions"](http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf)

## Introduction
By referening the base model of SCNet we have implemented 3 of its applications
1. Classification:
  DataSet: CIFAR10
  Model: SCNet50
2. Object Detection:
  DataSet: Dataset of 6K images with 7 categories
  Model: SCNet50
3. FaceMask Detection
  DataSet: Dataset of 4K images
  Model: SCNet50

Unable to upload datasets as its size is more that the git limits 


<div align="center">
  <img src="https://github.com/backseason/SCNet/blob/master/figures/SC-Conv.png">
</div>
<p align="center">
  Figure 1: Diagram of self-calibrated convolution.
</p>

## Useage
### Requirement
PyTorch>=0.4.1
### Examples 
```
git clone https://github.com/backseason/SCNet.git

from scnet import scnet50
model = scnet50(pretrained=True)

