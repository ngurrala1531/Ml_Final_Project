# SCNet
The official PyTorch implementation of CVPR 2020 paper ["Improving Convolutional Networks with Self-Calibrated Convolutions"](http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf)
## Introduction
We present a novel self-calibrated convolution that explicitly expands the fields-of-view of each convolutional layer via internal communications, thereby enriching the output features. In particular, unlike standard convolutions that combine spatial and channel-wise information,small kernels (e.g., 3 3), our self-calibrated convolution a novel method for building long-term spatial and inter-channel dependencies around each spatial location.
### Requirement
PyTorch>=0.4.1
```
## reference for SCNet architecture 
git clone https://github.com/backseason/SCNet.git

from scnet import scnet50
model = scnet50(pretrained=True)


