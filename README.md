# TeamMates and Contributions
The base code had some minor issues which were fixed individually by all the team members. We have implemented 3 projecs using SCNet50 each one majorly contributing to one of the three projects. 
Greeshma Guduguntla (B01037966): Implemented Object Detection using SCNet50, helped in implenting Classification task. 
Naga Ramya Gurrala (B01040125): Implemented Classification using SCNet50, Helped in implementing Object Detection task. 
Lakshmi Reshma Reddy Pallala (B01035894): Implemented Face Mask Detection using SCNet50. 

Equal contribution by all the team members in Project documentation.


# SCNet
The official PyTorch implementation of CVPR 2020 paper ["Improving Convolutional Networks with Self-Calibrated Convolutions"](http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf)

## Projects
By referening the base model of SCNet we have implemented 3 of its applications
1. Classification:
  DataSet: CIFAR10;
  Model: SCNet50
2. Object Detection:
  DataSet: Dataset of 6K images with 7 categories;
  link: https://drive.google.com/file/d/1kJSAEH2SGALlzTI0O9qmJPar64G9fInU/view
  Model: SCNet50
3. FaceMask Detection
  DataSet: Dataset of 4K images;
  Model: SCNet50; 
  link: https://drive.google.com/drive/folders/1BVDuW3bH6hsVrRxW3UX3NtNJgYc6MkW0


<div align="center">
  <img src="https://github.com/backseason/SCNet/blob/master/figures/SC-Conv.png">
</div>
<p align="center">
  Figure 1: Diagram of self-calibrated convolution.
</p>

## Usage
### Requirement
PyTorch>=0.4.1
### Examples 
```
git clone https://github.com/ngurrala1531/Ml_Final_Project.git



