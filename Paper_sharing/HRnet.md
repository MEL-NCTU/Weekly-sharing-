# Deep High-Resolution Representation Learning for Visual Recognition

## Introduction:

Most recently-developed classiﬁcation networks connect the convolutions from high resolution to low resolution in series, and lead to a low-resolution representation, which is further processed for classiﬁcation. However, high-resolution representations are needed for position-sensitive tasks e.g., semantic segmentation and object detection.  
  

The research team  present a novel architecture, namely High-Resolution Net (HRNet), which is able to maintain high-resolution representations through the whole process. Their approach connects high-to-low resolution convolution streams in parallel rather than in series. Thus, the approach is able to maintain the high resolution instead of recovering high resolution from low resolution, and accordingly the learned representation is potentially spatially more precise.

## Method:

Network connects high-to-low convolution streams in parallel. It maintains high-resolution representations through the whole process, and generates reliable
high-resolution representations through repeatedly fusing the representations from multi-resolution streams.

**Parallel Multi-Resolution Convolutions**
<p align="center">
  <img src="./figure/HRnet1.PNG"><br>
</p>

**Repeated Multi-Resolution Fusions**
The goal of the fusion module is to exchange the information across multi-resolution representations. 

<p align="center">
  <img src="./figure/HRnet2.PNG"><br>
</p>

<p align="center">
  <img src="./figure/HRnet3.PNG"><br>
</p>

## Results:

<p align="center">
  <img src="./figure/HRnet4.PNG"><br>
</p>
Semantic segmentation results on Cityscapes val(single scale and no ﬂipping). The GFLOPs is calculated on the input size 1024 × 2048.

<p align="center">
  <img src="./figure/HRnet5.PNG"><br>
</p>
Semantic segmentation results on PASCAL-Context. The methods are evaluated on 59 classes and 60 classes.

<p align="center">
  <img src="./figure/HRnet6.PNG"><br>
</p>
Semantic segmentation results on LIP. 

## Discussion

This method would definitely help to improve the performance, yet as it mention they use 4 GPUs to train. 
The computation and memory consuming is a big problem to solve. 


