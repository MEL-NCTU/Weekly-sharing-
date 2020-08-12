# FastFCN:Rethinking Dilated Convolution in the Backbone for Semantic Segmentation                  
      

## Introduction
Modern approaches for semantic segmentation usually employ dilated convolutions in the backbone to extract high resolution feature maps,which brings:       

* heavy computation
* complexity
* memory footprint


<img src =https://github.com/leochang123/github-test/blob/master/imag1.png widtg =500>

In order to tackle the aforementioned issue caused by dilated convolutions,we propose a novel joint upsampling module to replace the time and memory consuming dilated convolutions,namely Joint Pyramid Upsampling(JPU)

<img src = https://github.com/leochang123/github-test/blob/master/image2.png width =600>

## Method
**Joint Upsampling**:  Given a low-resolution target image and high-resolution guidance image,joint upsampling aims at generating a high-resolution target image by transferring detail and structures from the guidance image


<img src =https://github.com/leochang123/github-test/blob/master/image3.png width=375>

**Dilated Convolution**
* split the input feature into two groups(**S**)
* process each feature with the same convolution layer(**Cr**)
* merge the two generated features(**M**)

**Stride Convolution**
* process the input feature with regular convolution(**Cr**)
* remove the elements with an odd index(**R**)

<img src = https://github.com/leochang123/github-test/blob/master/image4.png width=700>


**Reformulating into Joint Upsampling**

Formally, given the input feature map x,the output feature map yd in DilatedFCN is obtained as follows:

<img src = https://github.com/leochang123/github-test/blob/master/image5.png width=400>

while in our method,the output feature map ys is generated as follows:

<img src = https://github.com/leochang123/github-test/blob/master/image6.png width=400>

The feature map y that approximates yd can be obtained as follows:

<img src =https://github.com/leochang123/github-test/blob/master/image7.png width=400>

**Structrue**

<img src = https://github.com/leochang123/github-test/blob/master/image8.png width=500>

## Result

<img src = https://github.com/leochang123/github-test/blob/master/Image9.png width=700>
<img src = https://github.com/leochang123/github-test/blob/master/image10.png width=700>
<img src = https://github.com/leochang123/github-test/blob/master/image11.png width=700>
<img src = https://github.com/leochang123/github-test/blob/master/image12.png width=700>

## Conclusion
By plugging JPU,several modern approaches for semantic segmentation achieve a better performance while runs much faster than before.
