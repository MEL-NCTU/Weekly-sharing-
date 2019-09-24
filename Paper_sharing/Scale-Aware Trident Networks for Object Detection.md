# Scale-Aware Trident Networks for Object Detection

## Introduction
* #### We present our investigation results about the effect of the receptive ﬁeld in scale variation. To our best knowledge, we are the ﬁrst to design controlled experiments to explore the receptive ﬁeld on the object detection task.
* #### We propose a novel Trident Network to deal with scale variation problem for object detection. Through multi-branch structure and scale-aware training, TridentNet could generate scale-speciﬁc feature maps with a uniform representational power.
* #### We propose a fast approximation, TridentNet Fast, with only one major branch via our weight-sharing trident-block design, thus introducing no additional parameters and computational cost during inference.
* #### We validate the effectiveness of our approach on the standard COCO benchmark with thorough ablation studies. Compared with the state-of-the-art methods, our proposed method achieves an mAP of 48.4 using a single model with ResNet-101 backbone.

## Method
* ### Dilated convolution
![](https://i.imgur.com/3nj2o07.jpg)
####
#### Use ResNet-50 and ResNet-101 as the backbone networks and vary the dilation rate d s of the 3×3 convolutions from 1 to 3 for the residual blocks in the conv4 stage.
![](https://i.imgur.com/ySk9hDs.jpg)
* ### Trident block
![](https://i.imgur.com/7Ucvwbf.jpg)      ![](https://i.imgur.com/rCmm0EN.jpg)






## Result
![](https://i.imgur.com/qDFcVOT.jpg)
![](https://i.imgur.com/FUDaXZE.jpg) ![](https://i.imgur.com/Zv0CHax.jpg)
![](https://i.imgur.com/F1wr2o0.jpg)
![](https://i.imgur.com/2XZqQzX.jpg)


## Conclution
* #### we present a simple object detection method called Trident Network to build in-network scale- speciﬁc feature maps with the uniform representational power. 
* #### A scale-aware training scheme is adopted for our multi-branch architecture to equip each branch with the specialized ability for corresponding scales. 
* #### The fast inferencemethod with the major branch makes TridentNet achieve signiﬁcant improvements over baseline methods without any extra parameters and computations.


