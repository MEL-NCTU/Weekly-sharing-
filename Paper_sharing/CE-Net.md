CE-Net : Context Encoder Network for 2D  Medical Image Segmentation
===
[TOC]

## Introduction
**Problem of Unet based approch segmentation:**
- **Pooling** and **strided convolution** lead to loss of spatial information

**Solution :** Context Encoder Network (**CE-Net**)

- Dense Atrous Convolution block (**DAC**)
- Residual Multi-Kernal Pooling block (**RMP**)

    



## Methods
### Model
>
![](https://github.com/MEL-NCTU/Weekly-sharing-/blob/master/Paper_sharing/Images/CE-Net_model.PNG?raw=true)

- **Encoder-Decoder Concept**

    - Encoder : extract high level features from pictures
    -  Decoder : generate result from extracted features
    

>My Thoughts :
>Encoder stage is like the network trying to understand a picture, it produce some high level feature during the process. These features are no longer pictures or something that can be visulized, they are generalized rules, such as location, relative position information or context information. After understanding a picture, the network use these rules to generate the result in the decoder stage.
>

- **General Ideas of How This Model Works**

    - Pretrained ResNet blocks prevent gradient vanishing
    - Encoder : extract more high level feature with **DAC** block
    - Decoder : preserve more spatial infomation with **RMP** block

### Dense Atrous Convolution (DAC)

<p align="center">
<img src="https://github.com/MEL-NCTU/Weekly-sharing-/blob/master/Paper_sharing/Images/CE-Net_DAC.PNG?raw=true" alt="drawing" width="500"/>
</p>

* Similar to Inception Model
* Using atrous convolutions with different **atrous rate**
* Capture features from different scale


> Atrous Convolution (Dilated Convolution)
> - Larger **receptive field** with same computation and memory cost
> - Better at extracting high-level features
> .
<p align="center">
<img src="https://github.com/MEL-NCTU/Weekly-sharing-/blob/master/Paper_sharing/Images/CE-Net_atrous.PNG?raw=true" alt="drawing" width="400"/>

</p>
     
### Residual Multi-kernal Pooling (RMP)

<p align="center">
 <img src="https://github.com/MEL-NCTU/Weekly-sharing-/blob/master/Paper_sharing/Images/CE-Net_RMP.PNG?raw=true" alt="drawing" width="500"/>
</p>

* Use different-size pooling kernal to the widen the field-of-views
* Use 1x1 Convolution to reduce computational cost (squeeze channels together)
### Dice Coefficient Loss Function

* Better for small objects, like blood vessel 
* Better than cross entropy in medical image segmentation
---

## Result
### 5 Experiment
* Optic disk
* Retinal vessel detection
* Lung Segmentation
* Cell Contour
* Retinal OCT
### Retinal Vessel Detection, Lung Segmentation, Cell Contour
* Single class segmentation


![](https://github.com/MEL-NCTU/Weekly-sharing-/blob/master/Paper_sharing/Images/CE-Net_result_pic.PNG?raw=true)



![](https://github.com/MEL-NCTU/Weekly-sharing-/blob/master/Paper_sharing/Images/CE-Net_3_acc.png?raw=true)
- Very High Sensitivity 

### Retinal OCT

![](https://github.com/MEL-NCTU/Weekly-sharing-/blob/master/Paper_sharing/Images/CE-Net_OCT.png?raw=true)

* Performs good in multi-class segmentation as well

## Conclusion
![](https://github.com/MEL-NCTU/Weekly-sharing-/blob/master/Paper_sharing/Images/CE-Net_conclusion.png?raw=true)

* ResNet backbone is good good
* Multi-size atrous conv is good good, and it is better than multi-size normal conv
* Multi-size pooling is good good
* CE-Net (DAC+RMP) is even better
## Some ~~Useless~~ Technical Details 
* ResNet34 Backbone Pretrained on ImageNet
* SGD Optimizer
* Pytorch
* GeForece Titain, Ubuntu 16.04

## Reference
- Inception model : 
https://ithelp.ithome.com.tw/m/articles/10205210
- Why 1x1 Convolution :
https://zhuanlan.zhihu.com/p/30182988
- WTF is high-level feature?
https://www.zhihu.com/question/264702008

**Jimmy Li** @NCTU

###### tags: `Segmentation` `2D Medical Image` `CE-net` `atrous convolution`
