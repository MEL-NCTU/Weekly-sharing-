# CE-Net: Context Encoder Network for 2D Medical Image Segmentation

## 1. Introduction
Base on U-Net, the consecutive pooling and strided convolutional operations lead to the loss of some spatial information. In this paper, we propose a context encoder network (referred to as CE-Net) to capture more high-level information and preserve spatial information for 2D medical image segmentation.

#### CE-Net contain :
* Feature encoder module
* Context extractor module
* Feature decoder module

#### The main contribution of this work :
* We propose a DAC block and RMP block to capture more high-level features and preserve more spatial information.
* We apply the proposed method in different tasks including optic disc segmentation, retinal vessel detection, lung segmentation, cell contour segmentation and retinal OCT layer segmentation. Results show that the proposed method outperforms the state-of-the-art methods in these different tasks.


## 2. Architecture
![](https://i.imgur.com/2OzGlWd.png)
### Context extractor module
* Dense Atrous Convolution Block (DAC Block)
DAC has four cascade branches with the gradual increment of the number of atrous convolution, from 1 to 1, 3, and 5, then the receptive ﬁeld of each branch will be 3, 7, 9, 19. Finally, we directly add the original features with other features, like shortcut mechanism in ResNet.
![](https://i.imgur.com/mo49cQb.png)

* Residual Multi-kernel Pooling Block (RMP Block)
The proposed RMP encodes global context information with four different-size receptive ﬁelds: 2×2, 3×3, 5×5 and 6×6. The four-level outputs contain the
feature maps with various sizes. To reduce the dimension of weights and computational cost, we use a 1×1 convolution after each level of pooling.  Then we upsample the low-dimension feature map to get the same size features as the original feature map via bilinear interpolation.
![](https://i.imgur.com/PruUqur.png)

## 3. Result
### A. Optic Disc Segmentation (ORIGA, Messidor and RIM-ONE-R1)
Crop 800×800
* ORIGA dataset contains 650 images with dimension 3072 × 2048.
* Messidor dataset consists of 1200 images with three different sizes: 1440 × 960, 2240 × 1488, 2340 × 1536.
* Fve different expert annotations in RIM-ONE-R1 dataset.

![](https://i.imgur.com/bG6AnOf.png)

### B. Retinal Vessel Detection (DRIVE)
![](https://i.imgur.com/9LmiqFb.png)

### C. Lung Segmentation (LUNA competition)
![](https://i.imgur.com/pbDbZfH.png)

### D. Cell contour Segmentation
![](https://i.imgur.com/7TgCvHj.png)

## 4. Visualization
![](https://i.imgur.com/eZ1mvuI.png)
![](https://i.imgur.com/C8maxKn.png)
