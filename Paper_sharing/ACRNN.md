# Attention based Convolutional Recurrent Neural Network for Environmental Sound Classification (**ACRNN**)

## ESC(**E**nviromental **S**ound **C**lassification)
1. MFCC+GMM(gaussian mixture model)
2. MFCC+RNN
3. log mel into RGB + CNN
4. Parallel RNN & CNN
5. attention mechnism-based RNN
6. **ARCNN**
## Main Contributions of this paper:
#### 1. Dealing with silent frames and semantically irrelevant frames:
  - Attention model to focus on discriminative features
#### 2. Analyzing temporal relations:
  - **Novel convolutional RNN model**(CNN to extract high level features and then input the features to bidirectional GRUs)

## Difference between LSTM & GRU
![](https://i.imgur.com/lrfyQsF.png)

## Architecture
![](https://i.imgur.com/iN0l0TE.jpg)
ACRNN, which combines convolutional RNN and a frame-level attention mechanism.
- 8 convolutional layers
- 2 bidirecrional GRU layers
(Dropout with probability of 0.5 is used for each GRU layer to avoid overfitting)
![](https://i.imgur.com/cEKrhU2.jpg)

Attention for CNN & RNN layers

----
## Feature Extraction
1. STFT hamming window size: 23ms & 50% overlap
2. 128-band Gammatone filter bank

![](https://i.imgur.com/jGqbUnp.png)

3. Spectrogram splited into 128 frames (approximately 1.5s in length) with 50% overlap
4. concatenate the log gammatone spectrogram and its delta information to a 3-D feature representation  
`X ∈ R^128×128×2^(Log-GTs)` as the input of the network.
## Visualization
![](https://i.imgur.com/C2Em4mT.jpg)

## Data Augmentation
- Time stretch:  a factor randomly selected from [0.8, 1.3] 
- Pitch shift: factor randomly selected from [-3.5, 3.5]

## Results
#### - Between models
![](https://i.imgur.com/PBaTJTr.jpg)

#### - Confusion matrix
![](https://i.imgur.com/7u1zXyW.jpg)

#### - Model settings
![](https://i.imgur.com/ZTCUuHF.jpg)

#### - Attention apply
![](https://i.imgur.com/nQEMlaC.jpg)

----

> Reviewed by Andrew Yang 2019/10/02


# Layers
> l1-l2: The first two stacked convolutional layers use 32 filters with a receptive field of (3,5) and stride of (1,1). This is followed by a max-pooling with a (4,3) stride to reduce the dimensions of feature maps. ReLU activation function is used.  

> l3-l4: The next two convolutional layers use 64 filters with a receptive field of (3,1) and stride of (1,1), and is used to learn local patterns along the frequency dimension. This is followed by a max-pooling with a (4,1) stride. ReLU activation function is used.


> l5-l6: The following pair of convolutional layers uses 128 filters with a receptive field of (1,5) and stride of (1,1), and is used to learn local patterns along the time dimension. This is followed by a max-pooling with a (1,3) stride. ReLU activation function is used.  

> l7-l8: The subsequent two convolutional layers use 256 filters with a receptive field of (3,3) and stride of (1,1) to learn joint time-frequency characteristics. This is followed by a max-pooling of a (2,2) stride. ReLU activation function is used.


> l9-l10: Two bidirectional GRU layers with 256 cells are used for temporal summarization, and tanh activation function is used. 

> Dropout with probability of 0.5 is used for each GRU layer to avoid overfitting. Batch normalization [10] is applied to the output of the convolutional layers to speed up training. L2-regularization is applied to the weights of each layer with a coefficient 0.0001.

