---
title: 'Project documentation template'
disqus: hackmd
---

Bag of Tricks for Image Classification with Convolutional Neural Networks
===


# Introduction
> Training procedure refinements,
> including changes in loss functions, data preprocessing,
and optimization methods also played a major role.


# 1.Efficient Training
## Largebatch training:
### Linear scaling learning rate:
> A large batch size reduces the noise in the gradient,so we may increase the learning rate to make a larger progress along the opposite of the gradient direction.
#### learning rate=0.1*b/256, as the initial is 0.1, b is batch size.

### Learning rate warmup.
#### learning rate = data epochs*learning rate / total epochs 

### Zero gamma 
> ![](https://i.imgur.com/tVBoAkX.png)
gama = 0 for all BN layers that sit at the end of a residual block.

### No bias decay
> only apply the regularization to weights to avoid overfitting.

## Low precision training
使用半精度FP16計算參數
### ![](https://i.imgur.com/S88W3zC.png)

## Experiment Results
![](https://i.imgur.com/hoagBY2.png)

# 2.Model Tweaks
![](https://i.imgur.com/6KngiCo.png)

![](https://i.imgur.com/W7iXBXm.png)

![](https://i.imgur.com/hPcrDcu.png)

# 3.Training Refinements

## Cosine Learning Rate Decay
![](https://i.imgur.com/etNoGxh.png)
![](https://i.imgur.com/lsXYu4Z.png)

## Label Smoothing
> 透過軟化 one-hot 標籤減少 overfitting 
![](https://i.imgur.com/Wfs7tL7.png)
![](https://i.imgur.com/VEJVs3I.png)



## Knowledge Distillation
通過訓練好的大模型得到更加適合推理的小模型
![](https://i.imgur.com/tRa7rka.png)


![](https://i.imgur.com/u3EoEE4.png)

![](https://i.imgur.com/0EoLCws.png)

## Mixup Trainning 
> Ramdonly sample two example to form a new example 
> (Xi,Yi) and (Xj,Yj)
![](https://i.imgur.com/ryV8CMN.png)

## Result
![](https://i.imgur.com/JFc0DiY.png)







