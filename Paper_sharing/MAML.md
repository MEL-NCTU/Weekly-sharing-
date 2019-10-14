Meta Learning (MAML) : Learning to learn
===
[TOC]

## Introduction
**Common Types of Meta-learning**
- Learning good **weight initializations** (MAML)
- Meta-models that **generate the parameters** of other models
- Learning **transferable optimizers** 

**MAML**

- Paper : https://arxiv.org/pdf/1703.03400.pdf
- Model-Agnostic Meta-Learning
    - Model Agnostic (Independent): You can adapt this method to any model



**What does MAML do?**
- Finding good weight initializations
- Fast adaption (good for few-shot learning)

- ~~automatically generates model structure~~

    


## Methods
---
### **One-shot/Few-shot Learning** ###
>
![](https://github.com/brendenlake/omniglot/blob/master/demo_strokes.png?raw=true)

- **Omniglot Dateset**

    - Human 
        - Learn very quickly
        - Few Examples
    - Approches
        - Transfer Learning
        - Meta-Learning : Learn to learn


- **N way K shot**
    - N way : N classes
    - K shot : K examples
- **Task**
    - Different dateset
    - Assign a task to evaluate how well a network learns 

---
### **General Ideas of How Meta-Learning Works** ###

- Weight Initialization : Train from a good strating point
- Fewer steps to reach best parameters
![](https://github.com/MEL-NCTU/Weekly-sharing-/blob/master/Paper_sharing/Images/Meta_1.PNG?raw=true)

### Sine Wave Regression Experiment

- 700000 differnet sine waves with differnet phase and amplitude
- Meta-Learning v.s. Pretrained
![](https://github.com/MEL-NCTU/Weekly-sharing-/blob/master/Paper_sharing/Images/Meta_2.PNG?raw=true)

-Result
![](https://github.com/MEL-NCTU/Weekly-sharing-/blob/master/Paper_sharing/Images/Meta_3.PNG?raw=true)


## Conclusion & Current Status
![](https://github.com/MEL-NCTU/Weekly-sharing-/blob/master/Paper_sharing/Images/Meta_5.PNG?raw=true)

- only on small/and simple tasks


## Reference
- MAML Paper:
https://arxiv.org/pdf/1703.03400.pdf
- Youtube 
https://www.youtube.com/watch?v=wT45v8sIMDM
- 李弘毅課程(介紹MAML) : 
https://www.youtube.com/watch?v=EkAqYbpCYAc
- Meta-Learning 知乎:
https://zhuanlan.zhihu.com/p/57864886


:::info
**Jimmy Li** @NCTU
:::

###### tags: `Meta-Learning` `Learn to Learn` 
