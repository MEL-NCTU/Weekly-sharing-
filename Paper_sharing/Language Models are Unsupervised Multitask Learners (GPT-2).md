# Language Models are Unsupervised Multitask Learners (GPT-2)

## Introduction 
---
* #### Demonstrate that language models begin to learn these tasks without any explicit supervision when trained on a new dataset of millions of webpages called WebText.
* #### Decoder block only in transformers
![](https://i.imgur.com/RTpOICR.jpg)

<br/>


## Architecture
---
* #### Layer normalization was moved to the input of each sub-block.
* #### An additional layer normalization was added after the ﬁnal self-attention block.

### Decoder Block
![](https://i.imgur.com/QYAtuAj.jpg)
### Masked Self-Attention
![](https://i.imgur.com/6lMs3xt.jpg)
### Input Encoding
The model looks up the embedding of the input word in its embedding matrix – one of the components we get as part of a trained model.Before handing that to the first block in the model, we need to incorporate positional encoding – a signal that indicates the order of the words in the sequence to the transformer blocks. Part of the trained model is a matrix that contains a positional encoding vector for each of the 1024 positions in the input.
![](https://i.imgur.com/aByLgQs.jpg)![](https://i.imgur.com/HFldHY4.jpg)
![](https://i.imgur.com/RvQUEmx.jpg)
<br/>

## Deeper Look in Architecture
---
* #### The first we encounter is the weight matrix that we use to create the queries, keys, and values.
![](https://i.imgur.com/cNN2Ky4.jpg)
<br/>
<br/>
* #### “Splitting” attention heads is simply reshaping the long vector into a matrix. The small GPT2 has 12 attention heads, so that would be the first dimension of the reshaped matrix.
![](https://i.imgur.com/5tHUDit.jpg)
![](https://i.imgur.com/la0wEQC.jpg)
<br/>
<br/>
* #### We multiply its query by all the other key vectors resulting in a score for each of the tokens.
![](https://i.imgur.com/ONoqXcj.jpg)
![](https://i.imgur.com/sjWUlSS.jpg)
![](https://i.imgur.com/Ea4tFmt.jpg)
<br/>
<br/>
* #### Here comes our second large weight matrix that projects the results of the attention heads into the output vector of the self-attention sublayer.And with this, we have produced the vector we can send along to the next layer.
![](https://i.imgur.com/bM7iwou.jpg)
<br/>
<br/>
* #### The fully-connected neural network is made up of two layers. The first layer is four times the size of the model (Since GPT2 small is 768, this network would have 768*4 = 3072 units).The second layer projects the result from the first layer back into model dimension (768 for the small GPT2). The result of this multiplication is the result of the transformer block for this token.
![](https://i.imgur.com/KnzNVAz.jpg)



<br/>

## Result
---
* ### Question Answer
![](https://i.imgur.com/G9rDO2A.jpg)

![](https://i.imgur.com/zV5I0Cn.jpg)
<br/>
* ### Different Dataset Result
![](https://i.imgur.com/1oov5zI.png)
<br/>

## Reference
---
* [http://jalammar.github.io/illustrated-gpt2/](https://)



