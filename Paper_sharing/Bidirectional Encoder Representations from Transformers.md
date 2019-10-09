# Bidirectional Encoder Representations from Transformers

## Unsupervised Feature-based Approaches
* onehotencoding
* word embedding
* contextualized word embedding
    * ELMO
    extract context-sensitive features from a left-to-right and a right-to-left language model
![](https://i.imgur.com/QvFS4Q8.png)



## BERT
* BERT takes advantages of multiple models
    * predict word from given context - Word2Vec CBOW
    * 2-layer birdirectional model - ELMo
    * Transformer instead of RNN - GPT(Generative Pre-training)
    > the BERT Transformer uses bidirectional self-attention
    > the GPT Transformer uses constrained self-attention where every token can only attend to context to its left.

## There are two steps in our framework
![](https://i.imgur.com/PHYcdJK.png)

* pre-training
        The model is trained on unlabeled data over different pre-training tasks.



* fine-tuning
    The pre-trained parameters, and all of the parameters are fine-tuned using labeled data from the downstream tasks.
    
## Pre-training BERT
* **Masked LM**
    ![](https://i.imgur.com/D7VmX2Y.png)

> 80%：替換為mask
10%：隨機替換為其它詞彙
10%：保留原來的詞彙。這部分正確的保留，保證了語言能力。
由於Transformer不知道要預測哪個詞語，所以它會強制學習到所有單詞的上下文表達。


* **Next Sentence Prediction (NSP)**
    when choosing the sentences A and B for each pretraining example, 50% of the time B is the actual next sentence that follows A (labeled as IsNext), and 50% of the time it is a random sentence from the corpus (labeled as NotNext).
    ![](https://i.imgur.com/sYIZEji.png)
> 輸入是A和B兩個句子，標記是IsNext或NotNext，用來判斷B是否是A後面的句子。這樣，就能從大規模預料中學習到一些句間關係。


## Model Architecture
* BERTBASE (L=12, H=768, A=12, Total Parameters=110M)
* BERTLARGE (L=24, H=1024, A=16, Total Parameters=340M)

L: the number of layers
H: the hidden size
A: the number of self-attention heads

## Fine-tuning
### SQuAD
![](https://i.imgur.com/0pJVINL.png)
##
* 2020「科技大擂台 與AI對話」(Fun Cup)的範例
![](https://i.imgur.com/UnaQ5cf.png)
##
![](https://i.imgur.com/EyFMoeg.png)




    