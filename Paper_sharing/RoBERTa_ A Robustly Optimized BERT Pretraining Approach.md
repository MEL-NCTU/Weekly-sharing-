# **RoBERTa: A Robustly Optimized BERT Pretraining Approach** #

[TOC]

https://hackmd.io/@rACDTXr0Ty-ojW6DM-B35w/H1e6N8iuH
## Training Procedure Analysis
### 1.Dynamic Masking:
> 使用Dynamic Mask 取代 原模型的Static Mask
> ![](https://i.imgur.com/WH1w7k5.png)

> ![](https://i.imgur.com/7H4jxpN.png)

### 2.Model Input Format and Next Sentence Prediction
> In the original Bert model predict the mask's token and Next Sentence Prediction (NSP)
> The NSP loss was hypothesized to be an important factor in training the original BERT model.
> However, some recent work has questioned the necessity of the NSP loss
> 
> To better understand this discrepancy, we compare several alternative training formats

**1.SEGMENT-PAIR+NSP:input format used in BERT**

**2.SENTENCE-PAIR+NSP:a pair of natural sentences + large batch size**

**3.FULL-SENTENCES:full sentences input(can cross document boundaries) +remove NSP**

**4.DOC-SENTENCES:FULL-SENTENCES input can not cross boundaries**

![](https://i.imgur.com/Bd4lFXW.png)













### 3.Training with large batches
![](https://i.imgur.com/Dmnawph.png)

### 4.Text Encoding
> Byte-Pair Encoding (BPE) (Sennrich et al., 2016) is a hybrid between character and word level
主要用於壓縮數據
https://zh.wikipedia.org/wiki/%E5%AD%97%E8%8A%82%E5%AF%B9%E7%BC%96%E7%A0%81
使用byte-level BPE 取代原模型 character-level BPE 可增加模型的parameter

## RoBERTa
> RoBERTa is trained with dynamic masking
(Section 4.1), FULL-SENTENCES without NSP
loss (Section 4.2), large mini-batches (Section 4.3)
and a larger byte-level BPE (Section 4.4)
![](https://i.imgur.com/iSqV6Gc.png)
![](https://i.imgur.com/b2xLfHr.png)
![](https://i.imgur.com/GHq0IPq.png)
![](https://i.imgur.com/FCEpOBa.png)



# Code
## English
https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/huggingface_pytorch-transformers.ipynb

https://github.com/pytorch/fairseq/tree/master/examples/roberta

### Load RoBERTa from torch.hub
```
import torch
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
roberta.eval()  # disable dropout (or leave in train mode to finetune)
```
### Sentence-pair classification tasks
```
#Download RoBERTa already finetuned for MNLI
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
roberta.eval()  # disable dropout for evaluation

with torch.no_grad():
    # Encode a pair of sentences and make a prediction
    tokens = roberta.encode('Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.')
    prediction = roberta.predict('mnli', tokens).argmax().item()
    assert prediction == 0  # contradiction

    # Encode another pair of sentences
    tokens = roberta.encode('Roberta is a heavily optimized version of BERT.', 'Roberta is based on BERT.')
    prediction = roberta.predict('mnli', tokens).argmax().item()
    assert prediction == 2  # entailment    
```

### Question answering
```
question_answering_model = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', 'bert-large-uncased-whole-word-masking-finetuned-squad')
question_answering_tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-large-uncased-whole-word-masking-finetuned-squad')

# The format is paragraph first and then question
text_1 = "Jim Henson was a puppeteer"
text_2 = "Who was Jim Henson ?"
indexed_tokens = question_answering_tokenizer.encode(text_1, text_2, add_special_tokens=True)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
segments_tensors = torch.tensor([segments_ids])
tokens_tensor = torch.tensor([indexed_tokens])

# Predict the start and end positions logits
with torch.no_grad():
    start_logits, end_logits = question_answering_model(tokens_tensor, token_type_ids=segments_tensors)

# get the highest prediction
answer = question_answering_tokenizer.decode(indexed_tokens[torch.argmax(start_logits):torch.argmax(end_logits)+1])
assert answer == "puppeteer"

# Or get the total loss which is the sum of the CrossEntropy loss for the start and end token positions (set model to train mode before if used for training)
start_positions, end_positions = torch.tensor([12]), torch.tensor([14])
multiple_choice_loss = question_answering_model(tokens_tensor, token_type_ids=segments_tensors, start_positions=start_positions, end_positions=end_positions)
```

## Chinese
Chinese pretrained model:https://github.com/brightmart/roberta_zh


fastai bert (classify comment)
https://github.com/ZeroLeon/Chinese_htlData_Classification_with_BERT

### LCQMC(Sentence Pair Matching)



| model | Dev | Test |
| -------- | -------- | -------- |
| Bert     | 89.4(88.4)|86.9(86.4)|
| ERNIE	|89.8 (89.6)	|87.2 (87.0) |
|BERT-wwm	|89.4 (89.2)	|87.0 (86.8)|
|RoBERTa-zh-base|	88.7	|87.0|
|RoBERTa-zh-Large	|89.9(89.6)	|87.2(86.7)|
|RoBERTa-zh-Large(20w_steps)|	89.7	|87.0|

### Reading comprehension test

best parameters for bert and roberta : epoch2, batch=32, lr=3e-5, warmup=0.1

#### cmrc2018
| models | DEV |
| -------- | -------- | -------- |
|sibert_base|	F1:87.521(88.628)EM:67.381(69.152)|
|sialbert_middle|F1:87.6956(87.878) EM:67.897(68.624)|
|哈工大讯飞 roberta_wwm_ext_base	|F1:87.521(88.628) EM:67.381(69.152)|
|brightmart roberta_middle|F1:86.841(87.242) EM:67.195(68.313)|
|brightmart roberta_large|	F1:88.608(89.431) EM:69.935(72.538)|

#### DRCD
|models|	DEV|
| -------- | -------- | -------- |
|siBert_base	|F1:93.343(93.524) EM:87.968(88.28)|
|siALBert_middle|	F1:93.865(93.975) EM:88.723(88.961)|
|哈工大讯飞 roberta_wwm_ext_base	|F1:94.257(94.48) EM:89.291(89.642)|
|brightmart roberta_large	|F1:94.933(95.057) EM:90.113(90.238)|

#### CJRC (with yes,knoe,unknown)
|models|	DEV|
| -------- | -------- | -------- |
|siBert_base	|F1:80.714(81.14) EM:64.44(65.04)|
|siALBert_middle|	F1:80.9838(81.299) EM:63.796(64.202)|
|哈工大讯飞 roberta_wwm_ext_base|	F1:81.510(81.684) EM:64.924(65.574)|
|brightmart roberta_large	|F1:80.16(80.475) EM:65.249(66.133)|



