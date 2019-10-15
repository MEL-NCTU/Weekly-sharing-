# XLNet: Generalized Autoregressive Pretraining for Language Understanding

## Introduction

Unsupervised representation learning has been highly successful in the domain of natural language processing.
Typically, these methods ﬁrst pretrain neural networks on large-scale unlabeled text corpora, and then ﬁnetune the models or representations on downstream tasks.
Among them, autoregressive (AR) language modeling and autoencoding (AE) have been the two most successful pretraining objectives.
The team propose XLNet, a generalized autoregressive method that leverages the best of both AR language modeling and AE while avoiding their limitations.

- XLNet maximizes the expected log likelihood of a sequence w.r.t. all possible permutations of the factorization order. Thanks to the permutation operation, the context for each position can consist of tokens from both left and right. **In expectation, each position learns to utilize contextual information from all positions.**
- XLNet does not rely on data corruption. Hence, XLNet does **not suffer from the pretrain-ﬁnetune discrepancy that BERT is subject to.** Meanwhile, the autoregressive objective also provides a natural way to use the product rule for factorizing the joint probability of the predicted tokens, eliminating the independence assumption made in BERT.
- XLNet integrates the segment recurrence mechanism and relative encoding scheme of **Transformer-XL** into pretraining, which empirically **improves the performance especially for tasks involving a longer text sequence.**
-  Naively applying a Transformer(-XL) architecture to permutation-based language modeling does not work **because the factorization order is arbitrary and the target is ambiguous.** As a solution, we propose to reparameterize the Transformer(-XL) network to remove the ambiguity.

## Method

First, The pros and cons of the two pretraining objectives (AR & AE) are compared in the following aspects:

- Independence Assumption: BERT factorizes the joint conditional probability p(x¯ | xˆ) based on an independence assumption that all masked tokens x¯ are separately reconstructed. In comparison, the AR language modeling objective factorizes pθ(x) using the product rule that holds universally without such an independence assumption.
- Input noise: The input to BERT contains artiﬁcial symbols like [MASK] that never occur in downstream tasks, which creates a pretrain-ﬁnetune discrepancy. In comparison, AR language modeling does not rely on any input corruption and does not suffer from this issue.
- Context dependency: The AR representation hθ(x1:t−1) is only conditioned on the tokens up to position t (i.e. tokens to the left), while the BERT representation Hθ(x)t has access to the contextual information on both sides. As a result, the BERT objective allows the model to be pretrained to better capture bidirectional context.

###  Objective: Permutation Language Modeling

![](https://i.imgur.com/0mB1Fdf.png)

- The proposed objective only permutes the factorization order, not the sequence order. In other words, we keep the original sequence order, use the positional encodings corresponding to the original sequence, and rely on a proper attention mask in Transformers to achieve permutation of the factorization order.

### Architecture: Two-Stream Self-Attention for Target-Aware Representations

For this parameterization to work, there are two requirements that are contradictory in a standard Transformer architecture: (1) to predict the token Xzt, gθ(xz<t, zt) should only use the position zt and not the content Xzt, otherwise the objective becomes trivial; (2) to predict the other tokens Xzj with j > t, gθ(xz<t, zt) should also encode the content Xzt to provide full contextual information. To resolve such a contradiction, we propose to use two sets of hidden representations instead of one:

-  The content representation hθ(xz≤t), or abbreviated as hzt, which serves a similar role to the standard hidden states in Transformer. This representation encodes both the context and Xzt itself.
-  The query representation gθ(xz<t, zt), or abbreviated as gzt, which only has access to the contextual information xz<t and the position zt, but not the content Xzt

![](https://i.imgur.com/7BWo1T0.png)

![](https://i.imgur.com/w7s9rcR.png)

Where Q, K, V denote the query, key, and value in an attention operation. 
To reduce the optimization difﬁculty, we choose to only predict the last tokens in a factorization order.

###  Incorporating Ideas from Transformer-XL

We integrate two important techniques in Transformer-XL, namely the relative positional encoding scheme and the segment recurrence mechanism.

### Modeling Multiple Segments

During the pretraining phase, following BERT, we randomly sample two segments (either from the same context or not) and treat the concatenation of two segments as one sequence to perform permutation language modeling. We only reuse the memory that belongs to the same context. Speciﬁcally, the input to our model is similar to BERT: [A, SEP, B, SEP, CLS], where “SEP” and “CLS” are two special symbols and “A” and “B” are the two segments.  
#### Relative Segment Encodings  
   Architecturally, different from BERT that adds an absolute segment embedding to the word embedding at each position, we extend the idea of relative encodings from Transformer-XL to also encode the segments. Given a pair of positions i and j in the sequence, if i and j are from the same segment, we use a segment encoding sij = s+ or otherwise sij = s−, where s+ and s− are learnable model parameters for each attention head.  
   
   **There are two beneﬁts of using relative segment encodings.** First, the inductive bias of relative encodings improves generalization. Second, it opens the possibility of ﬁnetuning on tasks that have more than two input segments, which is not possible using absolute segment encodings.
   
   
## Result

![](https://i.imgur.com/cVP8kaK.png)

