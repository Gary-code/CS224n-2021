# Lecture 01: Introduction & Word2vec



## Introduction

### Target

1. Basics first, then key **methods** used in NLP: Recurrent networks, attention, transformers, etc.

2. A big picture understanding of human languages and the difficulties in understanding and producing them.

3. An understanding of and ability to build systems (in PyTorch) for some of the **major  problems** in NLP: Word meaning, dependency parsing, machine translation, question answering.

### Assignments and Project

![image-20220609194858670](https://s2.loli.net/2022/06/09/kcm9ji1tn6EUvJA.png)



## WordVec

> Main question: How to represent the word into vector?
>
> [Word2vec (Mikolov et al. 2013)](https://arxiv.org/abs/1301.3781) is a framework for learning word vectors.

### Definition

![image-20220609195244586](https://s2.loli.net/2022/06/09/x8m29HSwMNOfjIy.png)

* Objective Function(Minimize the function)
  $$
  J(\theta)=-\frac{1}{T} \log L(\theta)=-\frac{1}{T} \sum_{t=1}^{T} \sum_{\substack{-m \leq j \leq m \\ j \neq 0}} \log P\left(w_{t+j} \mid w_{t} ; \theta\right)
  $$
  **How to calculate** $P(w_{t+j} \mid w_{t};\theta)$?

  * use two vectors per word $w$
    * $v_w$ when $w$ is a ==center word==
    * $u_w$ when $w$ is a ==context word==

  * Then for a center word $c$ and a context word $o$:

    * $$
      P(o \mid c)=\frac{\exp \left(u_{o}^{T} v_{c}\right)}{\sum_{w \in V} \exp \left(u_{w}^{T} v_{c}\right)}
      $$

  

  ![image-20220609195936885](https://s2.loli.net/2022/06/09/4V92XnBD5uNScfp.png)

**prediction function**

![image-20220609200429388](https://s2.loli.net/2022/06/09/IyS8Zct7h6qKRp5.png)

![image-20220609201213536](https://s2.loli.net/2022/06/09/sY1gt9kKyZzuQP4.png)

### Gradient

$$
\begin{aligned}
&\mathcal{U}_{\text {new }} \leftarrow \mathcal{U}_{\text {old }}-\alpha \nabla_{\mathcal{U}} J \\
&\mathcal{V}_{\text {old }} \leftarrow \mathcal{V}_{\text {old }}-\alpha \nabla_{\mathcal{V}} J
\end{aligned}
$$



[Genism Package for word2vec](https://radimrehurek.com/gensim/models/word2vec.html)

* 但这个包实际上在深度学习中**不常用**

[Gensim word vector visualization notebook](https://www.kaggle.com/code/yixuanzhou94/gensim-word-vector-visualization/notebook)



