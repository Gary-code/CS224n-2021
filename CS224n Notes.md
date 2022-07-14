# CS224n Notes

> Garygedegege



## Lecture 01: Introduction & Word2vec

### 1.1 Introduction

**Target：**

1. Basics first, then key **methods** used in NLP: Recurrent networks, attention, transformers, etc.

2. A big picture understanding of human languages and the difficulties in understanding and producing them.

3. An understanding of and ability to build systems (in PyTorch) for some of the **major  problems** in NLP: Word meaning, dependency parsing, machine translation, question answering.

**Assignments and Project**

![image-20220609194858670](https://s2.loli.net/2022/06/09/kcm9ji1tn6EUvJA.png)

### 1.2 WordVec

> Main question: How to represent the word into vector?
>
> [Word2vec (Mikolov et al. 2013)](https://arxiv.org/abs/1301.3781) is a framework for learning word vectors.

**Definition**

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

Gradient
$$
\begin{aligned}
&\mathcal{U}_{\text {new }} \leftarrow \mathcal{U}_{\text {old }}-\alpha \nabla_{\mathcal{U}} J \\
&\mathcal{V}_{\text {old }} \leftarrow \mathcal{V}_{\text {old }}-\alpha \nabla_{\mathcal{V}} J
\end{aligned}
$$



[Genism Package for word2vec](https://radimrehurek.com/gensim/models/word2vec.html)

* 但这个包实际上在深度学习中**不常用**

[Gensim word vector visualization notebook](https://www.kaggle.com/code/yixuanzhou94/gensim-word-vector-visualization/notebook)



## Lecture 02: Neural Classifier

### 2.1 Word Vectors

**Review**

![image-20220612211647182](https://s2.loli.net/2022/06/12/qDmvh4aLzwYNX75.png)

* $J(\theta)$ is a function of **all** windows in the corpus (often, billions!)
  * • So is $\nabla_{\theta}J(\theta)$ very expensive to compute
* Iteratively take gradients at each such **window** for SGD
* But in each window, we only have at most $2m + 1$ words, so $\nabla_{\theta} J_{t}(\theta)$ is very **sparse**!

$$
\nabla_{\theta} J_{t}(\theta)=\left[\begin{array}{l}
0 \\
\vdots \\
\nabla_{v_{\text {like }}} \\
\vdots \\
0 \\
\nabla_{u_{f}} \\
\vdots \\
\nabla_{\text {ulearning }} \\
\vdots
\end{array}\right] \in \mathbb{R}^{2 d V}
$$



* We might only update the word vectors that actually appear! 
* Solution: either you need sparse matrix update operations to **only update** certain rows of full embedding matrices $U$ and $V$, or you need to keep around a hash for word vectors.

**Word2vec algorithm family（Skip-grams）**

> If you have millions of word vectors and do distributed computing, it is important to **not have to send gigantic updates around**!
>
> * word vectors weill be **row vectors**

Why two vectors? 

*  Easier optimization. Average both at the end
*  But can implement the algorithm with just one vector per word … and it help

**Two model variants:**

1. **Skip-grams** (SG)

   Predict context (“outside”) words (position independent) given center word

2. Continuous Bag of Words (CBOW)

   Predict center word from (bag of) context words

We presented: **Skip-gram model** !!!



