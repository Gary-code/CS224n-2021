# Lecture 02: Neural Network

## 2.1 Word Vectors

### Review

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

###  Word2vec algorithm family（Skip-grams）

> If you have millions of word vectors and do distributed computing, it is important to **not have to send gigantic updates around**!
>
> * word vectors weill be **row vectors**

Why two vectors? 

*  Easier optimization. Average both at the end
* But can implement the algorithm with just one vector per word … and it help

**Two model variants:**

1. **Skip-grams** (SG)

   Predict context (“outside”) words (position independent) given center word

2. Continuous Bag of Words (CBOW)

   Predict center word from (bag of) context words

We presented: **Skip-gram model** !!!

