---
layout: post
title:  "ADAM optimizer"
date:   2021-05-16 21:49:11 -0400
categories: ML general 
permalink: /:categories/:title:output_ext
uselatex: true
---
## tl;dr
ADAM is an extension of SGD, where the **effective** learning rate changes over the training. In fact, the learning rate is a function of the $$1^{st}$$ and $$2^{nd}$$ moments of the gradient. Recommended default settings:

$$\alpha = 0.001$$, $$\beta_1 = 0.9$$, $$\beta_2 = 0.999$$, $$\epsilon = 10^{-8}$$

## ADAM (ADAptive Moment estimation)
ADAM is an extension of the SGD (stochastic gradient descent) and it stands for **Adaptive Moment Estimation**. It was introduced in [this][kingma-ba-2015] paper (by Kingma & Ba).

Benefits:

- Straightforward to implement
- Computationally efficient
- Little memory requirements
- Invariant to diagonal rescale of the gradients
- Well suited for problems that are large in terms of data and/or parameters
- Appropriate for non-stationary objectives
- Appropriate for problems with very noisy/or sparse gradients
- Hyper-parameters have intuitive interpretation and typically require little tuning

In SGD, there is a single learning rate for all weights update; this learning does not change during training. ADAM has different learning rates for each weight and separately adapts them during learning.

> The method computes individual adaptive learning rates for different parameters from **estimates of first and second moments of the gradients**.

Before we see the ADAM algorithm, here is the typical gradient descent update formula (for comparison):

$$ \theta_t \leftarrow \theta_{t-1} - \gamma \nabla_{\theta} f_t(\theta_{t-1})$$

where:

- $$f_t$$ is the objective function
- $$g_t = \nabla_\theta f_t(\theta)$$  is the vector of partial derivatives of $$f_t$$ w.r.t $$\theta$$ evaluated at timestep $$t$$.

In ADAM, the update formula is slightly different. The ADAM algorithm is as follows:

![ADAM algorithm]({{ 'assets/images/adam_algo.png' | relative_url }})

*Fig 1. ADAM algorithm (Image source: [Kingma, Ba (2015)](https://arxiv.org/pdf/1412.6980.pdf))*

Let's take a quick look at $$m_t$$ (biased first moment estimate):

$$m_0 = 0 \\ m_1 = (1-\beta_1) g_1 \\ m_2 = (1-\beta_1)\beta_1 g_1 + (1-\beta_1)g_2 \\ ... \\ m_N = (1-\beta_1)\beta_1^{N-1} g_1 ... + (1-\beta_1) g_N$$

- The "influence" of each $$g_i$$ decays at an exponential rate $$\beta_1$$
- Each update for $$m_t$$ is a weighted sum $$(\beta_1, 1-\beta_1)$$ of the previous value, $$m_{t-1}$$, and the current gradient value.

Let's take a quick look at $$v_t$$ (biased second raw moment estimate):

- Similar to $$m_t$$, The "influence" of each $$g_i^2$$ decays at an exponential rate $$\beta_2$$
- Similar to $$m_t$$, each update is a weighted sum of the previous value and the current second moment value
- Smaller/Bigger $$g_i^2$$ values ⇒ Higher/lower effective learning rates (due to $$\frac{1}{\sqrt{\hat{v}_t}+\epsilon}$$)

Let's take a quick look at $$\epsilon$$: well, there is not much to say , it exists to avoid division-by-zero cases.

![ADAM comparison]({{ '/assets/images/adam_comparison.png' | relative_url }})

*Fig 2. Training of multilayer neural networks on MNIST images (Image source: [Kingma, Ba (2015)](https://arxiv.org/pdf/1412.6980.pdf))*


**Important!** Note some **good default** settings found by the authors:

| Parameter    | Default Value |
|--------------|:-------------:|
| $$\alpha$$   | 0.001         |
| $$\beta_1$$  | 0.9           |
| $$\beta_2$$  | 0.999         |
| $$\epsilon$$ | $$10^{-8}$$   |

[kingma-ba-2015]: https://arxiv.org/pdf/1412.6980.pdf 

## ADADELTA (bonus)
Borrowing from the terminology/notation used for ADAM optimizer above. [Adadelta](https://arxiv.org/pdf/1212.5701.pdf) is simply:

$$\theta_t \leftarrow \theta_{t-1} - \alpha g_t / (\sqrt{\hat{v}_t} + \epsilon)$$