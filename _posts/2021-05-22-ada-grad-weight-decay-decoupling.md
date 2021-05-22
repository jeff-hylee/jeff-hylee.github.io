---
layout: post
title:  "Adaptive Gradient methods w/ decoupled weight-decay"
date:   2021-05-22 15:32:00 -0400
categories: ML general 
permalink: /:categories/:title:output_ext
uselatex: true
---
These are notes for the paper: [Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2019)](https://arxiv.org/pdf/1711.05101.pdf)

## tl;dr 
- L2 regularization and weight-decay are not equivalent for Adaptive gradient methods
- L2 regularization is not effective in ADAM
- Weight-decay is equally effective in both SGD and ADAM
- Optimal weight decay depends on the total number of batch passes/weight updates.
- Adam can substantially benefit from a scheduled learning rate multiplier.

In short, ADAM works better with **decoupled weight decay** than with just **L2 regularization**

## L2 norm regularization and weight-decay
While L2-norm regularization is equivalent to weight-decay in SGD, this is not true for Adaptive gradient methods.

> Intuitively, when Adam is run on a loss function $$f$$ plus L2 regularization, weights that tend to have large gradients in $$f$$ do not get regularized as much as they would with decoupled weight decay, since the gradient of the regularizer gets scaled along with the gradient off. This leads to an inequivalence of L2 and decoupled weight decay regularization for adaptive gradient algorithms

This is a driving motivation for us to find a formulation where L2 regularization and weight-decay are decoupled.

## Decoupling weight decay from gradient-based update

We start with the update formula with an explicit weight-decay term (Hanson & Pratt - need to fill in citation):

$$\mathbf{\theta}_{t+1} = (1-\lambda)\mathbf{\theta}_t - \alpha\nabla f_t(\theta_t)$$

where:

- $$\lambda$$: rate of the weight-decay per step
- $$\nabla f_t(\mathbf{\theta}_t)$$: $$t^{th}$$ batch gradient
- $$\alpha$$: learning rate

The paper has a proof that SGD with L2-norm regularization ($$f_t^{reg}(\mathbf{\theta}) = f_t(\mathbf{\theta}) + \frac{\lambda '}{2}\|\mathbf{\theta}\|^2_2$$) is directly equivalent to applying the above to a cost function without the L2-norm term.

### SGD - Decoupling L2 regularization and weight-decay

The authors propose the following **decoupling of L2 regularization and weight-decay for SGD**:

![SGD w/ decoupled weight-decay]({{ 'assets/images/sgd_decoupled_weight_decay.png' | relative_url }})

*Fig 1. SGD w/ decoupled weight-decay. Source: [Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2019)](https://arxiv.org/pdf/1711.05101.pdf)*

- In the calculation of $$g_t$$ (line 6), the pink term $$\lambda \theta_{t-1}$$ can be interpreted as the adjustment of the gradient if a L2 regularization term was present in the cost function.
- In the update of weights (line 9), the term $$\mathbf{m}_t$$ already incorporates L2 regularization and the green term $$-\eta_t \lambda \mathbf{\theta}_{t-1}$$ acts as the weight-decay

Obviously, it seems "unnecessary" to separate these terms for SGD, but this is used to generalize the same decoupling in ADAM (and other adaptive gradient methods). This was done as follows.

### ADAM - Decoupling L2 regularization and weight-decay (AKA: AdamW)

![ADAM w/ decoupled weight-decay]({{ 'assets/images/adam_decoupled_weight_decay.png' | relative_url }})

*Fig 2. ADAM w/ decoupled weight-decay. Source: [Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2019)](https://arxiv.org/pdf/1711.05101.pdf)*

Differently from the original ADAM formulation, the effects of the weight-decay term ($$\lambda \mathbf{\theta}_{t-1}$$ in the update equation) are not weakened (scaled) by the "adaptive"  gradient adjustment.

![ADAM vs ADAMW]({{ 'assets/images/adam_vs_adamw.png' | relative_url }})

*Fig 3. ADAM vs ADAMW. Image source: [Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2019)](https://arxiv.org/pdf/1711.05101.pdf)*

Note how decoupling weight-decay in ADAM results in significant differences in learning curves, especially for higher epochs. This is due to the weight-decay effect not being "weakened" by the adaptive gradient adjustment.

---
**note**

Now that we have L2 regularization and weight-decay decoupled, what would be the result of "playing" with the L2 regularization and, say, replacing it with L1-norm?

L1-norm regularization often results in sparse weight-vectors, so could this be potentially used for problems where we want to impose sparsity and weight-decay? 

---
