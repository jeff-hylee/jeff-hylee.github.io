---
layout: post
title:  "Custom PyTorch Autograd"
date:   2021-05-21 00:18:00 -0400
categories: ML general 
permalink: /:categories/:title:output_ext
uselatex: true
---
Here is a simple/toy implementation of a custom PyTorch Function with autograd support.

## Recap: Reverse accumulation - Automatic Differentiation 
Just a short recap on reverse accumulation automatic differentiation. What follows below is modified excerpt from this [post](/ml/general/auto-diff.html).

Automatic differentiation is a technique to evaluate partial derivatives by "traversing" a calculation DAG. Traversing is in quotes, because obviously we are more than just traversing, we are actually iteratively applying the chain rule as we move along. 

The PyTorch autograd functionality is basically an implementation of the reverse accumulation version of auto-differentiation. Here is a short recap:

**reverse accumulation** ⇒ partial derivative of the whole expression with respect to each sub-expression

In reverse accumulation AD, we rely on computing **partial derivatives of the whole expression with respect to each sub-expression** over the computational graph. This means that each of the nodes in the computation graph holds the information: $$\bar{w_i} = \partial f/\partial{w_i}$$


![https://upload.wikimedia.org/wikipedia/commons/a/a0/ReverseaccumulationAD.png](https://upload.wikimedia.org/wikipedia/commons/a/a0/ReverseaccumulationAD.png)

*Fig 1. Reverse accumulation AD. Image source: [wikipedia](https://en.wikipedia.org/wiki/Automatic_differentiation)*

This means:

- $$\bar{w_5} = \frac{\partial f}{\partial w_5} =  1$$ (seed)
- $$\bar{w_4} = \frac{\partial f}{\partial{w_4}} = \bar{w_5} \frac{\partial w_5}{\partial w_4} = \bar{w_5}$$ 
- $$\bar{w_3} = \bar{w_5} \frac{\partial w_5}{\partial {w_3}} = \bar{w_5}$$ 
- $$\bar{w_2} = \bar{w_3}\frac{\partial w_3}{\partial w_2} = \bar{w_3} w_1$$ 
- $$\bar{w_1} = \bar{w_3}\frac{\partial w_3}{\partial w_1} = \bar{w_3} w_2$$ 


## Implementing the function $$f(a,b) = a * b$$ in PyTorch.

In PyTorch, at the most basic level, if we want to implement the custom function $$f$$, we need to write at least 2 methods:

- `forward(ctx, *args, **kwargs)`: calculates the result of applying $$f$$ to inputs ($$a * b$$)
- `backward(ctx, *grad_outputs)`: calculates partial derivatives w.r.t. inputs of $$f$$ (in this case $$\frac{\partial J}{\partial a}$$ and $$\frac{\partial J}{\partial b}$$, where $$J$$ is the whole expression - most of the time, it's the cost function)

If the above is not clear. Here is more documentation from PyTorch on `Function.backward()`:

> Each argument is the gradient w.r.t the given output, and each returned value should be the gradient w.r.t. the corresponding input.

In the case of $$f$$, `backward()`:
- accepts $$\frac{\partial J}{\partial f}$$ as input
- outputs
  - $$\frac{\partial J}{\partial a} = \frac{\partial J}{\partial f} \frac{\partial f}{\partial a} = \frac{\partial J}{\partial f} b$$ 
  - $$\frac{\partial J}{\partial b} = \frac{\partial J}{\partial f} \frac{\partial f}{\partial b} = \frac{\partial J}{\partial f} a$$ 

That is:

```python
class f(torch.autograd.Function):

     @staticmethod
     def forward(ctx, a, b):
         # as the name suggests: ctx.save_for_backward() allows us to save some info
         # which can be accessed later during backward()
         ctx.save_for_backward(b, a)
         return a * b

     @staticmethod
     def backward(ctx, grad_output):
         b, a = ctx.saved_tensors
         return b * grad_output, a * grad_output
```

Finally, to use `f`, we can simply use `f.apply()`.

### Comparing the "native" multiplication with our custom `f`

![Torch native `*` operator]({{ 'assets/images/torch_autograd_native_multi.png' | relative_url }})

*Fig 2. Torch native `*` operator.*

![Torch custom `*` operator]({{ 'assets/images/torch_autograd_custom_multi.png' | relative_url }})

*Fig 3. Torch custom `*` operator.*

Phew! They match!

---
**note**

When calculating `backward()`, we need to find the partial derivative of `f` wrt to its inputs (i.e.: $$ \frac{\partial f}{\partial a}$$). I imagine there could be situations where this is hard.

**Q** What would happen if we implement that derivative using the approximate value given by [numerical differentiation](https://mathworld.wolfram.com/NumericalDifferentiation.html)?

$$ \frac{\partial f}{\partial a} \approx \frac{f(a+\epsilon)-f(a)}{\epsilon} $$ 

---
