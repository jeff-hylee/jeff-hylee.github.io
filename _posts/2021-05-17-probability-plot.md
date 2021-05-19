---
layout: post
title:  "Probability plot"
date:   2021-05-17 23:44:00 -0400
categories: data
permalink: /:categories/:title:output_ext
uselatex: true
---
Before diving into probability and quantile plots, we need to know about the concept of **empirical quantiles**. Empirical quantiles are similar to percentiles for a given data set. This is easier understood by looking at an example:

 Given a data set of $$n = 5$$, we first sort the data set and then estimate their (sorted) empirical quantiles as 0.2, 0.4, 0.6, 0.8, and 1.0 ($$p_i = i/n$$). Alternatively, we could have used a slighty different formula for $$p_i$$:

$$p_i = \frac{i-\alpha}{n+1-\alpha-\beta}$$

where $$\alpha$$ and $$\beta$$ are usually 0.4.

Now for the actual probability plot:

Let us assume that we have a set of numbers $$\{x_1, x_2, ..., x_n\}$$ and we wish to visually study whether the normality assumption is reasonable. The basic method is:

1. Sort the $$x_i$$-s from smallest to largest. Represent the sorted set of numbers as $$x(1), x(2), ..., x(n)$$. Hence, $$x(1)$$ is the minimum and $$x(n)$$ is the maximum of this data.
2. Define $$n$$ empirical quantiles
3. Find a set of numbers, $$z_1, z_2, ..., z_n$$ that would be expected from data that exactly follows the normal distribution. In our example, $$z_2$$ is the number that we would expect if we obtained $$5$$ values from a normal distribution, sorted them, and selected the second from the lowest. These are called the quantiles.
4. Construct a scatter plot with the pairs $$(x(1), z_1)$$, $$(x(2), z_2)$$, and so on. If the $$x_i$$-s came from a normal distribution, we would anticipate that the plotted points will roughly fall along a **straight line**. The degree of non-normality is suggested by the amount of curvature in the plot

If the distribution of observed matches the theoretical distribution, then the resulting plot would be a straight line.

Here is a simplified python implementation of the above:

```python
from scipy import stats 

#1. Sort values
data = data.sort_values()
#2. Compute empirical quantiles
empirical_quantiles = stats.mstats.plotting_positions(data)
#3. Compute theoretical quantiles - ppf is the inverse CDF
theoretical_quantiles = stats.norm.ppf(empirical_quantiles)
#4. Plot data vs theoretical quantiles
plt.scatter(theoretical_quantiles, data)
```

![Probability plot]({{ 'assets/images/prob_plot.png' | relative_url }})

*Fig 1. Probability plot for house prices from kaggle dataset*

**Some useful references**

[https://matplotlib.org/mpl-probscale/tutorial/closer_look_at_viz.html](https://matplotlib.org/mpl-probscale/tutorial/closer_look_at_viz.html)

[https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Probability_Plots.pdf](https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Probability_Plots.pdf)

